#!/usr/bin/env python3
"""
Build a synthetic heteronym-disambiguation JSON corpus using eSpeak NG + CMUdict.

For each source sentence that contains at least one token with multiple CMUdict
IPA readings, runs eSpeak twice (full text vs. text with that token removed),
diffs word-level IPA output to recover the pronunciation eSpeak chose for the
token, and if it matches one of the dictionary alternatives, emits a LibriG2P-
compatible record (char span + homograph_wordid = that IPA string).

Phoneme extraction uses the ``espeak-phonemizer`` PyPI package (ctypes bindings
to ``libespeak-ng``), not a subprocess to the ``espeak-ng`` executable. Heteronym
detection and per-token IPA recovery are implemented in ``heteronym.espeak_heteronyms``
for reuse (e.g. evaluation scripts).

Example::

    python scripts/build_heteronym_espeak_dataset.py \\
        --out-dir data/en_us/heteronym-training \\
        --max-sentences 100000

Requires ``libespeak-ng`` on the system and::

    pip install espeak-phonemizer datasets

Then train with LibriG2P valid only, e.g.::

    python train_heteronym.py --train-json data/en_us/heteronym-training/homograph_train.json \\
        --valid-json <path-to-homograph_valid.json>

"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cmudict_ipa import CmudictIpa

from heteronym.espeak_heteronyms import (
    EspeakPhonemizer,
    extract_examples_for_sentence,
    sentence_has_ambiguous_heteronym,
)

logger = logging.getLogger(__name__)

_SENT_BOUND = re.compile(r"(?<=[.!?])\s+")


def _rough_sentences(paragraph: str, *, max_chars: int) -> list[str]:
    paragraph = paragraph.strip()
    if not paragraph or paragraph.startswith("="):
        return []
    chunks = _SENT_BOUND.split(paragraph)
    out: list[str] = []
    for c in chunks:
        c = c.strip()
        if not c or len(c) > max_chars:
            continue
        out.append(c)
    if not out and len(paragraph) <= max_chars:
        out.append(paragraph)
    return out


def iter_lines_file(path: Path, *, max_sentences: int, max_chars: int) -> Iterator[str]:
    n = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if n >= max_sentences:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if len(line) > max_chars:
                continue
            yield line
            n += 1


# --- multiprocessing worker state ---

_worker_cmudict: CmudictIpa | None = None
_worker_phonemizer: EspeakPhonemizer | None = None
_worker_voice: str = "en-us"


def _mp_init(dict_path: str, voice: str) -> None:
    global _worker_cmudict, _worker_phonemizer, _worker_voice
    _worker_cmudict = CmudictIpa(dict_path)
    _worker_phonemizer = EspeakPhonemizer(default_voice=voice)
    _worker_voice = voice


def _mp_process_sentence(payload: tuple[str, int]) -> list[dict[str, Any]]:
    text, max_cand = payload
    assert _worker_cmudict is not None and _worker_phonemizer is not None
    rows = extract_examples_for_sentence(
        text,
        cmudict=_worker_cmudict,
        phonemizer=_worker_phonemizer,
        voice=_worker_voice,
        max_candidates=max_cand,
    )
    return [
        {
            "char": r.char_text,
            "homograph": r.homograph,
            "homograph_wordid": r.homograph_wordid,
            "homograph_char_start": r.homograph_char_start,
            "homograph_char_end": r.homograph_char_end,
        }
        for r in rows
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/en_us/heteronym-training"),
        help="directory for homograph_train.json and metadata",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="homograph_train.json",
        help="filename under --out-dir",
    )
    p.add_argument(
        "--dict-path",
        type=Path,
        default=Path("data/en_us/dict_filtered_heteronyms.txt"),
        help="CMUdict TSV (word<TAB>ipa) used for ambiguous-word detection",
    )
    p.add_argument(
        "--lines-file",
        type=Path,
        default=Path("data/en_us/wiki-text.txt"),
        help="one sentence per line",
    )
    p.add_argument(
        "--max-sentences", type=int, default=1_000_000, help="cap source sentences"
    )
    p.add_argument(
        "--max-sentence-chars",
        type=int,
        default=384,
        help="skip longer sentences (and bound wikitext clause length)",
    )
    p.add_argument(
        "--voice", type=str, default="en-us", help="eSpeak voice, e.g. en-us"
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=4,
        help="skip homographs with more CMU pronunciations than this (matches train_heteronym)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel processes (each loads libespeak-ng; >1 speeds up on multi-core machines)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    dict_path = args.dict_path.resolve()
    if not dict_path.is_file():
        raise SystemExit(f"Missing CMUdict TSV: {dict_path}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / args.output_json

    if not args.lines_file.is_file():
        raise SystemExit(f"Missing --lines-file: {args.lines_file}")
    sentence_iter = iter_lines_file(
        args.lines_file,
        max_sentences=args.max_sentences,
        max_chars=args.max_sentence_chars,
    )

    stats = {
        "sentences_seen": 0,
        "sentences_with_any_ambiguous_token": 0,
        "records_emitted": 0,
        "sentences_espeak_failed": 0,
    }

    blob: dict[str, Any] = {}
    rid = 0

    if args.workers <= 1:
        cmudict = CmudictIpa(dict_path)
        try:
            phonemizer = EspeakPhonemizer(default_voice=args.voice)
        except OSError as e:
            raise SystemExit(
                "Could not load libespeak-ng (needed by espeak-phonemizer). "
                "Install the eSpeak NG library package for your OS "
                "(e.g. Debian/Ubuntu: libespeak-ng1).\n"
                f"Original error: {e}"
            ) from e
        for text in tqdm(
            sentence_iter,
            desc="Sentences",
            unit="sent",
            total=args.max_sentences,
        ):
            stats["sentences_seen"] += 1
            if not sentence_has_ambiguous_heteronym(
                text,
                cmudict=cmudict,
                max_candidates=args.max_candidates,
            ):
                continue
            stats["sentences_with_any_ambiguous_token"] += 1
            rows = extract_examples_for_sentence(
                text,
                cmudict=cmudict,
                phonemizer=phonemizer,
                voice=args.voice,
                max_candidates=args.max_candidates,
            )
            for r in rows:
                blob[f"es_{rid:07d}"] = {
                    "char": r.char_text,
                    "homograph": r.homograph,
                    "homograph_wordid": r.homograph_wordid,
                    "homograph_char_start": r.homograph_char_start,
                    "homograph_char_end": r.homograph_char_end,
                }
                rid += 1
                stats["records_emitted"] += 1
    else:
        sentences_list = list(
            tqdm(
                sentence_iter,
                desc="Loading lines",
                unit="line",
                total=args.max_sentences,
            )
        )
        stats["sentences_seen"] = len(sentences_list)
        cmudict = CmudictIpa(dict_path)
        payloads = [
            (t, args.max_candidates)
            for t in tqdm(
                sentences_list,
                desc="Filtering heteronyms",
                unit="sent",
            )
            if sentence_has_ambiguous_heteronym(
                t,
                cmudict=cmudict,
                max_candidates=args.max_candidates,
            )
        ]
        stats["sentences_with_any_ambiguous_token"] = len(payloads)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_mp_init,
            initargs=(str(dict_path), args.voice),
        ) as ex:
            futures = [ex.submit(_mp_process_sentence, p) for p in payloads]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="eSpeak",
                unit="sent",
            ):
                try:
                    recs = fut.result()
                except Exception as e:  # pragma: no cover
                    logger.warning("worker failed: %s", e)
                    stats["sentences_espeak_failed"] += 1
                    continue
                for rec in recs:
                    blob[f"es_{rid:07d}"] = rec
                    rid += 1
                stats["records_emitted"] += len(recs)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)

    meta_path = out_dir / "build_metadata.json"
    meta = {
        "out_json": str(out_json),
        "dict_path": str(dict_path),
        "voice": args.voice,
        "max_candidates": args.max_candidates,
        "stats": stats,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "wrote %s (%d records). metadata %s",
        out_json,
        stats["records_emitted"],
        meta_path,
    )


if __name__ == "__main__":
    main()
