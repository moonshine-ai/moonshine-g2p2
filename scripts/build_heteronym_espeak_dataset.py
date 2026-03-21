#!/usr/bin/env python3
"""
Build a synthetic heteronym-disambiguation JSON corpus using eSpeak NG + CMUdict.

For each source sentence that contains at least one token with multiple CMUdict
IPA readings, runs eSpeak twice (full text vs. text with that token removed),
diffs tokenized IPA output to recover the pronunciation eSpeak chose for the
token, and if it matches one of the dictionary alternatives, emits a LibriG2P-
compatible record (char span + homograph_wordid = that IPA string).

Phoneme extraction uses the ``espeak-phonemizer`` PyPI package (ctypes bindings
to ``libespeak-ng``), not a subprocess to the ``espeak-ng`` executable.

Example::

    python scripts/build_heteronym_espeak_dataset.py \\
        --out-dir data/en_us/heteronym-training \\
        --max-sentences 100000

Requires ``libespeak-ng`` on the system and::

    pip install espeak-phonemizer datasets

Then train with LibriG2P valid only, e.g.::

    python train_heteronym.py --train-json data/en_us/heteronym-training/homograph_train.json \\
        --valid-json <path-to-homograph_valid.json>

CMUdict lists multiple pronunciations for many function words (``the``, ``a``, …). By default
those keys are skipped via ``--ignore-homograph-keys`` so the corpus skews toward content
words; pass ``--ignore-homograph-keys ''`` to keep them.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from espeak_phonemizer import Phonemizer as EspeakPhonemizer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Missing package `espeak-phonemizer` (ctypes bindings to libespeak-ng). "
        "Install with: pip install espeak-phonemizer"
    ) from e

from cmudict_ipa import CmudictIpa, normalize_word_for_lookup

logger = logging.getLogger(__name__)

_SENT_BOUND = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\S+")


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


def iter_wikitext_sentences(
    *,
    max_sentences: int,
    max_chars: int,
    split: str = "train",
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Wikitext mode needs the `datasets` package. "
            "Install with: pip install datasets\n"
            f"Original error: {e}"
        ) from e

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    n = 0
    for row in ds:
        if n >= max_sentences:
            break
        for sent in _rough_sentences(row["text"], max_chars=max_chars):
            if n >= max_sentences:
                break
            yield sent
            n += 1


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


def espeak_ipa_tokens(
    phonemizer: EspeakPhonemizer,
    text: str,
    *,
    voice: str,
) -> list[str]:
    """
    IPA phone tokens via ``espeak_phonemizer`` (same tokenization as
    ``espeak-ng --ipa`` with a space phoneme separator).
    """
    t = text.strip()
    if not t:
        return []
    try:
        raw = phonemizer.phonemize(
            t,
            voice=voice,
            phoneme_separator=" ",
            word_separator=" ",
        ).strip()
    except (AssertionError, OSError):
        return []
    if not raw:
        return []
    return [x for x in raw.split() if x]


def longest_insert_block(tokens_without: list[str], tokens_full: list[str]) -> list[str]:
    sm = SequenceMatcher(a=tokens_without, b=tokens_full, autojunk=False)
    best: list[str] = []
    best_len = 0
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "insert" and (j2 - j1) > best_len:
            best = tokens_full[j1:j2]
            best_len = j2 - j1
    return best


def normalize_ipa_compare(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def match_dictionary_alternative(extracted_joined: str, alts: list[str]) -> str | None:
    """
    Map eSpeak IPA string to one of the CMUdict IPA strings (canonical spelling
    from ``alts``). Tries exact NFC match, then match with length mark ``ː``
    stripped on both sides (eSpeak often uses long vowels where CMU does not).
    """
    e0 = normalize_ipa_compare(extracted_joined)
    for alt in alts:
        if normalize_ipa_compare(alt) == e0:
            return alt
    e1 = e0.replace("ː", "")
    for alt in alts:
        if normalize_ipa_compare(alt).replace("ː", "") == e1:
            return alt
    return None


@dataclass
class ExtractedExample:
    char_text: str
    homograph: str
    homograph_wordid: str
    homograph_char_start: int
    homograph_char_end: int


def extract_examples_for_sentence(
    text: str,
    *,
    cmudict: CmudictIpa,
    phonemizer: EspeakPhonemizer,
    voice: str,
    max_candidates: int,
    ignore_keys: frozenset[str],
) -> list[ExtractedExample]:
    if not text.strip():
        return []

    words = list(_WORD_RE.finditer(text))
    ambiguous_spans: list[tuple[re.Match[str], list[str]]] = []
    for m in words:
        key = normalize_word_for_lookup(m.group())
        if not key or key in ignore_keys:
            continue
        alts = cmudict.translate_to_ipa([m.group()])[0][1]
        if len(alts) < 2:
            continue
        if len(alts) > max_candidates:
            continue
        ambiguous_spans.append((m, list(alts)))

    if not ambiguous_spans:
        return []

    tokens_full = espeak_ipa_tokens(phonemizer, text, voice=voice)
    if not tokens_full:
        return []

    out: list[ExtractedExample] = []
    for m, alts in ambiguous_spans:
        removed = text[: m.start()] + text[m.end() :]
        removed = re.sub(r"  +", " ", removed)
        tokens_without = espeak_ipa_tokens(phonemizer, removed, voice=voice)
        if not tokens_without:
            continue
        block = longest_insert_block(tokens_without, tokens_full)
        if not block:
            continue
        joined = "".join(block)
        wordid = match_dictionary_alternative(joined, alts)
        if wordid is None:
            continue
        out.append(
            ExtractedExample(
                char_text=text,
                homograph=m.group(),
                homograph_wordid=wordid,
                homograph_char_start=m.start(),
                homograph_char_end=m.end(),
            )
        )
    return out


# --- multiprocessing worker state ---

_worker_cmudict: CmudictIpa | None = None
_worker_phonemizer: EspeakPhonemizer | None = None
_worker_voice: str = "en-us"
_mp_ignore_keys: frozenset[str] = frozenset()


def _mp_init(dict_path: str, voice: str, ignore_keys_tuple: tuple[str, ...]) -> None:
    global _worker_cmudict, _worker_phonemizer, _worker_voice, _mp_ignore_keys
    _worker_cmudict = CmudictIpa(dict_path)
    _worker_phonemizer = EspeakPhonemizer(default_voice=voice)
    _worker_voice = voice
    _mp_ignore_keys = frozenset(ignore_keys_tuple)


def _mp_process_sentence(payload: tuple[str, int]) -> list[dict[str, Any]]:
    text, max_cand = payload
    assert _worker_cmudict is not None and _worker_phonemizer is not None
    rows = extract_examples_for_sentence(
        text,
        cmudict=_worker_cmudict,
        phonemizer=_worker_phonemizer,
        voice=_worker_voice,
        max_candidates=max_cand,
        ignore_keys=_mp_ignore_keys,
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


def _parse_ignore_keys(s: str) -> frozenset[str]:
    if not s.strip():
        return frozenset()
    return frozenset(x.strip().lower() for x in s.split(",") if x.strip())


def _sentence_has_ambiguous_token(
    text: str,
    cmudict: CmudictIpa,
    max_candidates: int,
    ignore_keys: frozenset[str],
) -> bool:
    for m in _WORD_RE.finditer(text):
        key = normalize_word_for_lookup(m.group())
        if not key or key in ignore_keys:
            continue
        alts = cmudict.translate_to_ipa([m.group()])[0][1]
        if 1 < len(alts) <= max_candidates:
            return True
    return False


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
        default=Path("data/en_us/dict.tsv"),
        help="CMUdict TSV (word<TAB>ipa) used for ambiguous-word detection",
    )
    p.add_argument(
        "--source",
        choices=("wikitext", "lines"),
        default="wikitext",
        help="wikitext-103-raw train stream, or --lines-file",
    )
    p.add_argument(
        "--lines-file",
        type=Path,
        default=None,
        help="with --source lines: one sentence per line",
    )
    p.add_argument(
        "--wikitext-split",
        type=str,
        default="train",
        help="datasets split name for wikitext-103-raw-v1",
    )
    p.add_argument("--max-sentences", type=int, default=100_000, help="cap source sentences")
    p.add_argument(
        "--max-sentence-chars",
        type=int,
        default=384,
        help="skip longer sentences (and bound wikitext clause length)",
    )
    p.add_argument("--voice", type=str, default="en-us", help="eSpeak voice, e.g. en-us")
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
    p.add_argument(
        "--ignore-homograph-keys",
        type=str,
        default="the,a,to,of,and,or",
        help=(
            "comma-separated lowercase CMUdict keys to skip (reduces weak-form articles "
            "and similar); use '' to disable"
        ),
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
    ignore_keys = _parse_ignore_keys(args.ignore_homograph_keys)

    if args.source == "lines":
        if args.lines_file is None:
            raise SystemExit("--source lines requires --lines-file")
        if not args.lines_file.is_file():
            raise SystemExit(f"Missing --lines-file: {args.lines_file}")
        sentence_iter = iter_lines_file(
            args.lines_file,
            max_sentences=args.max_sentences,
            max_chars=args.max_sentence_chars,
        )
    else:
        sentence_iter = iter_wikitext_sentences(
            max_sentences=args.max_sentences,
            max_chars=args.max_sentence_chars,
            split=args.wikitext_split,
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
        for text in sentence_iter:
            stats["sentences_seen"] += 1
            if not _sentence_has_ambiguous_token(
                text, cmudict, args.max_candidates, ignore_keys
            ):
                continue
            stats["sentences_with_any_ambiguous_token"] += 1
            rows = extract_examples_for_sentence(
                text,
                cmudict=cmudict,
                phonemizer=phonemizer,
                voice=args.voice,
                max_candidates=args.max_candidates,
                ignore_keys=ignore_keys,
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
            if stats["sentences_seen"] % 2000 == 0:
                logger.info(
                    "progress sentences=%d records=%d",
                    stats["sentences_seen"],
                    stats["records_emitted"],
                )
    else:
        sentences_list = list(sentence_iter)
        stats["sentences_seen"] = len(sentences_list)
        cmudict = CmudictIpa(dict_path)
        payloads = [
            (t, args.max_candidates)
            for t in sentences_list
            if _sentence_has_ambiguous_token(t, cmudict, args.max_candidates, ignore_keys)
        ]
        stats["sentences_with_any_ambiguous_token"] = len(payloads)
        ignore_tuple = tuple(sorted(ignore_keys))
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_mp_init,
            initargs=(str(dict_path), args.voice, ignore_tuple),
        ) as ex:
            futures = [ex.submit(_mp_process_sentence, p) for p in payloads]
            for fut in as_completed(futures):
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
        "ignore_homograph_keys": sorted(ignore_keys),
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
