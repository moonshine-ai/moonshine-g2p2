#!/usr/bin/env python3
"""
Build OOV G2P training JSON from a pronouncing dictionary only (single words).

One example per ``word<TAB>ipa`` row. ``char`` is the **word alone** (no sentence
context); phones are eSpeak IPA tokens for that isolated surface.

Train/validation split is **by word key** (normalized lookup key) so the same
grapheme key never appears in both splits.

Writes ``oov_train.json`` and ``oov_valid.json`` under ``--out-dir``, plus
``build_metadata.json``.

Requires ``libespeak-ng`` and ``pip install espeak-phonemizer``.

Example::

    python scripts/build_oov_espeak_dataset.py --out-dir data/en_us/oov-training
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cmudict_ipa import CmudictIpa, normalize_word_for_lookup

from heteronym.espeak_heteronyms import EspeakPhonemizer
from oov.espeak_extract import extract_span_espeak_phonemes

logger = logging.getLogger(__name__)


def _json_row(word: str, phones: list[str], source: str) -> dict[str, Any]:
    return {
        "char": word,
        "word_char_start": 0,
        "word_char_end": len(word),
        "phonemes": phones,
        "source": source,
    }


def _assign_dict_examples_by_word(
    rows: list[tuple[str, str]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    by_word: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for w, ipa in rows:
        key = normalize_word_for_lookup(w) or w.lower()
        by_word[key].append((w, ipa))
    keys = sorted(by_word.keys())
    rng = __import__("random").Random(seed)
    rng.shuffle(keys)
    n_val = int(len(keys) * val_fraction)
    val_keys = set(keys[:n_val])
    train_r: list[tuple[str, str]] = []
    valid_r: list[tuple[str, str]] = []
    for k in keys:
        bucket = by_word[k]
        if k in val_keys:
            valid_r.extend(bucket)
        else:
            train_r.extend(bucket)
    return train_r, valid_r


_worker_phonemizer: EspeakPhonemizer | None = None
_worker_voice: str = "en-us"


def _mp_init(voice: str) -> None:
    global _worker_phonemizer, _worker_voice
    _worker_phonemizer = EspeakPhonemizer(default_voice=voice)
    _worker_voice = voice


def _mp_dict_word(payload: tuple[str, str]) -> dict[str, Any] | None:
    word, split = payload
    assert _worker_phonemizer is not None
    phones = extract_span_espeak_phonemes(
        _worker_phonemizer,
        word,
        _worker_voice,
        0,
        len(word),
    )
    if not phones:
        return None
    return _json_row(word, phones, f"dict_{split}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/en_us/oov-training"),
    )
    p.add_argument(
        "--dict-path",
        type=Path,
        default=Path("data/en_us/dict.tsv"),
        help="TSV word<TAB>ipa (CMU-style)",
    )
    p.add_argument("--val-fraction", type=float, default=0.02, help="fraction of dict keys in valid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--voice", type=str, default="en-us")
    p.add_argument("--max-dict-words", type=int, default=0, help="0 = all rows")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    dict_path = args.dict_path.resolve()
    if not dict_path.is_file():
        raise SystemExit(f"Missing dictionary TSV: {dict_path}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cmu = CmudictIpa(dict_path)
    rows: list[tuple[str, str]] = []
    for w, ipa in cmu.iter_pronunciation_rows():
        rows.append((w, ipa))
    if args.max_dict_words and len(rows) > args.max_dict_words:
        rng = __import__("random").Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.max_dict_words]

    train_rows, valid_rows = _assign_dict_examples_by_word(
        rows,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    stats: dict[str, Any] = {
        "dict_rows_total": len(rows),
        "dict_train_rows": len(train_rows),
        "dict_valid_rows": len(valid_rows),
        "dict_train_emitted": 0,
        "dict_valid_emitted": 0,
        "dict_espeak_failed": 0,
    }

    train_blob: dict[str, Any] = {}
    valid_blob: dict[str, Any] = {}
    nid = 0

    def add_train(obj: dict[str, Any]) -> None:
        nonlocal nid
        train_blob[str(nid)] = obj
        nid += 1

    def add_valid(obj: dict[str, Any]) -> None:
        nonlocal nid
        valid_blob[str(nid)] = obj
        nid += 1

    workers = max(1, int(args.workers))

    if workers == 1:
        try:
            phon = EspeakPhonemizer(default_voice=args.voice)
        except OSError as e:
            raise SystemExit(
                "Could not load libespeak-ng (needed by espeak-phonemizer).\n"
                f"Original error: {e}"
            ) from e
        for w, _ipa in tqdm(train_rows, desc="dict train"):
            phones = extract_span_espeak_phonemes(phon, w, args.voice, 0, len(w))
            if not phones:
                stats["dict_espeak_failed"] += 1
                continue
            add_train(_json_row(w, phones, "dict_train"))
            stats["dict_train_emitted"] += 1
        for w, _ipa in tqdm(valid_rows, desc="dict valid"):
            phones = extract_span_espeak_phonemes(phon, w, args.voice, 0, len(w))
            if not phones:
                stats["dict_espeak_failed"] += 1
                continue
            add_valid(_json_row(w, phones, "dict_valid"))
            stats["dict_valid_emitted"] += 1
    else:
        try:
            EspeakPhonemizer(default_voice=args.voice)
        except OSError as e:
            raise SystemExit(
                "Could not load libespeak-ng (needed by espeak-phonemizer).\n"
                f"Original error: {e}"
            ) from e
        dict_payloads = [(w, "train") for w, _ in train_rows] + [(w, "valid") for w, _ in valid_rows]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_mp_init,
            initargs=(args.voice,),
        ) as ex:
            futs = {ex.submit(_mp_dict_word, p): p for p in dict_payloads}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="dict (workers)"):
                p = futs[fut]
                word, split = p
                try:
                    row = fut.result()
                except Exception as e:
                    logger.warning("dict worker error for %r: %s", word, e)
                    stats["dict_espeak_failed"] += 1
                    continue
                if row is None:
                    stats["dict_espeak_failed"] += 1
                    continue
                if split == "train":
                    add_train(row)
                    stats["dict_train_emitted"] += 1
                else:
                    add_valid(row)
                    stats["dict_valid_emitted"] += 1

    train_path = out_dir / "oov_train.json"
    valid_path = out_dir / "oov_valid.json"
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_blob, f, ensure_ascii=False)
    with valid_path.open("w", encoding="utf-8") as f:
        json.dump(valid_blob, f, ensure_ascii=False)

    meta = {
        "train_json": str(train_path),
        "valid_json": str(valid_path),
        "stats": stats,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "voice": args.voice,
    }
    with (out_dir / "build_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "wrote %s (%d examples), %s (%d examples)",
        train_path,
        len(train_blob),
        valid_path,
        len(valid_blob),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
