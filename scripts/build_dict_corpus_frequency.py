#!/usr/bin/env python3
"""
Count token frequencies in a line-based text corpus and attach them to dictionary rows.

Scans *corpus* one line at a time (one sentence or paragraph per line). Each whitespace
token is mapped with :func:`cmudict_ipa.normalize_word_for_lookup` (same as G2P lookup).
Total token count is the number of non-empty normalized tokens. For each row in *dict*
(``word<TAB>ipa``), writes ``word<TAB>ipa<TAB>frequency`` where *frequency* is
``count / total`` on [0.0, 1.0] (e.g. 0.1 means 10% of all corpus tokens).

Example::

    python scripts/build_dict_corpus_frequency.py \\
        --corpus data/en_us/wiki-text.txt \\
        --dict data/en_us/dict_filtered_heteronyms.tsv \\
        --out data/en_us/dict_frequency.tsv

Uses ``tqdm`` progress bars for the corpus scan and the dict write loop when installed.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cmudict_ipa import normalize_word_for_lookup

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[misc, assignment]


def _progress(it, *, desc: str, unit: str):
    """Wrap *it* with tqdm when available; otherwise return *it* unchanged."""
    if _tqdm is None:
        return it
    return _tqdm(it, desc=desc, unit=unit)


def count_corpus_tokens(corpus_path: Path) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    total = 0
    with open(corpus_path, encoding="utf-8", errors="replace") as f:
        for line in _progress(f, desc="Scan corpus", unit=" lines"):
            for tok in line.split():
                key = normalize_word_for_lookup(tok)
                if not key:
                    continue
                counts[key] += 1
                total += 1
    return counts, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/en_us/wiki-text.txt"),
        help="UTF-8 text file, one line per record; tokens are split on whitespace.",
    )
    p.add_argument(
        "--dict",
        type=Path,
        dest="dict_path",
        default=Path("data/en_us/dict_filtered_heteronyms.tsv"),
        help="TSV with word in column 1 and IPA in column 2 (``#`` comment lines skipped).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/en_us/dict_frequency.tsv"),
        help="Output TSV: word, ipa, frequency.",
    )
    args = p.parse_args()

    corpus_path = args.corpus.resolve()
    dict_path = args.dict_path.resolve()
    out_path = args.out.resolve()

    if not corpus_path.is_file():
        raise SystemExit(f"corpus not found: {corpus_path}")
    if not dict_path.is_file():
        raise SystemExit(f"dict not found: {dict_path}")

    counts, total = count_corpus_tokens(corpus_path)
    if total == 0:
        raise SystemExit(f"no countable tokens in corpus: {corpus_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with open(dict_path, encoding="utf-8", errors="replace") as fin, open(
        out_path, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in _progress(fin, desc="Write frequencies", unit=" rows"):
            raw = line.rstrip("\n\r")
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if "\t" not in raw:
                continue
            word, ipa = raw.split("\t", 1)
            word = word.strip()
            ipa = ipa.strip()
            if not word or not ipa:
                continue
            key = normalize_word_for_lookup(word)
            c = counts[key] if key else 0
            freq = c / total
            fout.write(f"{word}\t{ipa}\t{freq:.17g}\n")
            n_out += 1

    print(f"Wrote {n_out} rows to {out_path} (corpus tokens={total}, types={len(counts)})")


if __name__ == "__main__":
    main()
