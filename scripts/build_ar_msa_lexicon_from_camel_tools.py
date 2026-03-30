#!/usr/bin/env python3
"""
Build ``data/ar_msa/dict.tsv`` entries (``undiac<TAB>ipa``) from Camel Tools disambiguation.

Requires ``pip install camel-tools`` and Calima MSA morphology + MLE data (install once with
``camel_data -i disambig-mle-calima-msa-r13``, which pulls ``morphology-db-msa-r13``).
Run from the repo root; output appends unique keys.

Uses ``camel_tools.disambig.mle.MLEDisambiguator`` (camel-tools 1.5+; older releases used
``disambig.mle_disambiguator``).

Example::

    python scripts/build_ar_msa_lexicon_from_camel_tools.py --wiki data/ar/wiki-text.txt --max-lines 5000
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arabic_ipa import strip_arabic_diacritics, word_to_ipa_with_assimilation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=Path, default=_REPO / "data" / "ar" / "wiki-text.txt")
    ap.add_argument("--out", type=Path, default=_REPO / "data" / "ar_msa" / "dict.tsv")
    ap.add_argument("--max-lines", type=int, default=2000)
    args = ap.parse_args()

    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.word import simple_word_tokenize

    disamb = MLEDisambiguator.pretrained()
    seen: set[str] = set()
    if args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if "\t" in line and not line.lstrip().startswith("#"):
                seen.add(line.split("\t", 1)[0].strip())

    n = 0
    out_lines: list[str] = []
    with args.wiki.open(encoding="utf-8") as f:
        for line in f:
            if n >= args.max_lines:
                break
            n += 1
            toks = simple_word_tokenize(line)
            if not toks:
                continue
            try:
                analyses_per_tok = disamb.disambiguate(toks)
            except Exception as e:
                print("disambiguate failed:", e, file=sys.stderr)
                continue
            for dw in analyses_per_tok:
                if not dw.analyses:
                    continue
                top = dw.analyses[0]
                diac = top.diac
                if not diac:
                    continue
                diac = unicodedata.normalize("NFC", diac)
                key = strip_arabic_diacritics(diac)
                if not key or key in seen or not any(0x0600 <= ord(c) <= 0x06FF for c in key):
                    continue
                ipa = word_to_ipa_with_assimilation(diac)
                if not ipa:
                    continue
                seen.add(key)
                out_lines.append(f"{key}\t{ipa}\n")

    with args.out.open("a", encoding="utf-8") as out:
        out.writelines(out_lines)
    print(f"Appended {len(out_lines)} lines to {args.out}")


if __name__ == "__main__":
    main()
