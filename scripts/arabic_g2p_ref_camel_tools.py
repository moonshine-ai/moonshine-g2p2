#!/usr/bin/env python3
"""
High-quality **MSA** reference G2P using **CAMeL Tools** morphological disambiguation (Calima /
Camel Morph–derived databases) plus the same IPA rules as :mod:`arabic_ipa`.

Install::

    pip install camel-tools

On first use, Camel Tools downloads analyzer DBs into its cache (see Camel Morph project:
https://github.com/CAMeL-Lab/camel_morph).

This script is for **evaluation and regression** against :mod:`arabic_rule_g2p` (ONNX tashkīl).
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arabic_ipa import word_to_ipa_with_assimilation  # noqa: E402

_DISAMB = None


def _mle_disamb():
    global _DISAMB
    if _DISAMB is None:
        from camel_tools.disambig.mle import MLEDisambiguator

        _DISAMB = MLEDisambiguator.pretrained()
    return _DISAMB


def _disambiguate_line(line: str) -> str:
    from camel_tools.tokenizers.word import simple_word_tokenize

    disamb = _mle_disamb()
    toks = simple_word_tokenize(line.strip())
    if not toks:
        return ""
    analyses_per_tok = disamb.disambiguate(toks)
    out_words: list[str] = []
    for dw in analyses_per_tok:
        if not dw.analyses:
            continue
        diac = dw.analyses[0].diac
        if diac:
            out_words.append(str(diac))
    return " ".join(out_words)


def ref_line_to_ipa(line: str) -> str:
    line = unicodedata.normalize("NFC", line.strip())
    if not line:
        return ""
    diac_phrase = _disambiguate_line(line)
    ipa_parts: list[str] = []
    for w in diac_phrase.split():
        if not w:
            continue
        if not any(0x0600 <= ord(c) <= 0x06FF for c in w):
            continue
        ipa_parts.append(word_to_ipa_with_assimilation(w))
    return " ".join(ipa_parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="MSA G2P via Camel Tools + arabic_ipa.")
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--stdin", action="store_true")
    args = ap.parse_args()
    if args.stdin or not args.text:
        text = sys.stdin.read()
    else:
        text = args.text
    print(ref_line_to_ipa(text))


if __name__ == "__main__":
    main()
