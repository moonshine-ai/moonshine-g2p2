#!/usr/bin/env python3
"""
Modern Standard Arabic (MSA) G2P: ONNX diacritization + rule/lexicon IPA (vocoder-oriented).

**Dependencies:** ``onnxruntime``, ``numpy``, and repo :mod:`ko_roberta_wordpiece` only.

Morphological **reference** quality is provided by Camel Tools / Camel Morph in
``scripts/arabic_g2p_ref_camel_tools.py``; this module uses a fixed Arabert tashkīl ONNX bundle
(see ``scripts/export_arabic_msa_diacritizer_onnx.py``).

Optional lexicon ``data/ar_msa/dict.tsv``: ``undiacritized_word<TAB>ipa`` (first hit wins).
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import Dict, Mapping

from arabic_diac_onnx_infer import ArabicDiacOnnx
from arabic_ipa import apply_onnx_partial_postprocess, strip_arabic_diacritics, word_to_ipa_with_assimilation

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT = _REPO_ROOT / "data" / "ar_msa" / "dict.tsv"
_DEFAULT_ONNX_DIR = _REPO_ROOT / "data" / "ar_msa" / "arabertv02_tashkeel_fadel_onnx"


def _load_lex_first(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            continue
        w, ipa = line.split("\t", 1)
        w = w.strip()
        ipa = ipa.strip().split()[0] if ipa.strip() else ""
        if w and ipa and w not in out:
            out[w] = ipa
    return out


def _has_arabic(s: str) -> bool:
    return any(0x0600 <= ord(c) <= 0x06FF for c in s)


class ArabicRuleG2p:
    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        dict_path: Path | None = None,
    ):
        self._onnx = ArabicDiacOnnx(model_dir=model_dir)
        self._dict_path = Path(dict_path) if dict_path is not None else _DEFAULT_DICT
        self._lex = _load_lex_first(self._dict_path)

    def g2p_word(self, word: str) -> str:
        w = unicodedata.normalize("NFC", word.strip())
        if not w or not _has_arabic(w):
            return ""
        key = strip_arabic_diacritics(w)
        if key in self._lex:
            return self._lex[key]
        diac = apply_onnx_partial_postprocess(self._onnx.diacritize(w))
        return word_to_ipa_with_assimilation(diac)

    def text_to_ipa(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text.strip())
        if not text:
            return ""
        parts: list[str] = []
        for raw in text.split():
            tok = raw.strip()
            if not tok:
                continue
            ipa = self.g2p_word(tok)
            if ipa:
                parts.append(ipa)
        return " ".join(parts)


def arabic_g2p_line(
    line: str,
    *,
    model_dir: Path | None = None,
    dict_path: Path | None = None,
    lex: Mapping[str, str] | None = None,
) -> str:
    """Functional API for parity tests (optional preloaded *lex* overrides file lexicon)."""
    line = unicodedata.normalize("NFC", line.rstrip("\n\r"))
    if not line.strip():
        return ""
    g = ArabicRuleG2p(model_dir=model_dir, dict_path=dict_path)
    if lex is not None:
        g._lex = dict(lex)
    return g.text_to_ipa(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="MSA Arabic G2P (ONNX tashkīl + rules).")
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--dict", type=Path, default=None)
    ap.add_argument("--stdin", action="store_true")
    args = ap.parse_args()

    g = ArabicRuleG2p(model_dir=args.model_dir, dict_path=args.dict)
    if args.stdin or not args.text:
        import sys

        phrase = sys.stdin.read()
    else:
        phrase = args.text
    print(g.text_to_ipa(phrase))


if __name__ == "__main__":
    main()
