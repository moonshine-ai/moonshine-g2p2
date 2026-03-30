#!/usr/bin/env python3
"""
Compare **ONNX+lexicon** Japanese G2P (:mod:`japanese_onnx_g2p`) to **MeCab+UniDic** reference
(:mod:`scripts.japanese_g2p_ref_mecab_openjtalk` pipeline).

Reports line-level exact-match rate and a cheap **normalized** similarity (alnum IPA chars only),
which is more forgiving of spacing / tie-bar quirks vs OpenJTalk phones.

Requires: onnxruntime, numpy, fugashi, unidic-lite (MeCab path). Optional: pyopenjtalk for extra column.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from japanese_onnx_g2p import JapaneseOnnxG2p  # noqa: E402


def _norm_ipa(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace(" ", "").replace("ː", "").lower()
    return re.sub(r"[^a-zɯæɛɪʊɔɑɾɕʑɲɴçɸβθðʔ]+", "", s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wiki", type=Path, nargs="?", default=_REPO / "data" / "ja" / "wiki-text.txt")
    ap.add_argument("--first-lines", type=int, default=100)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--dict", type=Path, default=None)
    ap.add_argument("--verbose-mismatches", type=int, default=5)
    args = ap.parse_args()

    import importlib.util

    ref_path = _REPO / "scripts" / "japanese_g2p_ref_mecab_openjtalk.py"
    spec = importlib.util.spec_from_file_location("japanese_g2p_ref_mecab_openjtalk", ref_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reference script")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    line_to_ipa_mecab = mod.line_to_ipa_mecab

    g = JapaneseOnnxG2p(model_dir=args.model_dir, dict_path=args.dict)
    exact = 0
    sim_acc = 0.0
    n = 0
    mismatches: list[tuple[str, str, str]] = []
    with args.wiki.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.first_lines:
                break
            s = line.rstrip("\n\r")
            a = g.text_to_ipa(s)
            b = line_to_ipa_mecab(s)
            n += 1
            if a == b:
                exact += 1
            else:
                if len(mismatches) < args.verbose_mismatches:
                    mismatches.append((s[:80], a, b))
            na = _norm_ipa(a)
            nb = _norm_ipa(b)
            if not na and not nb:
                sim_acc += 1.0
            elif not na or not nb:
                sim_acc += 0.0
            else:
                common = sum(1 for x, y in zip(na, nb) if x == y)
                sim_acc += common / max(len(na), len(nb))

    print(f"lines: {n}")
    print(f"exact IPA string match: {exact}/{n} ({100.0 * exact / max(n, 1):.1f}%)")
    print(f"mean normalized char similarity: {sim_acc / max(n, 1):.3f}")
    if mismatches:
        print("\nSample mismatches (input snippet | onnx | mecab_reading_ipa):")
        for snip, a, b in mismatches:
            print(f"  {snip!r}")
            print(f"    onnx:  {a}")
            print(f"    ref:   {b}")


if __name__ == "__main__":
    main()
