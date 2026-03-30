#!/usr/bin/env python3
"""Reference IPA lines for C++ parity (``japanese_onnx_g2p`` on wiki text)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wiki", type=Path)
    ap.add_argument("--first-lines", type=int, default=100)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--dict", type=Path, default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(_REPO_ROOT))
    from japanese_onnx_g2p import JapaneseOnnxG2p

    md = args.model_dir or (_REPO_ROOT / "data" / "ja" / "roberta_japanese_char_luw_upos_onnx")
    dct = args.dict or (_REPO_ROOT / "data" / "ja" / "dict.tsv")
    g = JapaneseOnnxG2p(model_dir=md, dict_path=dct)
    with args.wiki.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.first_lines:
                break
            print(g.text_to_ipa(line.rstrip("\n\r")))


if __name__ == "__main__":
    main()
