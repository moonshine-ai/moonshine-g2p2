#!/usr/bin/env python3
"""Reference lines for C++ parity: Korean RoBERTa morph+UPOS on wiki text (see korean_tok_pos.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wiki", type=Path)
    ap.add_argument("--first-lines", type=int, default=100)
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Default: <repo>/data/ko/roberta_korean_morph_upos_onnx",
    )
    args = ap.parse_args()
    md = args.model_dir or (_REPO_ROOT / "data" / "ko" / "roberta_korean_morph_upos_onnx")
    sys.path.insert(0, str(_REPO_ROOT))
    from korean_tok_pos import KoreanTokPosOnnx

    pipe = KoreanTokPosOnnx(md)
    lines: list[str] = []
    with args.wiki.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.first_lines:
                break
            lines.append(line.rstrip("\n\r"))
    for line in lines:
        pairs = pipe.annotate(line)
        out = "".join(f"{a}/{b} " for a, b in pairs)
        print(out)


if __name__ == "__main__":
    main()
