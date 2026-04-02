#!/usr/bin/env python3
"""
Compare :func:`arabic_rule_g2p.arabic_g2p_line` to the C++ ``arabic_rule_g2p`` CLI on the first *N*
lines of ``data/ar/wiki-text.txt``.

Usage (from repo root)::

    python3 scripts/arabic_parity_cpp_python.py --lines 100 \\
        --cpp moonshine-tts/build/arabic_rule_g2p --onnx-dir data/ar_msa/arabertv02_tashkeel_fadel_onnx
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arabic_rule_g2p import arabic_g2p_line  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Arabic Python vs C++ G2P parity.")
    ap.add_argument("--lines", type=int, default=100)
    ap.add_argument("--wiki", type=Path, default=_REPO / "data" / "ar" / "wiki-text.txt")
    ap.add_argument(
        "--cpp",
        type=Path,
        default=_REPO / "moonshine-tts" / "build" / "arabic_rule_g2p",
        help="Path to arabic_rule_g2p executable",
    )
    ap.add_argument("--dict", type=Path, default=_REPO / "data" / "ar_msa" / "dict.tsv")
    ap.add_argument(
        "--onnx-dir",
        type=Path,
        default=_REPO / "data" / "ar_msa" / "arabertv02_tashkeel_fadel_onnx",
    )
    args = ap.parse_args()

    if not args.cpp.is_file():
        print(f"Missing C++ binary: {args.cpp} (build with cmake --build moonshine-tts/build)", file=sys.stderr)
        sys.exit(2)
    if not args.wiki.is_file():
        print(f"Missing wiki: {args.wiki}", file=sys.stderr)
        sys.exit(2)

    mismatches = 0
    n = 0
    with args.wiki.open(encoding="utf-8") as f:
        for line in f:
            if n >= args.lines:
                break
            line = line.rstrip("\n")
            n += 1
            py_out = arabic_g2p_line(line, model_dir=args.onnx_dir, dict_path=args.dict)
            r = subprocess.run(
                [
                    str(args.cpp),
                    "--onnx-dir",
                    str(args.onnx_dir),
                    "--dict",
                    str(args.dict),
                    line,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if r.returncode != 0:
                print(f"C++ failed line {n}: {r.stderr}", file=sys.stderr)
                mismatches += 1
                continue
            cpp_out = r.stdout.rstrip("\n")
            if py_out != cpp_out:
                mismatches += 1
                if mismatches <= 12:
                    print(f"--- mismatch line {n} ---")
                    print("IN:", line[:200])
                    print("PY:", py_out[:300])
                    print("C++:", cpp_out[:300])

    print(f"Compared {n} lines; mismatches: {mismatches}")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
