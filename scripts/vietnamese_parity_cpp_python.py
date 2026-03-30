#!/usr/bin/env python3
"""
Compare :func:`vietnamese_rule_g2p.vietnamese_g2p_line` to the C++ ``vietnamese_rule_g2p`` CLI
on the first *N* non-empty lines of ``data/vi/wiki-text.txt``.

Usage (from repo root)::

    python3 scripts/vietnamese_parity_cpp_python.py --lines 100 \\
        --cpp cpp/build/vietnamese_rule_g2p
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vietnamese_rule_g2p import vietnamese_g2p_line  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Vietnamese Python vs C++ G2P parity.")
    ap.add_argument("--lines", type=int, default=100)
    ap.add_argument("--wiki", type=Path, default=_REPO / "data" / "vi" / "wiki-text.txt")
    ap.add_argument(
        "--cpp",
        type=Path,
        default=_REPO / "cpp" / "build" / "vietnamese_rule_g2p",
        help="Path to vietnamese_rule_g2p executable",
    )
    ap.add_argument("--dict", type=Path, default=_REPO / "data" / "vi" / "dict.tsv")
    args = ap.parse_args()

    if not args.cpp.is_file():
        print(f"Missing C++ binary: {args.cpp} (build with cmake --build cpp/build)", file=sys.stderr)
        sys.exit(2)
    if not args.wiki.is_file():
        print(f"Missing wiki: {args.wiki}", file=sys.stderr)
        sys.exit(2)

    mismatches = 0
    n = 0
    with args.wiki.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if n >= args.lines:
                break
            n += 1
            py_out = vietnamese_g2p_line(line, dict_path=args.dict)
            r = subprocess.run(
                [str(args.cpp), "--dict", str(args.dict), line],
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
