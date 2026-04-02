#!/usr/bin/env python3
"""Compare ``french_g2p.text_to_ipa`` (Python) vs ``french_g2p_batch`` (C++ core) line-by-line.

Example (from repo root, after building ``moonshine-tts/build-core/french_g2p_batch``)::

    ./scripts/compare_french_py_cpp_wikitext.py --lines 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from french_g2p import FrenchG2PConfig, text_to_ipa


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lines", type=int, default=100, metavar="N", help="Number of lines from wiki file (default: 100)")
    p.add_argument(
        "--wiki",
        type=Path,
        default=_REPO / "data" / "fr" / "wiki-text.txt",
        help="Source text file (default: data/fr/wiki-text.txt)",
    )
    p.add_argument(
        "--batch",
        type=Path,
        default=_REPO / "moonshine-tts" / "build-core" / "french_g2p_batch",
        help="Path to french_g2p_batch executable",
    )
    p.add_argument("--dict", type=Path, default=_REPO / "data" / "fr" / "dict.tsv")
    p.add_argument("--csv-dir", type=Path, default=_REPO / "data" / "fr")
    args = p.parse_args()

    if not args.wiki.is_file():
        print(f"error: wiki file not found: {args.wiki}", file=sys.stderr)
        return 1
    if not args.batch.is_file():
        print(f"error: build french_g2p_batch first: {args.batch}", file=sys.stderr)
        return 1

    lines: list[str] = []
    with args.wiki.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.lines:
                break
            lines.append(line.rstrip("\n\r"))

    cfg = FrenchG2PConfig()
    py_out = [text_to_ipa(line, config=cfg) if line.strip() else "" for line in lines]

    stdin = "\n".join(lines) + "\n"
    r = subprocess.run(
        [str(args.batch), "--dict", str(args.dict), "--csv-dir", str(args.csv_dir)],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode

    cpp_lines = r.stdout.splitlines()
    while len(cpp_lines) < len(lines):
        cpp_lines.append("")
    cpp_lines = cpp_lines[: len(lines)]

    def strip_stress(s: str) -> str:
        return s.replace("\u02c8", "").replace("\u02cc", "")

    exact = stress_only = mismatch = 0
    for py, cc in zip(py_out, cpp_lines):
        if py == cc:
            exact += 1
        elif strip_stress(py) == strip_stress(cc):
            stress_only += 1
        else:
            mismatch += 1

    print(f"{args.wiki.name}: first {len(lines)} lines")
    print(f"  exact: {exact}, stress-only diff: {stress_only}, content mismatch: {mismatch}")

    if mismatch or stress_only:
        for i, (inp, py, cc) in enumerate(zip(lines, py_out, cpp_lines)):
            if py != cc:
                kind = "stress" if strip_stress(py) == strip_stress(cc) else "content"
                print(f"\n--- line {i + 1} ({kind}) ---")
                print(f"in: {inp[:200]!r}...")
                print(f"PY:  {py[:300]!r}...")
                print(f"C++: {cc[:300]!r}...")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
