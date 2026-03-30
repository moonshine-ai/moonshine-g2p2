#!/usr/bin/env python3
"""Reference IPA for C++ parity tests (invoked with a UTF-8 file path as argv[1])."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turkish_rule_g2p import text_to_ipa  # noqa: E402


def main() -> None:
    args = list(sys.argv[1:])
    if not args:
        raise SystemExit(2)
    path = Path(args[0])
    args = args[1:]
    raw = path.read_text(encoding="utf-8")
    if args and args[0] == "--first-lines":
        n = int(args[1]) if len(args) > 1 else 5
        for line in raw.splitlines()[:n]:
            print(text_to_ipa(line))
        return
    print(text_to_ipa(raw.rstrip("\n")), end="")


if __name__ == "__main__":
    main()
