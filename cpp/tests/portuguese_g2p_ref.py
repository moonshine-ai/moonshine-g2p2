#!/usr/bin/env python3
"""Reference IPA for C++ parity tests (invoked with a UTF-8 file path as argv[1])."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from portuguese_rule_g2p import text_to_ipa  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(2)
    path = Path(sys.argv[1])
    raw = path.read_text(encoding="utf-8")
    variant = "pt_br"
    i = 2
    if i < len(sys.argv) and sys.argv[i] == "--portugal":
        variant = "pt_pt"
        i += 1
    if i < len(sys.argv) and sys.argv[i] == "--first-lines":
        n = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 5
        for line in raw.splitlines()[:n]:
            print(text_to_ipa(line, variant=variant))
        return
    print(text_to_ipa(raw.rstrip("\n"), variant=variant), end="")


if __name__ == "__main__":
    main()
