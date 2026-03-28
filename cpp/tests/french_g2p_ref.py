#!/usr/bin/env python3
"""Reference IPA for C++ parity tests (invoked with a UTF-8 file path as argv[1])."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from french_g2p import text_to_ipa  # noqa: E402

_DICT = ROOT / "data" / "fr" / "dict.tsv"
_CSV_DIR = ROOT / "data" / "fr"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(2)
    path = Path(sys.argv[1])
    raw = path.read_text(encoding="utf-8")
    if len(sys.argv) >= 3 and sys.argv[2] == "--first-lines":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        for line in raw.splitlines()[:n]:
            print(text_to_ipa(line, dict_path=_DICT, csv_dir=_CSV_DIR))
        return
    print(text_to_ipa(raw.rstrip("\n"), dict_path=_DICT, csv_dir=_CSV_DIR), end="")


if __name__ == "__main__":
    main()
