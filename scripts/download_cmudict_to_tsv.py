#!/usr/bin/env python3
"""
Download CMUdict (ARPAbet) and write IPA ``data/en_us/dict.tsv`` (word<TAB>ipa per line).

Dictionary source:
https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict
"""

from __future__ import annotations

import sys
import urllib.request
from io import StringIO
from pathlib import Path

# Repo root (parent of ``scripts/``).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cmudict_ipa import CmudictIpa

CMUDICT_RAW_URL = (
    "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
)
DEFAULT_OUT = _ROOT / "data" / "en_us" / "dict.tsv"


def main() -> None:
    out = DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        CMUDICT_RAW_URL,
        headers={"User-Agent": "moonshine-g2p-dict-fetch/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    d = CmudictIpa(StringIO(text), format="cmudict")
    rows = list(d.iter_pronunciation_rows())
    with open(out, "w", encoding="utf-8") as f:
        for word, ipa in rows:
            f.write(f"{word}\t{ipa}\n")
    print(f"Wrote {out} ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
