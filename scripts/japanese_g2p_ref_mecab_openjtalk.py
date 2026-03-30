#!/usr/bin/env python3
"""
High-quality **reference** Japanese G2P for evaluation (not ONNX-only).

Uses **fugashi** + **unidic-lite** (MeCab) for readings, then the same
:func:`japanese_kana_to_ipa.reading_katakana_to_ipa` as the dependency-free path so IPA is
comparable.

Optionally prints **pyopenjtalk** space-separated HTS-style phones in a second column when
``--with-openjtalk`` is set (requires ``pyopenjtalk``).

Install (example)::

    pip install fugashi unidic-lite pyopenjtalk
"""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from japanese_kana_to_ipa import reading_katakana_to_ipa  # noqa: E402


def mecab_katakana_reading(text: str) -> str:
    from fugashi import Tagger

    t = Tagger()
    raw = unicodedata.normalize("NFC", text.strip())
    chunks: list[str] = []
    for w in t(raw):
        f = w.feature
        y = getattr(f, "pron", None) or getattr(f, "kana", None) or ""
        if isinstance(y, str) and y.strip():
            chunks.append(y.strip())
        else:
            chunks.append(w.surface)
    return "".join(chunks)


def line_to_ipa_mecab(text: str) -> str:
    k = mecab_katakana_reading(text)
    if not k.strip():
        return ""
    return reading_katakana_to_ipa(k)


def line_to_phones_openjtalk(text: str) -> str:
    import pyopenjtalk

    return pyopenjtalk.g2p(unicodedata.normalize("NFC", text.strip()), kana=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="MeCab/UniDic + IPA reference (+ optional OpenJTalk).")
    ap.add_argument("text", nargs="?", default="東京に行きます。")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--wiki", type=Path, default=None, help="Print one IPA line per wiki line.")
    ap.add_argument("--first-lines", type=int, default=0)
    ap.add_argument("--with-openjtalk", action="store_true")
    args = ap.parse_args()

    if args.wiki is not None:
        with args.wiki.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.first_lines and i >= args.first_lines:
                    break
                s = line.rstrip("\n\r")
                ipa = line_to_ipa_mecab(s)
                if args.with_openjtalk:
                    try:
                        oj = line_to_phones_openjtalk(s)
                    except Exception as e:  # noqa: BLE001
                        oj = f"<error:{e}>"
                    print(f"{ipa}\t{oj}")
                else:
                    print(ipa)
        return

    if args.stdin:
        import sys as _sys

        text = _sys.stdin.read()
    else:
        text = args.text
    ipa = line_to_ipa_mecab(text)
    if args.with_openjtalk:
        print(ipa + "\t" + line_to_phones_openjtalk(text))
    else:
        print(ipa)


if __name__ == "__main__":
    main()
