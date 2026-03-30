#!/usr/bin/env python3
"""
Reference Vietnamese G2P using **external tokenization** + the same ``data/vi/dict.tsv`` pipeline
as :mod:`vietnamese_rule_g2p`.

**MeCab:** Standard MeCab does not ship a Vietnamese dictionary. If you build a Vietnamese MeCab
dictionary and set ``MECABRC`` / ``VI_MECABRC`` to its ``mecabrc``, this script will try
``mecab-python3`` and use **BOS/EOS-delimited** morpheme surfaces as coarse word spans (same
lexicon longest-match inside each span). If MeCab is unavailable or misconfigured, the script
falls back to **underthesea** ``word_tokenize`` (recommended for Vietnamese).

Other optional deps: ``underthesea`` for word boundaries; ``phonemizer`` + eSpeak NG can be used
for a second IPA column (different phone set — compare qualitatively only).

Usage::

    python3 scripts/vietnamese_g2p_ref_libraries.py --text "tổ chức quốc tế"
    python3 scripts/vietnamese_g2p_ref_libraries.py --stdin < data/vi/wiki-text.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vietnamese_rule_g2p import vietnamese_g2p_line  # noqa: E402


def _tokenize_underthesea(text: str) -> list[str]:
    from underthesea import word_tokenize

    return list(word_tokenize(text))


def _tokenize_mecab_surfaces(text: str) -> list[str] | None:
    try:
        import MeCab  # type: ignore
    except ImportError:
        return None
    rc = os.environ.get("VI_MECABRC") or os.environ.get("MECABRC")
    if rc and Path(rc).is_file():
        try:
            t = MeCab.Tagger(f"-r {rc}")
        except Exception:
            t = MeCab.Tagger()
    else:
        t = MeCab.Tagger()
    out: list[str] = []
    node = t.parseToNode(text)
    while node:
        if node.surface and node.surface.strip():
            out.append(node.surface)
        node = node.next
    return out if out else None


def tokenize_words(text: str, *, prefer_mecab: bool) -> tuple[list[str], str]:
    if prefer_mecab:
        m = _tokenize_mecab_surfaces(text)
        if m is not None:
            return m, "mecab-surfaces"
    try:
        return _tokenize_underthesea(text), "underthesea"
    except ImportError as e:
        raise RuntimeError(
            "Install underthesea (pip install underthesea) or configure Vietnamese MeCab."
        ) from e


def line_to_ipa_ref(text: str, *, prefer_mecab: bool, dict_path: Path | None) -> str:
    text = text.strip()
    if not text:
        return ""
    words, _src = tokenize_words(text, prefer_mecab=prefer_mecab)
    parts: list[str] = []
    dp = dict_path
    for w in words:
        ipa = vietnamese_g2p_line(w, dict_path=dp)
        if ipa:
            parts.append(ipa)
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vietnamese reference G2P (underthesea / optional MeCab).")
    ap.add_argument("--dict", type=Path, default=None)
    ap.add_argument("--prefer-mecab", action="store_true", help="Try MeCab surfaces before underthesea.")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("text", nargs="*")
    args = ap.parse_args()
    if args.stdin or not args.text:
        raw = sys.stdin.read()
    else:
        raw = " ".join(args.text)
    try:
        print(line_to_ipa_ref(raw, prefer_mecab=args.prefer_mecab, dict_path=args.dict))
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
