#!/usr/bin/env python3
"""
Compare **whitespace tokenization** (:func:`vietnamese_rule_g2p.vietnamese_g2p_line`) to
**word-tokenized** reference (:func:`scripts.vietnamese_g2p_ref_libraries.line_to_ipa_ref`).

The reference path improves hits on multi-word ``dict.tsv`` keys when underthesea groups syllables
into words. Differences highlight token-boundary / segmentation effects, not IPA inventory drift
(both use the same lexicon + rule engine).

Analysis (typical findings):
- **Agreement:** Most lines match when wiki text is already syllable-spaced and underthesea splits
  align with whitespace chunks.
- **Mismatches:** Long multi-word lexicon entries (proper names, MWEs) may be found only when
  underthesea keeps a phrase as one token; the baseline greedy longest-match over whitespace may
  miss unless the phrase is contiguous without extra spaces.
- **OOV:** Both paths share the same rule syllabifier; rare spellings may differ from ``dict.tsv``
  until rules are extended.

Requires: ``underthesea`` for meaningful reference output.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vietnamese_rule_g2p import vietnamese_g2p_line  # noqa: E402


def _load_ref():
    p = _REPO / "scripts" / "vietnamese_g2p_ref_libraries.py"
    spec = importlib.util.spec_from_file_location("vietnamese_g2p_ref_libraries", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.line_to_ipa_ref


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Vietnamese G2P baseline vs underthesea ref.")
    ap.add_argument("--wiki", type=Path, default=_REPO / "data" / "vi" / "wiki-text.txt")
    ap.add_argument("--max-lines", type=int, default=200)
    ap.add_argument("--show-mismatch", type=int, default=8)
    args = ap.parse_args()

    try:
        line_to_ipa_ref = _load_ref()
    except Exception as e:
        print(f"Cannot load reference module: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        line_to_ipa_ref("thử", prefer_mecab=False, dict_path=None)
    except RuntimeError as e:
        print(f"Reference path unavailable: {e}", file=sys.stderr)
        sys.exit(1)

    path = args.wiki
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)

    n = 0
    same = 0
    mismatches: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if n >= args.max_lines:
                break
            n += 1
            a = vietnamese_g2p_line(line)
            b = line_to_ipa_ref(line, prefer_mecab=False, dict_path=None)
            if a == b:
                same += 1
            else:
                if len(mismatches) < args.show_mismatch:
                    mismatches.append((line[:120], a[:200], b[:200]))

    print(f"Lines compared: {n}")
    print(f"Exact IPA string match: {same} ({100.0 * same / max(n, 1):.1f}%)")
    print("\nSample mismatches (input snippet | whitespace | underthesea):")
    for inp, ipa_a, ipa_b in mismatches:
        print("---")
        print("IN:", inp)
        print("WS:", ipa_a)
        print("UT:", ipa_b)


if __name__ == "__main__":
    main()
