#!/usr/bin/env python3
"""
Compare **rules-only** :func:`hindi_rule_g2p.devanagari_word_to_ipa` (empty lexicon) to
**lexicon reference** pronunciations in ``data/hi/dict.tsv``.

The reference is whatever is in the TSV (e.g. Wiktionary); the rule engine uses its own
broad IPA heuristics, so expect low strict agreement. Normalized tiers strip stress and
syllable dots and apply a few symbol mappings for a fairer overlap measure.

Examples::

    python scripts/eval_hindi_lexicon_vs_rules.py
    python scripts/eval_hindi_lexicon_vs_rules.py --dict data/hi/dict.tsv
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hindi_rule_g2p import devanagari_word_to_ipa  # noqa: E402

_DEFAULT_DICT = _ROOT / "data" / "hi" / "dict.tsv"

_STRESS = frozenset("ˈˌ")
# Syllable boundary dots from rule output; reference may use them too.
_BOUNDARY = "."


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def normalize_loose(s: str) -> str:
    """NFC, drop stress and syllable dots (keep nasal tiebars etc.)."""
    t = unicodedata.normalize("NFC", s)
    for ch in _STRESS:
        t = t.replace(ch, "")
    t = t.replace(_BOUNDARY, "")
    return t


def normalize_broad(s: str) -> str:
    """Loose + a few common script/lexicon alternations."""
    t = normalize_loose(s)
    t = t.replace("ɑ", "a")
    return t


def load_lexicon_entries(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "\t" not in s:
            continue
        w, ipa = s.split("\t", 1)
        w, ipa = w.strip(), ipa.strip()
        if w and ipa:
            rows.append((w, ipa))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict", type=Path, default=_DEFAULT_DICT, help="TSV with word<TAB>ipa reference")
    ap.add_argument("--no-stress", action="store_true", help="Compare with rules --no-stress equivalent")
    args = ap.parse_args()

    path = args.dict.resolve()
    if not path.is_file():
        print(f"Missing dict: {path}")
        return 1

    entries = load_lexicon_entries(path)
    with_stress = not args.no_stress

    n = 0
    rule_empty = 0
    strict_ok = 0
    loose_ok = 0
    broad_ok = 0
    sum_rel_dist_loose = 0.0
    sum_rel_dist_broad = 0.0

    examples_strict_mismatch: list[tuple[str, str, str]] = []
    examples_loose_mismatch: list[tuple[str, str, str]] = []

    for word, ref in entries:
        hyp = devanagari_word_to_ipa(word, lexicon={}, with_stress=with_stress)
        n += 1
        if not hyp:
            rule_empty += 1
            continue

        r0, h0 = unicodedata.normalize("NFC", ref), unicodedata.normalize("NFC", hyp)
        if r0 == h0:
            strict_ok += 1
        elif len(examples_strict_mismatch) < 12:
            examples_strict_mismatch.append((word, ref, hyp))

        rl, hl = normalize_loose(ref), normalize_loose(hyp)
        if rl == hl:
            loose_ok += 1
        elif len(examples_loose_mismatch) < 12:
            examples_loose_mismatch.append((word, ref, hyp))

        rb, hb = normalize_broad(ref), normalize_broad(hyp)
        if rb == hb:
            broad_ok += 1

        mxl = max(len(rl), len(hl), 1)
        sum_rel_dist_loose += _levenshtein(rl, hl) / mxl
        mxb = max(len(rb), len(hb), 1)
        sum_rel_dist_broad += _levenshtein(rb, hb) / mxb

    denom = n - rule_empty
    print(f"dict={path}")
    print(f"entries={n} rules_non_empty={denom} rules_empty_output={rule_empty}")
    if denom <= 0:
        return 0

    def pct(x: int) -> float:
        return 100.0 * x / denom

    print(
        f"exact_match_strict={strict_ok}/{denom} ({pct(strict_ok):.1f}%)  "
        f"(NFC string equality vs lexicon IPA)"
    )
    print(
        f"exact_match_loose={loose_ok}/{denom} ({pct(loose_ok):.1f}%)  "
        f"(strip stress ˈˌ and syllable dots .)"
    )
    print(
        f"exact_match_broad={broad_ok}/{denom} ({pct(broad_ok):.1f}%)  "
        f"(loose + map ɑ→a)"
    )
    print(
        f"mean_normalized_edit_distance_loose={sum_rel_dist_loose/denom:.3f}  "
        f"(0=identical after loose norm)"
    )
    print(
        f"mean_normalized_edit_distance_broad={sum_rel_dist_broad/denom:.3f}  "
        f"(0=identical after broad norm)"
    )
    print(f"mean_similarity_loose={1.0 - sum_rel_dist_loose/denom:.3f}")

    if examples_strict_mismatch:
        print("\nSample strict mismatches (word | lexicon | rules):")
        for w, r, h in examples_strict_mismatch:
            print(f"  {w} | {r} | {h}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
