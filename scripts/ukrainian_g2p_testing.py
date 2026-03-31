#!/usr/bin/env python3
"""
High-quality **Ukrainian** G2P testing harness using external libraries (primarily **phonemizer**
+ **eSpeak NG**). Intended for regression checks against :mod:`ukrainian_rule_g2p`.

Features:
  - Batch sentences or first *N* lines of ``data/uk/wiki-text.txt``
  - Optional side-by-side diff vs rule-based output (character-level similarity summary)
  - Documents recommended install paths

Example::

  python scripts/ukrainian_g2p_testing.py --sample
  python scripts/ukrainian_g2p_testing.py --wiki-lines 50 --compare-rule
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ukrainian_rule_g2p import text_to_ipa  # noqa: E402

_ref_path = ROOT / "scripts" / "ukrainian_g2p_ref_library.py"
_spec = importlib.util.spec_from_file_location("ukrainian_g2p_ref_library", _ref_path)
_ref = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ref)
ipa_phonemizer_espeak = _ref.ipa_phonemizer_espeak
ipa_epitran = _ref.ipa_epitran

SAMPLE = [
    "м'ясо",
    "Україна",
    "Було 5 котів.",
    "географія",
    "сонце",
]


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ukrainian G2P testing (phonemizer / Epitran vs rules).")
    ap.add_argument("--sample", action="store_true", help="Run built-in sample sentences")
    ap.add_argument("--wiki-lines", type=int, metavar="N", help="Add first N lines of data/uk/wiki-text.txt")
    ap.add_argument(
        "--compare-rule",
        action="store_true",
        help="Print rule-based line under each reference line",
    )
    ap.add_argument("--text", nargs="+", help="Extra UTF-8 sentences")
    args = ap.parse_args()

    lines: list[str] = []
    if args.text:
        lines.append(" ".join(args.text))
    if args.wiki_lines:
        wiki = ROOT / "data" / "uk" / "wiki-text.txt"
        if wiki.is_file():
            lines.extend(wiki.read_text(encoding="utf-8").splitlines()[: args.wiki_lines])
        else:
            print(f"warning: missing {wiki}", file=sys.stderr)

    if not lines:
        lines.extend(SAMPLE)

    ref_es = ipa_phonemizer_espeak("тест") is not None
    ref_ep = ipa_epitran("тест") is not None
    print("Backends:", file=sys.stderr)
    print(f"  phonemizer+eSpeak (uk): {ref_es}", file=sys.stderr)
    print(f"  epitran:                {ref_ep}", file=sys.stderr)
    print(file=sys.stderr)

    sim_es: list[float] = []
    for sent in lines:
        if not sent.strip():
            continue
        es = ipa_phonemizer_espeak(sent) if ref_es else None
        ep = ipa_epitran(sent) if ref_ep else None
        rule = text_to_ipa(sent)
        print("---")
        print("TXT:", sent[:100] + ("…" if len(sent) > 100 else ""))
        if es is not None:
            print("ESPK:", es)
            sim_es.append(_similarity(rule.replace(" ", ""), es.replace(" ", "")))
        else:
            print("ESPK:", "(unavailable)")
        if ep is not None:
            print("EPTR:", ep)
        if args.compare_rule:
            print("RULE:", rule)

    if sim_es:
        avg = sum(sim_es) / len(sim_es)
        print("---", file=sys.stderr)
        print(f"(Heuristic) mean char-set similarity RULE vs eSpeak: {avg:.3f}", file=sys.stderr)
        print("Low similarity is expected where stress, clusters, or voicing differ.", file=sys.stderr)


if __name__ == "__main__":
    main()
