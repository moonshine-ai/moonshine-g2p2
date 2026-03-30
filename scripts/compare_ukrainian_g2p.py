#!/usr/bin/env python3
"""
Compare **dependency-free** :mod:`ukrainian_rule_g2p` to a **library-backed** reference
(:func:`scripts.ukrainian_g2p_ref_library.ipa_phonemizer_espeak` / Epitran).

Prints side-by-side IPA and a short analysis of systematic differences (stress, notation).

Does **not** fail if eSpeak/Epitran is missing — reference columns show ``(unavailable)``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

from ukrainian_rule_g2p import text_to_ipa  # noqa: E402

_ref_path = ROOT / "scripts" / "ukrainian_g2p_ref_library.py"
_spec = importlib.util.spec_from_file_location("ukrainian_g2p_ref_library", _ref_path)
_ref = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ref)
ipa_phonemizer_espeak = _ref.ipa_phonemizer_espeak
ipa_epitran = _ref.ipa_epitran


DEFAULT_SENTENCES = [
    "м'ясо",
    "кінь",
    "ґрунт",
    "Україна",
    "їжак",
    "дім",
    "сонце",
    "ніч",
    "Було 5 котів.",
    "географія",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare rule-based Ukrainian G2P to library reference.")
    ap.add_argument(
        "--wiki-lines",
        type=int,
        metavar="N",
        help="Also compare first N lines of data/uk/wiki-text.txt (can be slow with eSpeak)",
    )
    args = ap.parse_args()

    ref_es = ipa_phonemizer_espeak("тест") is not None
    ref_ep = ipa_epitran("тест") is not None
    print("Reference availability:", file=sys.stderr)
    print(f"  phonemizer+eSpeak NG: {ref_es}", file=sys.stderr)
    print(f"  epitran (ukr-Cyrl*):  {ref_ep}", file=sys.stderr)
    print(file=sys.stderr)

    lines = list(DEFAULT_SENTENCES)
    if args.wiki_lines:
        wiki = ROOT / "data" / "uk" / "wiki-text.txt"
        if wiki.is_file():
            raw = wiki.read_text(encoding="utf-8").splitlines()
            lines.extend(raw[: args.wiki_lines])
        else:
            print(f"warning: missing {wiki}", file=sys.stderr)

    for sent in lines:
        rule = text_to_ipa(sent)
        es = ipa_phonemizer_espeak(sent) if ref_es else None
        ep = ipa_epitran(sent) if ref_ep else None
        print("---")
        print("TXT:", sent[:120] + ("…" if len(sent) > 120 else ""))
        print("RULE:", rule)
        print("ESPK:", es if es is not None else "(unavailable)")
        print("EPTR:", ep if ep is not None else "(unavailable)")

    print("---")
    print(
        """
Analysis (within rule-based constraints):
- **Stress:** Literary Ukrainian does not mark stress in running text; eSpeak uses a lexicon and
  heuristics, while the rule engine applies a fixed **penultimate-syllable** ˈ. Mismatches on stress
  placement are expected, not bugs in the letter-to-phone mapping.
- **Voicing assimilation / cluster reduction:** Not modeled (would need morphology or a dictionary).
  eSpeak may simplify clusters (e.g. in *сонце*) where spelling keeps full graphemes.
- **в allophony:** Rules use ʋ before vowels / glide contexts and **w** before consonants and
  word-finally; eSpeak may use a single symbol or a different approximant choice.
- **Palatalization:** Lookahead for є і ї ю я and ь matches standard orthographic rules; apostrophe
  blocks palatalization before those vowels (м'ясо → mja…).
- **Stress stripping:** All Unicode Mn combining marks are removed before G2P **except** U+0308
  (diaeresis), so dictionary stress marks disappear but **ї** stays distinct from **і** after NFD.
- **Digits:** Cardinals (and ``n-n`` ranges) expand to Ukrainian words before G2P; compare reference
  on the same surface when testing digit sentences.
""".strip()
    )


if __name__ == "__main__":
    main()
