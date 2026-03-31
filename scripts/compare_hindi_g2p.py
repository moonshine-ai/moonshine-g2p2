#!/usr/bin/env python3
"""
Compare **dependency-free** :mod:`hindi_rule_g2p` to **phonemizer + eSpeak NG** (``hi``).

Prints side-by-side IPA and a short analysis. Does **not** fail if eSpeak is missing.

Example::

  python scripts/compare_hindi_g2p.py
  python scripts/compare_hindi_g2p.py --wiki-lines 20
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hindi_rule_g2p import text_to_ipa  # noqa: E402

_ref_path = ROOT / "scripts" / "hindi_g2p_ref_library.py"
_spec = importlib.util.spec_from_file_location("hindi_g2p_ref_library", _ref_path)
_ref = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ref)
ipa_phonemizer_espeak = _ref.ipa_phonemizer_espeak

DEFAULT_SENTENCES = [
    "कमल",
    "हिन्दी",
    "संचार",
    "मैं ४२ साल का हूँ।",
    "भारत",
    "समझ",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare rule-based Hindi G2P to eSpeak reference.")
    ap.add_argument("--wiki-lines", type=int, metavar="N", help="Append first N lines of data/hi/wiki-text.txt")
    args = ap.parse_args()

    ref_ok = ipa_phonemizer_espeak("टेस्ट") is not None
    print("Reference (phonemizer + eSpeak hi):", ref_ok, file=sys.stderr)
    print(file=sys.stderr)

    lines = list(DEFAULT_SENTENCES)
    if args.wiki_lines:
        wiki = ROOT / "data" / "hi" / "wiki-text.txt"
        if wiki.is_file():
            raw = wiki.read_text(encoding="utf-8").splitlines()
            lines.extend(raw[: args.wiki_lines])
        else:
            print(f"warning: missing {wiki}", file=sys.stderr)

    for sent in lines:
        rule = text_to_ipa(sent)
        es = ipa_phonemizer_espeak(sent) if ref_ok else None
        print("---")
        print("TXT:", sent[:120] + ("…" if len(sent) > 120 else ""))
        print("RULE:", rule)
        print("ESPK:", es if es is not None else "(unavailable)")

    print("---")
    print(
        """
Analysis (within rule-based + lexicon constraints):
- **Schwa syncope:** Only a coarse approximation (final inherent schwa; medial syncope before
  affricates like झ). eSpeak uses a lexicon and heuristics; mismatches on polysyllabic OOV words
  are expected.
- **Anusvara / chandrabindu:** Homorganic nasal before stops/affricates; word-final or isolated
  ं uses combining nasalization on the vowel (U+0303). eSpeak may use different nasal placement
  or allophone choices.
- **ऋ / rare vowels:** Not modeled; OOV nukta or English mixed script tokens are skipped in
  rule output.
- **Stress:** Rule engine uses a light weight-based ˈ; eSpeak has its own stress rules.
- **Digits:** ASCII and Devanagari digit runs expand to Hindi cardinals before G2P (same surface
  for comparison when testing digit sentences).
- **C++ parity:** The C++ backend must match UTF-8 IPA byte-for-byte for multigraphs like tʃ;
  anusvara lookahead uses explicit UTF-8 prefix tests so plain ``t`` does not swallow ``tʃ``.
""".strip()
    )


if __name__ == "__main__":
    main()
