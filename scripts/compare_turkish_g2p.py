#!/usr/bin/env python3
"""
Compare **dependency-free** :mod:`turkish_rule_g2p` to a **library-backed** reference
(:func:`scripts.turkish_g2p_ref_library.ipa_phonemizer_espeak` / Epitran).

Prints side-by-side IPA and a short analysis of systematic differences (stress, inventory notation).

Does **not** fail if eSpeak/Epitran is missing — reference columns show ``(unavailable)``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

from turkish_rule_g2p import text_to_ipa  # noqa: E402

_ref_path = ROOT / "scripts" / "turkish_g2p_ref_library.py"
_spec = importlib.util.spec_from_file_location("turkish_g2p_ref_library", _ref_path)
_ref = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ref)
ipa_phonemizer_espeak = _ref.ipa_phonemizer_espeak
ipa_epitran = _ref.ipa_epitran


DEFAULT_SENTENCES = [
    "dağ",
    "değer",
    "kitap",
    "kızılçam",
    "İstanbul'da üç kitap var.",
    "1206-1227 arasında",
    "Ankara başkenttir.",
    "Türkiye Cumhuriyeti",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare rule-based Turkish G2P to library reference.")
    ap.add_argument(
        "--wiki-lines",
        type=int,
        metavar="N",
        help="Also compare first N lines of data/tr/wiki-text.txt (can be slow with eSpeak)",
    )
    args = ap.parse_args()

    ref_es = ipa_phonemizer_espeak("test") is not None
    ref_ep = ipa_epitran("test") is not None
    print("Reference availability:", file=sys.stderr)
    print(f"  phonemizer+eSpeak NG: {ref_es}", file=sys.stderr)
    print(f"  epitran tur-Latn:     {ref_ep}", file=sys.stderr)
    print(file=sys.stderr)

    lines = list(DEFAULT_SENTENCES)
    if args.wiki_lines:
        wiki = ROOT / "data" / "tr" / "wiki-text.txt"
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
- **Stress:** The rule engine uses default **final-syllable** ˈ on every orthographic word. Turkish
  has many exceptions (placenames like İstanbul / Ankara, some loans). eSpeak encodes richer
  lexical stress, so mismatches there are expected, not bugs in the letter-to-phone table.
- **ğ:** Rules use vowel length (ː) after a vowel and /ɰ/ or /j/ between vowels (back vs front).
  eSpeak may use a different tie-break or omit a separate glide symbol — compare qualitatively.
- **Palatal k/g:** Mapping to /c/ and /ɟ/ before front vowels matches standard descriptions; broad
  transcriptions sometimes keep abstract /k/ /ɡ/ — notation drift vs phonetic disagreement.
- **ö/ü:** Rule output uses IPA /ø/ and /y/; some references use /œ/ or tie-barred allophones.
- **Digits:** Rule-based text expands cardinals (and ``n-n`` ranges) to Turkish words before G2P;
  ensure the reference sentence is compared on the same surface (digits vs words) when testing.
- **Apostrophe:** ASCII ``'`` and curly quotes are **word-boundary** characters so ``İstanbul'da`` is
  three surface spans (the quotes are preserved in output). This avoids merging noun+suffix into one
  misspelled grapheme string (utf8proc can classify ASCII apostrophe as letter-like).
""".strip()
    )


if __name__ == "__main__":
    main()
