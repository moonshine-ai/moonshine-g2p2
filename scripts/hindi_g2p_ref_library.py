#!/usr/bin/env python3
"""
Hindi G2P **reference** output using external tools (evaluation only; not part of Moonshine runtime).

**Preferred:** `phonemizer` + **eSpeak NG** (language code ``hi``).

  pip install phonemizer
  # Debian/Ubuntu: apt install espeak-ng

Example::

  python scripts/hindi_g2p_ref_library.py "हिन्दी भाषा"
  python scripts/hindi_g2p_ref_library.py --compare-rule "मैं ५ साल का हूँ।"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ipa_phonemizer_espeak(
    text: str,
    *,
    with_stress: bool = True,
    preserve_punctuation: bool = True,
) -> str | None:
    try:
        from phonemizer import phonemize
    except ImportError:
        return None
    t = text.strip()
    if not t:
        return ""
    try:
        return phonemize(
            t,
            language="hi",
            backend="espeak",
            with_stress=with_stress,
            preserve_punctuation=preserve_punctuation,
            strip=True,
        ).strip()
    except (OSError, RuntimeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Hindi reference G2P via phonemizer + eSpeak NG.")
    p.add_argument("text", nargs="*", help="UTF-8 Hindi text (default: stdin)")
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin")
    p.add_argument("--no-stress", action="store_true", help="Ask eSpeak for unstressed IPA")
    p.add_argument(
        "--compare-rule",
        action="store_true",
        help="Also print hindi_rule_g2p.text_to_ipa for contrast",
    )
    args = p.parse_args()
    if args.stdin or not args.text:
        phrase = sys.stdin.read()
    else:
        phrase = " ".join(args.text)

    ipa = ipa_phonemizer_espeak(phrase, with_stress=not args.no_stress)
    if ipa is None:
        print(
            "error: phonemizer + espeak-ng not available. pip install phonemizer; apt install espeak-ng",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(ipa)
    if args.compare_rule:
        from hindi_rule_g2p import text_to_ipa

        print("# rule-based (dependency-free):", file=sys.stderr)
        print(text_to_ipa(phrase), file=sys.stderr)


if __name__ == "__main__":
    main()
