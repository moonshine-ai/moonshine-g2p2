#!/usr/bin/env python3
"""
Ukrainian G2P **reference** output using external tools (for evaluation, not bundled in Moonshine).

**Preferred stack:** `phonemizer` + **eSpeak NG** (language code ``uk``).

  pip install phonemizer
  # Debian/Ubuntu: apt install espeak-ng

**Fallback:** `epitran` if a suitable Ukrainian map is available (often thinner than eSpeak).

Example::

  python scripts/ukrainian_g2p_ref_library.py "Українська мова."
  python scripts/ukrainian_g2p_ref_library.py --compare-rule "Було 5 котів."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ipa_phonemizer_espeak(text: str) -> str | None:
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
            language="uk",
            backend="espeak",
            with_stress=True,
            preserve_punctuation=True,
            strip=False,
        ).strip()
    except (OSError, RuntimeError, ValueError):
        return None


def ipa_epitran(text: str) -> str | None:
    try:
        import epitran  # type: ignore[import-untyped]
    except ImportError:
        return None
    for code in ("ukr-Cyrl", "ukr-Cyrl-bab", "ukr-Cyrl-s12"):
        try:
            tr = epitran.Epitran(code)
        except (ValueError, FileNotFoundError):
            continue
        return tr.transliterate(text.strip())
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Ukrainian reference G2P via phonemizer/eSpeak or Epitran.")
    p.add_argument("text", nargs="*", help="UTF-8 Ukrainian text (default: read stdin)")
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin")
    p.add_argument(
        "--backend",
        choices=("auto", "espeak", "epitran"),
        default="auto",
        help="auto: eSpeak via phonemizer if available, else Epitran",
    )
    p.add_argument(
        "--compare-rule",
        action="store_true",
        help="Also print a line from ukrainian_rule_g2p.text_to_ipa for contrast",
    )
    args = p.parse_args()
    if args.stdin or not args.text:
        phrase = sys.stdin.read()
    else:
        phrase = " ".join(args.text)

    ipa: str | None = None
    backend_used = ""
    if args.backend == "epitran":
        ipa = ipa_epitran(phrase)
        backend_used = "epitran"
    elif args.backend == "espeak":
        ipa = ipa_phonemizer_espeak(phrase)
        backend_used = "phonemizer+espeak"
    else:
        ipa = ipa_phonemizer_espeak(phrase)
        if ipa is not None:
            backend_used = "phonemizer+espeak"
        else:
            ipa = ipa_epitran(phrase)
            backend_used = "epitran"

    if ipa is None:
        print(
            "error: no backend available. Install phonemizer + espeak-ng, or pip install epitran.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"# backend: {backend_used}", file=sys.stderr)
    print(ipa)
    if args.compare_rule:
        from ukrainian_rule_g2p import text_to_ipa

        print("# rule-based (dependency-free):", file=sys.stderr)
        print(text_to_ipa(phrase))


if __name__ == "__main__":
    main()
