#!/usr/bin/env python3
"""
Sample ``wiki-text.txt`` lines for Brazilian and European Portuguese, run
:mod:`portuguese_rule_g2p` per word, and compare to eSpeak NG (``pt-br`` / ``pt``).

This repository stores wiki exports as ``data/pt_br/wiki-text.txt`` and
``data/pt_pt/wiki-text.txt`` (not ``wiki-text.tsv``).

Comparison uses :func:`portuguese_rule_g2p.coarse_ipa_for_compare` on both outputs
so minor stress/dot placement differences still count as matches when the phone
string is close enough.

Usage::

    python scripts/verify_portuguese_g2p_wiki.py --variant pt_br --max-lines 100
    python scripts/verify_portuguese_g2p_wiki.py --variant pt_pt --max-lines 100 --stats

``--stats`` adds a coarse-mismatch breakdown (lexicon hit vs rule/OOV path).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portuguese_rule_g2p import (  # noqa: E402
    coarse_ipa_for_compare,
    default_dict_path,
    default_espeak_voice,
    load_portuguese_lexicon,
    normalize_lookup_key,
    word_to_ipa,
)

_WORD_RE = re.compile(r"[\w'\-]+", flags=re.UNICODE)


def _wiki_path(variant: str) -> Path:
    return ROOT / "data" / variant / "wiki-text.txt"


def main() -> None:
    p = argparse.ArgumentParser(description="Verify Portuguese G2P vs eSpeak on wiki-text samples.")
    p.add_argument("--variant", choices=("pt_br", "pt_pt"), required=True)
    p.add_argument("--wiki", type=Path, default=None, help="Override wiki text file path.")
    p.add_argument("--max-lines", type=int, default=100, metavar="N", help="Scan at most N non-empty lines.")
    p.add_argument("--max-tokens", type=int, default=5000, metavar="N", help="Stop after N word tokens.")
    p.add_argument(
        "--show-mismatches",
        type=int,
        default=12,
        metavar="K",
        help="Print up to K coarse mismatches (0=none).",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print coarse-mismatch counts for lexicon vs OOV (rule) paths.",
    )
    args = p.parse_args()

    wiki = args.wiki or _wiki_path(args.variant)
    if not wiki.is_file():
        raise SystemExit(f"wiki file not found: {wiki}")

    variant = args.variant
    dict_path = default_dict_path(variant)
    lex: dict[str, str] = {}
    if dict_path.is_file():
        lex = load_portuguese_lexicon(dict_path, variant=variant)
    voice = default_espeak_voice(variant)

    try:
        from heteronym.espeak_heteronyms import EspeakPhonemizer, espeak_phonemize_ipa_raw
    except ImportError:
        EspeakPhonemizer = None  # type: ignore[misc,assignment]
        espeak_phonemize_ipa_raw = None  # type: ignore[misc,assignment]

    if EspeakPhonemizer is None:
        raise SystemExit("heteronym.espeak_heteronyms not importable; install project deps.")

    phon = EspeakPhonemizer(default_voice=voice)

    matched = 0
    compared = 0
    missing_es = 0
    mismatches: list[tuple[str, str, str]] = []
    miss_lex = 0
    miss_oov = 0

    lines_seen = 0
    for line in wiki.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        lines_seen += 1
        if lines_seen > args.max_lines:
            break
        for m in _WORD_RE.finditer(line):
            tok = m.group(0)
            key = normalize_lookup_key(tok)
            if len(key) < 2 or not any(c.isalpha() for c in key):
                continue
            compared += 1
            if compared > args.max_tokens:
                break

            ours = word_to_ipa(tok, variant=variant, lexicon=lex or None)
            if not ours:
                continue
            try:
                es_raw = espeak_phonemize_ipa_raw(phon, tok, voice=voice)
            except (AssertionError, OSError, RuntimeError):
                missing_es += 1
                continue
            if not es_raw:
                missing_es += 1
                continue
            # eSpeak returns a line; take first token's IPA if multiple.
            es_tok = es_raw.strip().split()[0] if es_raw else ""

            c_ours = coarse_ipa_for_compare(ours)
            c_es = coarse_ipa_for_compare(es_tok)
            if c_ours == c_es:
                matched += 1
            else:
                if args.stats:
                    if key in lex:
                        miss_lex += 1
                    else:
                        miss_oov += 1
                if len(mismatches) < args.show_mismatches:
                    mismatches.append((tok, ours, es_tok))
        if compared > args.max_tokens:
            break

    denom = compared - missing_es
    pct = 100.0 * matched / denom if denom else 0.0
    print(f"variant={variant} voice={voice} wiki={wiki}")
    print(f"tokens_seen={compared} espeak_ok={denom} coarse_exact_match={matched} ({pct:.1f}%)")
    if missing_es:
        print(f"espeak_skipped={missing_es}")
    if args.stats:
        miss_total = miss_lex + miss_oov
        print(
            f"coarse_mismatch_breakdown: lexicon={miss_lex} oov_rules={miss_oov} "
            f"(of {miss_total} mismatches)"
        )
    if args.show_mismatches and mismatches:
        print("sample mismatches (word | ours | espeak):")
        for w, o, e in mismatches:
            print(f"  {w} | {o} | {e}")


if __name__ == "__main__":
    main()
