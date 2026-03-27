#!/usr/bin/env python3
"""Compare german_rule_g2p (lexicon + rules) vs eSpeak NG; print agreement tiers for review."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from german_rule_g2p import (
    espeak_ng_ipa_line,
    load_german_lexicon,
    text_to_ipa,
    word_to_ipa,
)

# Common German lemmas, short phrases, and a few tricky spellings.
ITEMS = [
    "der",
    "die",
    "das",
    "und",
    "ist",
    "in",
    "zu",
    "den",
    "von",
    "mit",
    "sich",
    "auf",
    "für",
    "nicht",
    "ein",
    "auch",
    "als",
    "an",
    "nach",
    "wie",
    "bei",
    "eine",
    "über",
    "Hallo",
    "Guten Tag",
    "Danke",
    "Bitte",
    "Entschuldigung",
    "schön",
    "groß",
    "Straße",
    "Fußball",
    "tschüss",
    "Zeitung",
    "Möglichkeit",
    "Arbeit",
    "Kind",
    "Kinder",
    "sprechen",
    "verstehen",
    "machen",
    "Buch",
    "Fisch",
    "Milch",
    "durch",
    "ich",
    "nichts",
    "Fräulein",
    "Müller",
    "Köln",
    "Quelle",
    "Schrank",
    "Pfanne",
    "Apfel",
    "essen",
    "lesen",
    "geben",
    "nehmen",
    "wissen",
    "können",
    "müssen",
    "sollen",
    "haben",
    "sein",
    "werden",
    "heute",
    "morgen",
    "gestern",
    "immer",
    "nie",
    "vielleicht",
    "natürlich",
    "Deutschland",
    "Berlin",
    "Wien",
    "Guten Morgen",
    "Auf Wiedersehen",
    "Wie geht es dir",
    "Ich verstehe nicht",
]


def _normalize_loose(s: str) -> str:
    """Aggressive normalization for broad agreement (not phonemic equality)."""
    t = s.lower()
    t = re.sub(r"[ˈˌ]", "", t)
    t = t.replace("ɾ", "r")
    t = t.replace("ʁ", "r")
    t = t.replace("χ", "x")
    t = t.replace("ç", "c")
    t = re.sub(r"ː", "", t)
    t = t.replace("ɐ̯", "ɐ")
    t = re.sub(r"([aeiouəøʏyɔɛɪʊ])n̩", r"\1n", t)
    t = re.sub(r"n̩", "n", t)
    t = t.replace("t͡s", "ts")
    t = t.replace("d͡ʒ", "dʒ")
    t = t.replace("p͡f", "pf")
    return t


def _lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
            )
        prev = cur
    return prev[lb]


def main() -> None:
    voice = "de"
    dict_path = _REPO / "models" / "de" / "dict.tsv"
    lex = load_german_lexicon(dict_path)

    rows: list[tuple[str, str, str | None, str, int, int]] = []
    missing = 0
    for item in ITEMS:
        r = (
            word_to_ipa(item, lexicon=lex)
            if " " not in item.strip()
            else text_to_ipa(item, lexicon=lex)
        )
        e = espeak_ng_ipa_line(item, voice=voice)
        if e is None:
            missing += 1
            rows.append((item, r, None, "no_espeak", 99, 99))
            continue
        nr, ne = _normalize_loose(r), _normalize_loose(e)
        ld = _lev(nr, ne)
        ld_raw = _lev(
            r.lower().replace("ˈ", "").replace("ˌ", ""),
            e.lower().replace("ˈ", "").replace("ˌ", ""),
        )
        if nr == ne:
            tier = "A_same_loose"
        elif ld <= 2:
            tier = "B_close"
        elif ld <= 5:
            tier = "C_moderate"
        else:
            tier = "D_large"
        rows.append((item, r, e, tier, ld, ld_raw))

    print(f"items={len(ITEMS)} espeak_missing={missing} voice={voice}\n")
    for tier_name in ("A_same_loose", "B_close", "C_moderate", "D_large", "no_espeak"):
        bucket = [x for x in rows if x[3] == tier_name]
        print(f"=== {tier_name} ({len(bucket)}) ===")
        for w, r, e, _t, ld, ldr in bucket:
            if e is None:
                print(f"  {w:32} rules={r}")
            else:
                print(f"  {w:32} rules={r}")
                print(f"  {'':32} espeak={e}  (lev_loose={ld} lev_raw~={ldr})")
        print()

    print("=== D_large detail (review) ===")
    for w, r, e, t, ld, ldr in rows:
        if t == "D_large":
            print(f"{w}\n  R {r}\n  E {e}\n  lev_loose={ld}\n")


if __name__ == "__main__":
    main()
