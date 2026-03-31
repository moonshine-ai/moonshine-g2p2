#!/usr/bin/env python3
"""
Hindi grapheme-to-phoneme (G2P) — broad IPA for Standard Hindi (Devanagari).

**Pure Python:** no Indic morphological analyzers. Uses a **lexicon** (``data/hi/dict.tsv``,
``word<TAB>ipa``) when the full token matches, otherwise **rule-based** Devanagari parsing:
conjuncts (halant), matras, nukta consonants, anusvara assimilation, chandrabindu
nasalization, optional **final schwa syncope**, and simple **stress** (penultimate moraic
weight heuristic).

**Digits:** ASCII and Devanagari digit runs expand to Hindi cardinal words via
:mod:`hindi_numbers` before G2P.

Limitations::
    Schwa syncope is only approximated (final inherent schwa + a small medial heuristic).
    Latin tokens are skipped (no output). Sandhi across words is not modeled.

CLI: ``python3 hindi_rule_g2p.py [--no-stress] [--no-expand-digits] …``
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from hindi_numbers import (
    expand_cardinal_digits_to_hindi_words,
    expand_devanagari_digit_runs_in_text,
    expand_digit_tokens_in_text,
)

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_HI_DICT = _REPO_ROOT / "data" / "hi" / "dict.tsv"

_LEXICON_CACHE: dict[str, dict[str, str]] = {}

# --- Unicode ---
_VIRAMA = 0x094D
_NUKTA = 0x093C
_ANUSVARA = 0x0902
_CHANDRA = 0x0901
_VISARGA = 0x0903
_ZWJ = 0x200D
_ZWNJ = 0x200C

# Independent vowels (अ … औ)
_INDEP_VOWEL: dict[int, str] = {
    0x0905: "ə",
    0x0906: "aː",
    0x0907: "ɪ",
    0x0908: "iː",
    0x0909: "ʊ",
    0x090A: "uː",
    0x090F: "eː",
    0x0910: "ɛː",
    0x0913: "oː",
    0x0914: "ɔː",
}

_MATRA: dict[int, str] = {
    0x093E: "aː",
    0x093F: "ɪ",
    0x0940: "iː",
    0x0941: "ʊ",
    0x0942: "uː",
    0x0947: "eː",
    0x0948: "ɛː",
    0x094B: "oː",
    0x094C: "ɔː",
}

# Base consonant → IPA (no nukta)
_BASE_CONS: dict[int, str] = {
    0x0915: "k",
    0x0916: "kʰ",
    0x0917: "g",
    0x0918: "gʰ",
    0x0919: "ŋ",
    0x091A: "tʃ",
    0x091B: "tʃʰ",
    0x091C: "dʒ",
    0x091D: "dʒʰ",
    0x091E: "ɲ",
    0x091F: "ʈ",
    0x0920: "ʈʰ",
    0x0921: "ɖ",
    0x0922: "ɖʰ",
    0x0923: "ɳ",
    0x0924: "t",
    0x0925: "tʰ",
    0x0926: "d",
    0x0927: "dʰ",
    0x0928: "n",
    0x092A: "p",
    0x092B: "pʰ",
    0x092C: "b",
    0x092D: "bʰ",
    0x092E: "m",
    0x092F: "j",
    0x0930: "r",
    0x0932: "l",
    0x0933: "ɭ",
    0x0935: "ʋ",
    0x0936: "ʃ",
    0x0937: "ʂ",
    0x0938: "s",
    0x0939: "ɦ",
}

_NUKTA_OVERRIDE: dict[int, str] = {
    0x0915: "q",
    0x0916: "x",
    0x0917: "ɣ",
    0x091C: "z",
    0x0921: "ɽ",
    0x0922: "ɽʰ",
    0x092B: "f",
}


def _is_devanagari_digit(cp: int) -> bool:
    return 0x0966 <= cp <= 0x096F


def _is_consonant(cp: int) -> bool:
    return cp in _BASE_CONS


def _cons_ipa(base: int, nukta: bool) -> str:
    if nukta and base in _NUKTA_OVERRIDE:
        return _NUKTA_OVERRIDE[base]
    return _BASE_CONS[base]


@dataclass
class Syllable:
    onset: list[str] = field(default_factory=list)  # consonant phones in cluster
    vowel: str | None = None  # None = halant-closed cluster (no nucleus)
    inherent_schwa: bool = False
    chandrabindu: bool = False
    anusvara: bool = False
    visarga: bool = False


def _nasal_for_place(first_onset: str | None) -> str:
    if not first_onset:
        return "ŋ"
    if first_onset.startswith("k") or first_onset.startswith("g") or first_onset == "q":
        return "ŋ"
    if first_onset.startswith("tʃ") or first_onset.startswith("dʒ") or first_onset == "ɲ":
        return "ɲ"
    if first_onset.startswith("ʈ") or first_onset.startswith("ɖ") or first_onset in ("ɳ", "ɽ", "ɽʰ"):
        return "ɳ"
    if first_onset.startswith("t") or first_onset.startswith("d") or first_onset == "n":
        return "n"
    if first_onset.startswith("p") or first_onset.startswith("b") or first_onset == "m":
        return "m"
    return "n"


def _parse_devanagari_to_syllables(word: str) -> list[Syllable] | None:
    """Return syllables or None if word has no Devanagari letters."""
    s = unicodedata.normalize("NFC", word)
    cps = [ord(ch) for ch in s]
    n = len(cps)
    i = 0
    out: list[Syllable] = []

    def skip_joiners() -> None:
        nonlocal i
        while i < n and cps[i] in (_ZWJ, _ZWNJ):
            i += 1

    has_letter = False
    while i < n:
        skip_joiners()
        if i >= n:
            break
        cp = cps[i]
        if _is_devanagari_digit(cp):
            return None
        if cp in _INDEP_VOWEL:
            has_letter = True
            out.append(Syllable(onset=[], vowel=_INDEP_VOWEL[cp], inherent_schwa=False))
            i += 1
            skip_joiners()
            if i < n and cps[i] == _CHANDRA:
                out[-1].chandrabindu = True
                i += 1
            if i < n and cps[i] == _ANUSVARA:
                out[-1].anusvara = True
                i += 1
            if i < n and cps[i] == _VISARGA:
                out[-1].visarga = True
                i += 1
            continue
        if not _is_consonant(cp):
            i += 1
            continue

        has_letter = True
        onset: list[str] = []
        halant_end = False
        while i < n:
            skip_joiners()
            if i >= n or not _is_consonant(cps[i]):
                break
            base = cps[i]
            i += 1
            nukta = i < n and cps[i] == _NUKTA
            if nukta:
                i += 1
            onset.append(_cons_ipa(base, nukta))
            if i < n and cps[i] == _VIRAMA:
                i += 1
                skip_joiners()
                if i < n and _is_consonant(cps[i]):
                    continue
                halant_end = True
                break
            break

        if halant_end:
            out.append(Syllable(onset=onset, vowel=None, inherent_schwa=False))
            if i < n and cps[i] == _VISARGA:
                out[-1].visarga = True
                i += 1
            continue

        if i < n and cps[i] in _MATRA:
            v = _MATRA[cps[i]]
            i += 1
            inc = False
        else:
            v = "ə"
            inc = True
        sy = Syllable(onset=onset, vowel=v, inherent_schwa=inc)
        if i < n and cps[i] == _CHANDRA:
            sy.chandrabindu = True
            i += 1
        if i < n and cps[i] == _ANUSVARA:
            sy.anusvara = True
            i += 1
        if i < n and cps[i] == _VISARGA:
            sy.visarga = True
            i += 1
        out.append(sy)

    if not has_letter:
        return None
    return out


def _apply_schwa_syncope(syls: list[Syllable]) -> None:
    """In-place: final inherent schwa; light medial schwa before sonorant/heavy next (heuristic)."""
    if len(syls) < 2:
        return
    last = syls[-1]
    if last.vowel == "ə" and last.inherent_schwa:
        last.vowel = ""
        last.inherent_schwa = False

    # Medial syncope (heuristic): inherent schwa before affricate/fricative onsets (e.g. समझना).
    # Do not use plain nasals/stops here — that wrongly deletes the vowel in कमल (क…म).
    i = 0
    while i < len(syls) - 1:
        a, b = syls[i], syls[i + 1]
        bo = b.onset[0] if b.onset else ""
        if a.vowel == "ə" and a.inherent_schwa and (
            bo.startswith("dʒ") or bo.startswith("tʃ") or bo.startswith("ʃ") or bo == "ɲ"
        ):
            a.vowel = ""
            a.inherent_schwa = False
        i += 1


def _syllable_weight(s: Syllable) -> int:
    """Mora-ish weight for stress: long vowels / diphthongs heavier."""
    if not s.vowel:
        return 0
    if s.vowel in ("aː", "iː", "uː", "eː", "oː", "ɛː", "ɔː"):
        return 2
    return 1


def _assign_stress(ipa_syllables: list[str], weights: list[int]) -> str:
    if not ipa_syllables:
        return ""
    if len(ipa_syllables) == 1:
        return ipa_syllables[0]
    best_i = 0
    best_w = -1
    for i, w in enumerate(weights):
        if w > best_w:
            best_w = w
            best_i = i
    if best_w <= 0:
        best_i = max(0, len(ipa_syllables) - 2)
    parts = []
    for i, p in enumerate(ipa_syllables):
        if i == best_i and best_w > 0:
            parts.append("ˈ" + p)
        else:
            parts.append(p)
    return ".".join(parts)


def _render_syllables(syls: list[Syllable], with_stress: bool) -> str:
    if not syls:
        return ""

    def render_one(j: int) -> str:
        s = syls[j]
        body = "".join(s.onset)
        if s.vowel is None:
            if s.visarga:
                body += "ɦ"
            return body
        v = s.vowel
        if s.chandrabindu and v:
            v = v + "̃"
        if s.anusvara:
            nxt_onset = None
            for k in range(j + 1, len(syls)):
                if syls[k].onset:
                    nxt_onset = syls[k].onset[0]
                    break
            if nxt_onset is None:
                v = (v or "") + "̃"
            else:
                body += _nasal_for_place(nxt_onset)
        body += v
        if s.visarga:
            body += "ɦ"
        return body

    raw = [render_one(j) for j in range(len(syls))]
    weights = [_syllable_weight(syls[j]) for j in range(len(syls))]
    merged = [r for r in raw if r]
    if not with_stress or not merged:
        return ".".join(merged)
    return _assign_stress(merged, weights)


def _load_lexicon(path: Path) -> dict[str, str]:
    p = str(path.resolve())
    if p in _LEXICON_CACHE:
        return _LEXICON_CACHE[p]
    lex: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                continue
            w, ipa = line.split("\t", 1)
            w = w.strip()
            ipa = ipa.strip()
            if w and ipa and w not in lex:
                lex[w] = ipa
    _LEXICON_CACHE[p] = lex
    return lex


def _strip_edges_punct(w: str) -> tuple[str, str, str]:
    """Return (left, core, right) ASCII/Unicode punctuation stripped from core word."""
    i = 0
    j = len(w)
    while i < j and unicodedata.category(w[i]) in ("Po", "Pd", "Pe", "Pi", "Ps", "Pf", "Pc"):
        i += 1
    while j > i and unicodedata.category(w[j - 1]) in ("Po", "Pd", "Pe", "Pi", "Ps", "Pf", "Pc"):
        j -= 1
    return w[:i], w[i:j], w[j:]


def _has_devanagari(s: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in s)


def devanagari_word_to_ipa(
    word: str,
    *,
    lexicon: dict[str, str] | None = None,
    with_stress: bool = True,
    expand_cardinal_digits: bool = False,
) -> str:
    _left, core, _right = _strip_edges_punct(word)
    if not core:
        return ""
    if expand_cardinal_digits and re.fullmatch(r"\d+", core):
        core = expand_cardinal_digits_to_hindi_words(core)
    nfc = unicodedata.normalize("NFC", core)
    lex = lexicon if lexicon is not None else _load_lexicon(_DEFAULT_HI_DICT)
    if nfc in lex:
        return lex[nfc]
    syls = _parse_devanagari_to_syllables(nfc)
    if syls is None:
        return ""
    _apply_schwa_syncope(syls)
    return _render_syllables(syls, with_stress=with_stress)


def text_to_ipa(
    text: str,
    *,
    dict_path: Path | None = None,
    with_stress: bool = True,
    expand_cardinal_digits: bool = True,
) -> str:
    path = dict_path if dict_path is not None else _DEFAULT_HI_DICT
    lex = _load_lexicon(path)
    t = text
    if expand_cardinal_digits:
        t = expand_digit_tokens_in_text(t)
        t = expand_devanagari_digit_runs_in_text(t)
    parts_out: list[str] = []
    for raw in t.split():
        if _has_devanagari(raw):
            ipa = devanagari_word_to_ipa(
                raw,
                lexicon=lex,
                with_stress=with_stress,
                expand_cardinal_digits=False,
            )
            if ipa:
                parts_out.append(ipa)
        elif re.fullmatch(r"\d+", _strip_edges_punct(raw)[1]) and expand_cardinal_digits:
            _l, c, _r = _strip_edges_punct(raw)
            ipa = devanagari_word_to_ipa(
                c,
                lexicon=lex,
                with_stress=with_stress,
                expand_cardinal_digits=True,
            )
            if ipa:
                parts_out.append(ipa)
    return " ".join(parts_out)


def dialect_ids() -> list[str]:
    return ["hi", "hi-IN", "hindi"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Hindi Devanagari → IPA (lexicon + rules).")
    ap.add_argument("text", nargs="*", help="UTF-8 text (default: read stdin)")
    ap.add_argument("--dict", type=Path, default=None, help="Path to dict.tsv")
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--no-expand-digits", action="store_true")
    ap.add_argument("--stdin", action="store_true")
    args = ap.parse_args()
    if args.stdin or not args.text:
        phrase = __import__("sys").stdin.read()
    else:
        phrase = " ".join(args.text)
    print(
        text_to_ipa(
            phrase.rstrip("\n"),
            dict_path=args.dict,
            with_stress=not args.no_stress,
            expand_cardinal_digits=not args.no_expand_digits,
        )
    )


if __name__ == "__main__":
    main()
