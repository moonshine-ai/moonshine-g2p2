#!/usr/bin/env python3
"""
Rule-based Ukrainian grapheme-to-phoneme (IPA) for TTS / vocoders.

Maps Cyrillic to IPA with **palatalization lookahead** (є і ї ю я, ь), **apostrophe**
blocking (м'ясо → /mjaso/), **г** /ɦ/ vs **ґ** /ɡ/, **и** /ɪ/ vs **і** /i/, **в** allophony
(ʋ before vowels / j-glide contexts, **w** before consonants and word-finally), and
digraphs **дж** /dʒ/, **дз** /dz/.

Default **penultimate stress** (ˈ before the vowel of the second-to-last syllable) is a coarse
stand-in for lexical stress (not marked in ordinary orthography). Digits expand to Ukrainian
cardinals via :mod:`ukrainian_numbers`.

No lexicon or morphological analyzer — voicing assimilation, cluster reduction, and lexical stress
are not modeled beyond this heuristic.
"""

from __future__ import annotations

import argparse
import re
import unicodedata

from ukrainian_numbers import expand_cardinal_digits_to_ukrainian_words, expand_digit_tokens_in_text

_DIGIT_PASS_THROUGH_RE = re.compile(r"^[0-9]+$")

# Letters participating in G2P (lowercase NFC); apostrophe handled separately.
_VOWEL_LETTERS = frozenset("аеєиіїоюяу")
_SOFT_VOWELS = frozenset("єіїюя")
_HARD_NO_PAL_BEFORE_SOFT = frozenset("жчшщ")  # stay “hard”; no ʲ before єі…
# Consonants that can receive ʲ before є і ї ю я or from ь
_PALATALIZABLE = frozenset("бвгґдзклмнпрстфхц")

_BASE_CONS = {
    "б": "b",
    "п": "p",
    "м": "m",
    "ф": "f",
    "г": "ɦ",
    "ґ": "ɡ",
    "д": "d",
    "т": "t",
    "н": "n",
    "л": "l",
    "р": "ɾ",
    "с": "s",
    "з": "z",
    "ж": "ʒ",
    "ш": "ʃ",
    "ч": "tʃ",
    "щ": "ʃtʃ",
    "ц": "ts",
    "к": "k",
    "х": "x",
}


def _strip_stress_marks(s: str) -> str:
    """Remove **all** combining marks except U+0308 (diaeresis) so NFD(ї) stays distinct from і."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFD", s):
        o = ord(ch)
        if unicodedata.category(ch) == "Mn" and o != 0x308:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def ukrainian_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def _next_letter_index(chars: list[str], start: int) -> int | None:
    j = start
    while j < len(chars):
        if chars[j] == "'":
            j += 1
            continue
        return j
    return None


def _peek_after_apostrophe(chars: list[str], cons_end: int) -> tuple[bool, int | None]:
    """If chars[cons_end+1] is apostrophe and a soft vowel follows, return (True, vowel_index)."""
    j = cons_end + 1
    if j >= len(chars) or chars[j] != "'":
        return False, None
    k = j + 1
    while k < len(chars) and chars[k] == "'":
        k += 1
    if k < len(chars) and chars[k] in _SOFT_VOWELS:
        return True, k
    return False, None


def _v_allophone(chars: list[str], i: int) -> str:
    """в ~ ʋ / w by following context (apostrophe skipped for lookahead)."""
    j = _next_letter_index(chars, i + 1)
    if j is None:
        return "w"
    nxt = chars[j]
    if nxt in _VOWEL_LETTERS or nxt == "й":
        return "ʋ"
    if nxt == "ь":
        return "w"
    return "w"


def _palatalize_last(pieces: list[str]) -> None:
    # Skip trailing non-syllabic markers
    for idx in range(len(pieces) - 1, -1, -1):
        p = pieces[idx]
        if not p:
            continue
        if p in ("ˈ", "ˌ"):
            continue
        if p.endswith("ʲ"):
            return
        if p in ("dʒ", "dz", "tʃ", "ts", "ʃtʃ", "ʒ", "ʃ"):
            return
        if len(p) == 1 or (len(p) == 2 and p[1] == "ʲ"):
            pieces[idx] = p + "ʲ"
            return
        # multi-char like tʃ — do not palatalize
        return


def _is_vowel_ipa_piece(p: str) -> bool:
    """True for nucleus/glide fragments (not consonant letters)."""
    if not p:
        return True
    if p == "j":
        return True
    if p[0] == "j" and len(p) > 1:
        return True
    return p[0] in "aɛeɪiou"


def _piece_ends_palatalized_consonant(pieces: list[str]) -> bool:
    for p in reversed(pieces):
        if not p or p == "ˈ":
            continue
        if _is_vowel_ipa_piece(p):
            continue
        if p in ("dʒ", "dz", "tʃ", "ts", "ʃtʃ", "ʒ", "ʃ"):
            return False
        return p.endswith("ʲ")
    return False


def _vowel_ipa(
    ch: str,
    *,
    force_j: bool,
    after_vowel_letter: bool,
    word_onset: bool,
    pieces: list[str],
) -> str:
    """Single vowel letter → IPA string (may be multiple phones, e.g. ji)."""
    if force_j or word_onset or after_vowel_letter:
        if ch == "я":
            return "ja"
        if ch == "ю":
            return "ju"
        if ch == "є":
            return "jɛ"
        if ch == "ї":
            return "ji"
    if ch == "я":
        return "a"
    if ch == "ю":
        return "u"
    if ch == "є":
        return "ɛ"
    if ch == "ї":
        return "i" if _piece_ends_palatalized_consonant(pieces) else "ji"
    if ch == "а":
        return "a"
    if ch == "е":
        return "ɛ"
    if ch == "и":
        return "ɪ"
    if ch == "і":
        return "i"
    if ch == "о":
        return "o"
    if ch == "у":
        return "u"
    return ""


def insert_primary_stress_penultimate(ipa: str) -> str:
    """Place ˈ before the vowel of the **penultimate** syllable (coarse Ukrainian heuristic)."""
    vowels = frozenset("aɛeɪiou")
    if "ˈ" in ipa or "ˌ" in ipa:
        return ipa
    chars = list(ipa)
    syll_starts: list[int] = []
    i = 0
    n = len(chars)
    while i < n:
        if chars[i] == "j" and i + 1 < n and chars[i + 1] in vowels:
            syll_starts.append(i)
            i += 2
            continue
        if chars[i] in vowels:
            syll_starts.append(i)
            i += 1
            continue
        i += 1
    if not syll_starts:
        return ipa
    if len(syll_starts) == 1:
        stress_at = syll_starts[0]
    else:
        stress_at = syll_starts[-2]
    return "".join(chars[:stress_at] + ["ˈ"] + chars[stress_at:])


def word_to_ipa(word: str, *, with_stress: bool = True, expand_cardinal_digits: bool = True) -> str:
    wraw = word.strip()
    if not wraw:
        return ""
    if expand_cardinal_digits and wraw.isdigit():
        phrase = expand_cardinal_digits_to_ukrainian_words(wraw)
        if phrase != wraw:
            return text_to_ipa(phrase, with_stress=with_stress, expand_cardinal_digits=False)
        return wraw
    if not expand_cardinal_digits and _DIGIT_PASS_THROUGH_RE.fullmatch(wraw):
        return wraw

    w = ukrainian_lower(wraw)
    w = _strip_stress_marks(w)
    chars: list[str] = []
    for c in w:
        if c in "'\u2019\u2018":
            chars.append("'")
        elif c.isalpha():
            chars.append(c)

    if not chars:
        return ""

    pieces: list[str] = []
    i = 0
    prev_was_vowel_letter = False
    word_onset = True
    force_j_vowel = False
    prev_was_hard_affricate = False

    while i < len(chars):
        if chars[i] == "'":
            i += 1
            continue

        if i + 1 < len(chars) and chars[i] == "д" and chars[i + 1] == "ж":
            pieces.append("dʒ")
            i += 2
            word_onset = False
            prev_was_vowel_letter = False
            prev_was_hard_affricate = True
            continue
        if i + 1 < len(chars) and chars[i] == "д" and chars[i + 1] == "з":
            pieces.append("dz")
            i += 2
            word_onset = False
            prev_was_vowel_letter = False
            prev_was_hard_affricate = True
            continue

        ch = chars[i]

        if ch == "ь":
            _palatalize_last(pieces)
            i += 1
            prev_was_hard_affricate = False
            continue

        if ch == "й":
            pieces.append("j")
            i += 1
            word_onset = False
            prev_was_vowel_letter = False
            prev_was_hard_affricate = False
            continue

        if ch in _VOWEL_LETTERS:
            ipa_v = _vowel_ipa(
                ch,
                force_j=force_j_vowel,
                after_vowel_letter=prev_was_vowel_letter,
                word_onset=word_onset,
                pieces=pieces,
            )
            if force_j_vowel:
                force_j_vowel = False
            pieces.append(ipa_v)
            i += 1
            word_onset = False
            prev_was_vowel_letter = True
            prev_was_hard_affricate = False
            continue

        # consonant
        if ch not in _BASE_CONS and ch != "в":
            i += 1
            continue

        apostrophe_block, vowel_i = _peek_after_apostrophe(chars, i)
        ni = _next_letter_index(chars, i + 1)
        next_ch = chars[ni] if ni is not None else None

        will_palatalize = False
        if prev_was_hard_affricate:
            will_palatalize = False
        elif (
            not apostrophe_block
            and next_ch in _SOFT_VOWELS
            and ch in _PALATALIZABLE
            and ch not in _HARD_NO_PAL_BEFORE_SOFT
        ):
            will_palatalize = True
        elif (
            not apostrophe_block
            and next_ch == "і"
            and ch in _PALATALIZABLE
            and ch not in _HARD_NO_PAL_BEFORE_SOFT
        ):
            will_palatalize = True

        if ch == "в":
            pieces.append(_v_allophone(chars, i))
        elif ch in _BASE_CONS:
            pieces.append(_BASE_CONS[ch])
        else:
            i += 1
            continue

        if will_palatalize:
            _palatalize_last(pieces)

        if apostrophe_block and vowel_i is not None:
            force_j_vowel = True
            i = vowel_i
            word_onset = False
            prev_was_vowel_letter = False
            prev_was_hard_affricate = False
            continue

        i += 1
        word_onset = False
        prev_was_vowel_letter = False
        prev_was_hard_affricate = ch in _HARD_NO_PAL_BEFORE_SOFT

    ipa = "".join(pieces)
    if with_stress:
        ipa = insert_primary_stress_penultimate(ipa)
    return ipa


def _is_uk_scan_word_char(ch: str) -> bool:
    """Word character for text scanning (mirrors C++ / :mod:`turkish_rule_g2p` style)."""
    o = ord(ch)
    if ch in "'\u2019\u2018":
        return True
    if ch.isascii() and (ch.isalnum() or ch == "_"):
        return True
    if ch.isalpha():
        return True
    if 0x0400 <= o <= 0x04FF:
        return True
    return False


def text_to_ipa(text: str, *, with_stress: bool = True, expand_cardinal_digits: bool = True) -> str:
    if expand_cardinal_digits:
        text = expand_digit_tokens_in_text(text)

    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            parts.append(" ")
            while i < n and text[i].isspace():
                i += 1
            continue
        if _is_uk_scan_word_char(ch):
            start = i
            i += 1
            while i < n and _is_uk_scan_word_char(text[i]):
                i += 1
            tok = text[start:i]
            sub = tok.split("-")
            ipas = [word_to_ipa(p, with_stress=with_stress, expand_cardinal_digits=False) for p in sub]
            parts.append("-".join(ipas))
            continue
        start = i
        i += 1
        while i < n:
            ch2 = text[i]
            if ch2.isspace() or _is_uk_scan_word_char(ch2):
                break
            i += 1
        parts.append(text[start:i])

    out = "".join(parts)
    out = re.sub(r" +", " ", out).strip()
    return out


def dialect_ids() -> list[str]:
    return ["uk", "uk-UA", "ukrainian"]


def main() -> None:
    p = argparse.ArgumentParser(description="Rule-based Ukrainian text to IPA.")
    p.add_argument("text", nargs="*", help="Ukrainian text (if empty, read stdin unless --stdin)")
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin")
    p.add_argument("--no-stress", action="store_true", help="Do not insert ˈ (penultimate heuristic).")
    p.add_argument(
        "--no-expand-digits",
        action="store_true",
        help="Leave digit sequences as digits (no spoken Ukrainian cardinal expansion).",
    )
    args = p.parse_args()
    if args.stdin or not args.text:
        phrase = __import__("sys").stdin.read()
    else:
        phrase = " ".join(args.text)
    print(
        text_to_ipa(
            phrase,
            with_stress=not args.no_stress,
            expand_cardinal_digits=not args.no_expand_digits,
        )
    )


if __name__ == "__main__":
    main()
