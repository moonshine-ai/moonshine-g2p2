#!/usr/bin/env python3
"""
Rule-based Turkish grapheme-to-phoneme (IPA) conversion for TTS / vocoders.

Modern Turkish orthography is nearly phonemic; this module maps letters to IPA with
contextual **ğ** (lengthening vs intervocalic glide), **k/g** palatalization before front vowels,
and default **final-syllable** primary stress (ˈ before the last vowel).

Digits expand to Turkish cardinals via :mod:`turkish_numbers` (up to 999_999; hyphen ranges
``1933-1945`` → two cardinals separated by `` - ``).

No morphological analyzer or lexicon — loanwords and placenames with non-final stress are not
special-cased (see :mod:`scripts.turkish_g2p_ref_library` for an eSpeak-based reference line).
"""

from __future__ import annotations

import argparse
import re
import unicodedata

from turkish_numbers import expand_cardinal_digits_to_turkish_words, expand_digit_tokens_in_text

_VOWELS = frozenset("aeıioöuüâêîôû")
_FRONT_VOWELS = frozenset("eiöü")
_BACK_VOWELS = frozenset("aıou")

_DIGIT_PASS_THROUGH_RE = re.compile(r"^[0-9]+$")

_SIMPLE_MAP: dict[str, str] = {
    "a": "a",
    "b": "b",
    "c": "dʒ",
    "ç": "tʃ",
    "d": "d",
    "e": "e",
    "f": "f",
    "h": "h",
    "ı": "ɯ",
    "i": "i",
    "j": "ʒ",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "ö": "ø",
    "p": "p",
    "r": "ɾ",
    "s": "s",
    "ş": "ʃ",
    "t": "t",
    "u": "u",
    "ü": "y",
    "v": "v",
    "y": "j",
    "z": "z",
    # Rare in native words; practical pass-through for Latin text
    "q": "k",
    "w": "v",
    "x": "ks",
    # Diacritic variants sometimes seen
    "â": "a",
    "ê": "e",
    "î": "i",
    "ô": "o",
    "û": "u",
}


def turkish_lower(s: str) -> str:
    """Turkish-specific lowercasing (İ→i, I→ı) plus NFC."""
    t = unicodedata.normalize("NFC", s)
    t = t.replace("İ", "i").replace("I", "ı")
    return t.lower()


def _prev_letter(w: str, i: int) -> str | None:
    for j in range(i - 1, -1, -1):
        if w[j] in _SIMPLE_MAP or w[j] in "gğk":
            return w[j]
    return None


def _next_letter(w: str, i: int) -> str | None:
    for j in range(i + 1, len(w)):
        if w[j] in _SIMPLE_MAP or w[j] in "gğk":
            return w[j]
    return None


def _next_vowel(w: str, start: int) -> str | None:
    for j in range(start, len(w)):
        c = w[j]
        if c in _VOWELS:
            return c
    return None


def _last_vowel_before(w: str, end: int) -> str | None:
    for j in range(end - 1, -1, -1):
        if w[j] in _VOWELS:
            return w[j]
    return None


def _harmony_vowel_for_consonant(w: str, cons_index: int) -> str | None:
    nxt = _next_vowel(w, cons_index + 1)
    if nxt is not None:
        return nxt
    return _last_vowel_before(w, cons_index)


def _is_front_vowel(ch: str) -> bool:
    return ch in _FRONT_VOWELS


def _map_consonant_k_or_g(ch: str, w: str, i: int) -> str:
    hv = _harmony_vowel_for_consonant(w, i)
    if hv is None:
        return "k" if ch == "k" else "ɡ"
    front = _is_front_vowel(hv)
    if ch == "k":
        return "c" if front else "k"
    return "ɟ" if front else "ɡ"


def _letters_only(w: str) -> str:
    out: list[str] = []
    for c in w:
        if c == "'":
            continue
        if c in _SIMPLE_MAP or c in "gğk":
            out.append(c)
    return "".join(out)


def word_to_ipa(word: str, *, with_stress: bool = True, expand_cardinal_digits: bool = True) -> str:
    wraw = word.strip()
    if not wraw:
        return ""
    if expand_cardinal_digits and wraw.isdigit():
        phrase = expand_cardinal_digits_to_turkish_words(wraw)
        if phrase != wraw:
            return text_to_ipa(phrase, with_stress=with_stress, expand_cardinal_digits=False)
        return wraw
    if not expand_cardinal_digits and _DIGIT_PASS_THROUGH_RE.fullmatch(wraw):
        return wraw

    w = turkish_lower(wraw)
    w = w.replace("'", "")
    letters = _letters_only(w)
    if not letters:
        return ""

    pieces: list[str] = []
    i = 0
    n = len(letters)
    while i < n:
        c = letters[i]
        if c == "ğ":
            prev = _prev_letter(letters, i)
            nxt = _next_letter(letters, i)
            if prev is not None and prev in _VOWELS and nxt is not None and nxt in _VOWELS:
                pieces.append("ɰ" if prev in _BACK_VOWELS else "j")
            elif prev is not None and prev in _VOWELS:
                for k in range(len(pieces) - 1, -1, -1):
                    seg = pieces[k]
                    if not seg:
                        continue
                    last_cp = seg[-1]
                    if last_cp == "ː":
                        continue
                    if last_cp in "aeiɯouøy":
                        pieces[k] = seg + "ː"
                        break
            # Else: silent (e.g. consonant context); omit
            i += 1
            continue
        if c == "k":
            pieces.append(_map_consonant_k_or_g("k", letters, i))
            i += 1
            continue
        if c == "g":
            pieces.append(_map_consonant_k_or_g("g", letters, i))
            i += 1
            continue
        m = _SIMPLE_MAP.get(c)
        if m is not None:
            pieces.append(m)
        i += 1

    ipa = "".join(pieces)
    if with_stress:
        ipa = insert_primary_stress_final_syllable(ipa)
    return ipa


def insert_primary_stress_final_syllable(ipa: str) -> str:
    """Place ˈ immediately before the last vowel nucleus (supports trailing length ː)."""
    vowels = frozenset("aeiɯouøy")
    j = len(ipa) - 1
    while j >= 0:
        if ipa[j] == "ː" and j >= 1:
            j -= 1
            if ipa[j] in vowels:
                if j >= 1 and ipa[j - 1] == "ˈ":
                    return ipa
                return ipa[:j] + "ˈ" + ipa[j : j + 2] + ipa[j + 2 :]
            j -= 1
            continue
        if ipa[j] in vowels:
            if j >= 1 and ipa[j - 1] == "ˈ":
                return ipa
            return ipa[:j] + "ˈ" + ipa[j:]
        j -= 1
    return ipa


def text_to_ipa(
    text: str,
    *,
    with_stress: bool = True,
    expand_cardinal_digits: bool = True,
) -> str:
    if expand_cardinal_digits:
        text = expand_digit_tokens_in_text(text)

    def is_turk_word_char(ch: str) -> bool:
        o = ord(ch)
        # Apostrophe separates proper nouns from suffixes (İstanbul'da); do not merge into one token.
        if ch in "'\u2019\u2018":
            return False
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            return True
        # Turkish + common Latin letters
        if ch in "çğıöşüÇĞİÖŞÜ":
            return True
        if 0xC0 <= o <= 0x024F and ch.isalpha():
            return True
        return False

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
        if is_turk_word_char(ch):
            start = i
            i += 1
            while i < n and is_turk_word_char(text[i]):
                i += 1
            tok = text[start:i]
            parts.append(word_to_ipa(tok, with_stress=with_stress, expand_cardinal_digits=False))
            continue
        start = i
        i += 1
        while i < n:
            ch2 = text[i]
            if ch2.isspace() or is_turk_word_char(ch2):
                break
            i += 1
        parts.append(text[start:i])

    out = "".join(parts)
    out = re.sub(r" +", " ", out).strip()
    return out


def dialect_ids() -> list[str]:
    return dedupe_ids(["tr", "tr-TR", "turkish"])


def dedupe_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        k = x.casefold().replace("_", "-")
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rule-based Turkish text to IPA.")
    p.add_argument("text", nargs="*", help="Turkish text (if empty, read stdin unless --stdin)")
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin")
    p.add_argument("--no-stress", action="store_true", help="Omit primary stress ˈ")
    p.add_argument(
        "--no-expand-digits",
        action="store_true",
        help="Leave digit sequences as digits (no spoken Turkish cardinal expansion).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    phrase = " ".join(args.text) if args.text else ""
    if args.stdin or not phrase:
        import sys

        phrase = sys.stdin.read()
    ipa = text_to_ipa(
        phrase,
        with_stress=not args.no_stress,
        expand_cardinal_digits=not args.no_expand_digits,
    )
    print(ipa)


if __name__ == "__main__":
    main()
