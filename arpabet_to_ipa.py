"""
CMU ARPAbet phone strings to IPA segments.

Maps individual tokens (e.g. ``AH0``, ``T``) and joins phone sequences into one IPA string.
"""

from __future__ import annotations

from typing import Iterable

_VOWEL_BODIES = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }
)

_VOWEL_BASE = {
    "AA": "ɑ",
    "AE": "æ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
}

_CONSONANTS = {
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


def _vowel_ipa(body: str, stress: int | None) -> str:
    if body == "AH":
        if stress == 0:
            return "ə"
        return "ʌ"
    if body == "ER":
        if stress == 0:
            return "ɚ"
        return "ɝ"
    return _VOWEL_BASE[body]


def arpabet_phone_to_ipa(token: str) -> str:
    """Map one CMU ARPAbet token (e.g. ``AH0``, ``T``) to IPA."""
    if not token:
        return ""
    stress = None
    if token[-1] in "012":
        stress = int(token[-1])
        body = token[:-1]
    else:
        body = token

    if body in _CONSONANTS:
        return _CONSONANTS[body]
    if body in _VOWEL_BODIES:
        sym = _vowel_ipa(body, stress)
        if stress is not None:
            if stress == 1:
                return "\u02C8" + sym
            if stress == 2:
                return "\u02CC" + sym
        return sym
    return token


def arpabet_words_to_ipa(phones: Iterable[str]) -> str:
    """Join ARPAbet phones into a single IPA string (no spaces between segments)."""
    return "".join(arpabet_phone_to_ipa(p) for p in phones if p)
