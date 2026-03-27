"""
French cardinal expansion for digit sequences (e.g. years) and IPA for hyphenated compounds.

Orthography follows metropolitan French (soixante-dix, quatre-vingt). Hyphenated forms use a
static IPA table aligned with eSpeak NG ``fr`` for vocoder-friendly multi-stress strings.
"""

from __future__ import annotations

import re

# eSpeak-style IPA for hyphenated compounds (voice ``fr``); keeps multiple ˈ like eSpeak.
_CARDINAL_COMPOUND_IPA: dict[str, str] = {
    "cinquante": "sɛ̃kˈɑ̃t",
    "cinquante-cinq": "sɛ̃kˈɑ̃tsˈɛ̃k",
    "cinquante-deux": "sɛ̃kˈɑ̃tdˈø",
    "cinquante-et-un": "sɛ̃kˈɑ̃teˈœ̃",
    "cinquante-huit": "sɛ̃kˈɑ̃tyˈit",
    "cinquante-neuf": "sɛ̃kˈɑ̃tnˈœf",
    "cinquante-quatre": "sɛ̃kˈɑ̃tkˈatʁ",
    "cinquante-sept": "sɛ̃kˈɑ̃tsˈɛt",
    "cinquante-six": "sɛ̃kˈɑ̃tsˈis",
    "cinquante-trois": "sɛ̃kˈɑ̃ttʁwˈa",
    "dix-huit": "dˈizyˈit",
    "dix-neuf": "dˈiznˈœf",
    "dix-sept": "dˈisˈɛt",
    "quarante": "kaʁˈɑ̃t",
    "quarante-cinq": "kaʁˈɑ̃tsˈɛ̃k",
    "quarante-deux": "kaʁˈɑ̃tdˈø",
    "quarante-et-un": "kaʁˈɑ̃teˈœ̃",
    "quarante-huit": "kaʁˈɑ̃tyˈit",
    "quarante-neuf": "kaʁˈɑ̃tnˈœf",
    "quarante-quatre": "kaʁˈɑ̃tkˈatʁ",
    "quarante-sept": "kaʁˈɑ̃tsˈɛt",
    "quarante-six": "kaʁˈɑ̃tsˈis",
    "quarante-trois": "kaʁˈɑ̃ttʁwˈa",
    "quatre-vingt-cinq": "kˈatʁvˈɛ̃tsˈɛ̃k",
    "quatre-vingt-deux": "kˈatʁvˈɛ̃tdˈø",
    "quatre-vingt-dix": "kˈatʁvˈɛ̃dˈis",
    "quatre-vingt-dix-huit": "kˈatʁvˈɛ̃dˈizyˈit",
    "quatre-vingt-dix-neuf": "kˈatʁvˈɛ̃dˈiznˈœf",
    "quatre-vingt-dix-sept": "kˈatʁvˈɛ̃dˈisˈɛt",
    "quatre-vingt-douze": "kˈatʁvˈɛ̃dˈuz",
    "quatre-vingt-huit": "kˈatʁvˈɛ̃tyˈit",
    "quatre-vingt-neuf": "kˈatʁvˈɛ̃tnˈœf",
    "quatre-vingt-onze": "kˈatʁvˈɛ̃tˈɔ̃z",
    "quatre-vingt-quatorze": "kˈatʁvˈɛ̃katˈɔʁz",
    "quatre-vingt-quatre": "kˈatʁvˈɛ̃tkˈatʁ",
    "quatre-vingt-quinze": "kˈatʁvˈɛ̃kˈɛ̃z",
    "quatre-vingt-seize": "kˈatʁvˈɛ̃sˈɛz",
    "quatre-vingt-sept": "kˈatʁvˈɛ̃tsˈɛt",
    "quatre-vingt-six": "kˈatʁvˈɛ̃tsˈis",
    "quatre-vingt-treize": "kˈatʁvˈɛ̃tʁˈɛz",
    "quatre-vingt-trois": "kˈatʁvˈɛ̃ttʁwˈa",
    "quatre-vingt-un": "kˈatʁvˈɛ̃ˈœ̃",
    "quatre-vingts": "kˈatʁvˈɛ̃",
    "soixante-cinq": "swasˈɑ̃tsˈɛ̃k",
    "soixante-deux": "swasˈɑ̃tdˈø",
    "soixante-dix": "swasˈɑ̃tdˈis",
    "soixante-dix-huit": "swasˈɑ̃tdˈizyˈit",
    "soixante-dix-neuf": "swasˈɑ̃tdˈiznˈœf",
    "soixante-dix-sept": "swasˈɑ̃tdˈisˈɛt",
    "soixante-douze": "swasˈɑ̃tdˈuz",
    "soixante-et-onze": "swasˈɑ̃teˈɔ̃z",
    "soixante-huit": "swasˈɑ̃tyˈit",
    "soixante-neuf": "swasˈɑ̃tnˈœf",
    "soixante-onze": "swasˈɑ̃tˈɔ̃z",
    "soixante-quatorze": "swasˈɑ̃tkatˈɔʁz",
    "soixante-quatre": "swasˈɑ̃tkˈatʁ",
    "soixante-quinze": "swasˈɑ̃tkˈɛ̃z",
    "soixante-seize": "swasˈɑ̃tsˈɛz",
    "soixante-sept": "swasˈɑ̃tsˈɛt",
    "soixante-six": "swasˈɑ̃tsˈis",
    "soixante-treize": "swasˈɑ̃ttʁˈɛz",
    "soixante-trois": "swasˈɑ̃ttʁwˈa",
    "soixante-un": "swasˈɑ̃tˈœ̃",
    "trente": "tʁˈɑ̃t",
    "trente-cinq": "tʁˈɑ̃tsˈɛ̃k",
    "trente-deux": "tʁˈɑ̃tdˈø",
    "trente-et-un": "tʁˈɑ̃teˈœ̃",
    "trente-huit": "tʁˈɑ̃tyˈit",
    "trente-neuf": "tʁˈɑ̃tnˈœf",
    "trente-quatre": "tʁˈɑ̃tkˈatʁ",
    "trente-sept": "tʁˈɑ̃tsˈɛt",
    "trente-six": "tʁˈɑ̃tsˈis",
    "trente-trois": "tʁˈɑ̃ttʁwˈa",
    "vingt": "vˈɛ̃",
    "vingt-cinq": "vˈɛ̃tsˈɛ̃k",
    "vingt-deux": "vˈɛ̃tdˈø",
    "vingt-et-un": "vˈɛ̃teˈœ̃",
    "vingt-huit": "vˈɛ̃tyˈit",
    "vingt-neuf": "vˈɛ̃tnˈœf",
    "vingt-quatre": "vˈɛ̃tkˈatʁ",
    "vingt-sept": "vˈɛ̃tsˈɛt",
    "vingt-six": "vˈɛ̃tsˈis",
    "vingt-trois": "vˈɛ̃ttʁwˈa",
}

_DIGIT_WORD = (
    "zéro",
    "un",
    "deux",
    "trois",
    "quatre",
    "cinq",
    "six",
    "sept",
    "huit",
    "neuf",
)

_UNITS = (
    "zéro",
    "un",
    "deux",
    "trois",
    "quatre",
    "cinq",
    "six",
    "sept",
    "huit",
    "neuf",
    "dix",
    "onze",
    "douze",
    "treize",
    "quatorze",
    "quinze",
    "seize",
)


def cardinal_compound_ipa(word: str) -> str | None:
    """Return cached eSpeak-style IPA for *word* if it is a known hyphenated cardinal form."""
    return _CARDINAL_COMPOUND_IPA.get(word)


def _below_100(n: int) -> list[str]:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 17:
        return [_UNITS[n]]
    if n < 20:
        return [f"dix-{_UNITS[n - 10]}"]
    if n < 60:
        tens = (n // 10) * 10
        u = n % 10
        tens_w = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante"}[tens]
        if u == 0:
            return [tens_w]
        if u == 1:
            return [f"{tens_w}-et-un"]
        return [f"{tens_w}-{_UNITS[u]}"]
    if n < 70:
        return [f"soixante-{_UNITS[n - 60]}"]
    if n < 80:
        u = n - 70
        if u == 1:
            return ["soixante-et-onze"]
        return [f"soixante-{_UNITS[10 + u]}"]
    if n < 100:
        u = n - 80
        if u == 0:
            return ["quatre-vingts"]
        if u == 10:
            return ["quatre-vingt-dix"]
        if u < 10:
            return [f"quatre-vingt-{_UNITS[u]}"]
        if u < 17:
            return [f"quatre-vingt-{_UNITS[u]}"]
        return [f"quatre-vingt-dix-{_UNITS[u - 10]}"]
    return []


def _below_1000(n: int) -> list[str]:
    if n < 0 or n >= 1000:
        raise ValueError(n)
    if n == 0:
        return []
    h = n // 100
    r = n % 100
    parts: list[str] = []
    if h == 0:
        return _below_100(r)
    if h == 1:
        if r == 0:
            return ["cent"]
        return ["cent"] + _below_100(r)
    if r == 0:
        return [_UNITS[h], "cents"]
    return [_UNITS[h], "cent"] + _below_100(r)


def _below_1_000_000(n: int) -> list[str]:
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    if n < 1000:
        return _below_1000(n)
    q, r = divmod(n, 1000)
    parts: list[str] = []
    if q == 1:
        parts.append("mille")
    else:
        parts.extend(_below_1000(q))
        parts.append("mille")
    if r:
        parts.extend(_below_1000(r))
    return parts


def expand_cardinal_digits_to_french_words(s: str) -> str:
    """
    Replace a non-empty digit string with a French cardinal **word** phrase (space-separated).

    * Leading zeros (e.g. ``007``) are read digit-by-digit.
    * ``0`` → ``zéro``.
    * Integers ``> 999_999`` are left unchanged (returns *s*).
    """
    if not s.isdigit():
        return s
    if len(s) > 1 and s[0] == "0":
        return " ".join(_DIGIT_WORD[int(c)] for c in s)
    n = int(s)
    if n > 999_999:
        return s
    if n == 0:
        return "zéro"
    return " ".join(_below_1_000_000(n))


def expand_digit_tokens_in_text(text: str) -> str:
    r"""Expand ``\b\d+\b`` spans with :func:`expand_cardinal_digits_to_french_words`."""
    return re.sub(
        r"\b\d+\b",
        lambda m: expand_cardinal_digits_to_french_words(m.group(0)),
        text,
    )
