"""
German cardinal expansion for digit sequences (counts, years), for G2P pipelines.

Produces space-separated German **words** (orthography) so :mod:`german_rule_g2p` can look up
each token in ``data/de/dict.tsv`` / ``models/de/dict.tsv`` or fall back to rules.

* Leading zeros (e.g. ``007``) are read digit-by-digit (*null null sieben*).
* ``0`` → *null*.
* Hyphenated spans ``\\b\\d+-\\d+\\b`` become two cardinals separated by `` bis `` (common for
  year ranges in running text).
* Integers ``> 999_999`` are left unchanged (returns *s*).
"""

from __future__ import annotations

import re

_DIGIT_STANDALONE = (
    "null",
    "eins",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
)

# Compounds use *ein* (not *eins*) before *und* + tens.
_UNIT_COMPOUND = ("", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun")

_TEENS = {
    10: "zehn",
    11: "elf",
    12: "zwölf",
    13: "dreizehn",
    14: "vierzehn",
    15: "fünfzehn",
    16: "sechzehn",
    17: "siebzehn",
    18: "achtzehn",
    19: "neunzehn",
}

_TENS = (
    "",
    "",
    "zwanzig",
    "dreißig",
    "vierzig",
    "fünfzig",
    "sechzig",
    "siebzig",
    "achtzig",
    "neunzig",
)


def _under_100_word(n: int) -> str:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return _DIGIT_STANDALONE[n]
    if n < 20:
        return _TEENS[n]
    t, u = divmod(n, 10)
    tens_w = _TENS[t]
    if u == 0:
        return tens_w
    return _UNIT_COMPOUND[u] + "und" + tens_w


def _hundred_head(h: int) -> str:
    if h < 1 or h > 9:
        raise ValueError(h)
    if h == 1:
        return "hundert"
    stems = ("zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun")
    return stems[h - 2] + "hundert"


def _tokens_1_999(n: int) -> list[str]:
    if n < 1 or n > 999:
        raise ValueError(n)
    if n < 100:
        return [_under_100_word(n)]
    h, r = divmod(n, 100)
    head = _hundred_head(h)
    if r == 0:
        return [head]
    return [head, _under_100_word(r)]


def _tokens_thousands(q: int) -> list[str]:
    if q < 1 or q > 999:
        raise ValueError(q)
    if q == 1:
        return ["eintausend"]
    return _tokens_1_999(q) + ["tausend"]


def _below_1_000_000_tokens(n: int) -> list[str]:
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    if n < 1000:
        return _tokens_1_999(n)
    q, r = divmod(n, 1000)
    left = _tokens_thousands(q)
    if r == 0:
        return left
    return left + _tokens_1_999(r)


def expand_cardinal_digits_to_german_words(s: str) -> str:
    """
    Replace a non-empty digit string with German cardinal **words** (space-separated tokens).

    * Leading zeros → digit-by-digit.
    * ``> 999_999`` → *s* unchanged.
    """
    if not s.isdigit():
        return s
    if len(s) > 1 and s[0] == "0":
        return " ".join(_DIGIT_STANDALONE[int(c)] for c in s)
    n = int(s)
    if n > 999_999:
        return s
    if n == 0:
        return "null"
    return " ".join(_below_1_000_000_tokens(n))


def expand_digit_tokens_in_text(text: str) -> str:
    r"""
    Expand ``\b\d+-\d+\b`` as *A bis B*, then each ``\b\d+\b`` with
    :func:`expand_cardinal_digits_to_german_words`.
    """

    def _range(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return (
            f"{expand_cardinal_digits_to_german_words(a)} bis "
            f"{expand_cardinal_digits_to_german_words(b)}"
        )

    text = re.sub(r"\b(\d+)-(\d+)\b", _range, text)
    return re.sub(
        r"\b\d+\b",
        lambda m: expand_cardinal_digits_to_german_words(m.group(0)),
        text,
    )
