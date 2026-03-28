"""
Spanish cardinal expansion for digit sequences (counts, years), for G2P pipelines.

Produces space-separated Spanish **words** (orthography) so :mod:`spanish_rule_g2p` can G2P each
token. Works for Latin American and Peninsular presets (cardinal words are shared; G2P differs on
⟨z⟩/⟨c⟩ only).

* Leading zeros → digit-by-digit (*cero cero siete*).
* ``0`` → *cero*.
* Hyphenated spans ``\\b\\d+-\\d+\\b`` become two cardinals separated by `` - ``.
* Integers ``> 999_999`` are left unchanged.
"""

from __future__ import annotations

import re

_DIGIT = (
    "cero",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
)

# 10–29 (inclusive) as single orthographic words.
_SPECIAL_UNDER_30: dict[int, str] = {
    10: "diez",
    11: "once",
    12: "doce",
    13: "trece",
    14: "catorce",
    15: "quince",
    16: "dieciséis",
    17: "diecisiete",
    18: "dieciocho",
    19: "diecinueve",
    20: "veinte",
    21: "veintiuno",
    22: "veintidós",
    23: "veintitrés",
    24: "veinticuatro",
    25: "veinticinco",
    26: "veintiséis",
    27: "veintisiete",
    28: "veintiocho",
    29: "veintinueve",
}

_TENS = (
    "",
    "",
    "veinte",
    "treinta",
    "cuarenta",
    "cincuenta",
    "sesenta",
    "setenta",
    "ochenta",
    "noventa",
)

_HUNDREDS = (
    "",
    "",
    "doscientos",
    "trescientos",
    "cuatrocientos",
    "quinientos",
    "seiscientos",
    "setecientos",
    "ochocientos",
    "novecientos",
)


def _under_100_tokens(n: int) -> list[str]:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return [_DIGIT[n]]
    if n < 30:
        return [_SPECIAL_UNDER_30[n]]
    t, u = divmod(n, 10)
    tens = _TENS[t]
    if u == 0:
        return [tens]
    return [tens, "y", _DIGIT[u]]


def _below_1000_tokens(n: int) -> list[str]:
    if n < 0 or n >= 1000:
        raise ValueError(n)
    if n < 100:
        return _under_100_tokens(n)
    h, r = divmod(n, 100)
    if h == 1:
        if r == 0:
            return ["cien"]
        return ["ciento", *_under_100_tokens(r)]
    head = _HUNDREDS[h]
    if r == 0:
        return [head]
    return [head, *_under_100_tokens(r)]


def _below_1_000_000_tokens(n: int) -> list[str]:
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    if n < 1000:
        return _below_1000_tokens(n)
    q, r = divmod(n, 1000)
    if q == 1:
        left = ["mil"]
    else:
        left = _below_1000_tokens(q) + ["mil"]
    if r == 0:
        return left
    return left + _below_1000_tokens(r)


def expand_cardinal_digits_to_spanish_words(s: str) -> str:
    if not s.isdigit():
        return s
    if len(s) > 1 and s[0] == "0":
        return " ".join(_DIGIT[int(c)] for c in s)
    n = int(s)
    if n > 999_999:
        return s
    if n == 0:
        return "cero"
    return " ".join(_below_1_000_000_tokens(n))


def expand_digit_tokens_in_text(text: str) -> str:
    r"""
    Expand ``\b\d+-\d+\b`` as ``A - B`` (two cardinals), then each ``\b\d+\b`` with
    :func:`expand_cardinal_digits_to_spanish_words`.
    """

    def _range(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return (
            f"{expand_cardinal_digits_to_spanish_words(a)} - "
            f"{expand_cardinal_digits_to_spanish_words(b)}"
        )

    text = re.sub(r"\b(\d+)-(\d+)\b", _range, text)
    return re.sub(
        r"\b\d+\b",
        lambda m: expand_cardinal_digits_to_spanish_words(m.group(0)),
        text,
    )
