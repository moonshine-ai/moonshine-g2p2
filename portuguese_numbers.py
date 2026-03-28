"""
Portuguese cardinal expansion for digit sequences (counts, years), for G2P pipelines.

Produces space-separated Portuguese **words** (orthography) so :mod:`portuguese_rule_g2p` can
look up each token in ``data/pt_br/dict.tsv`` / ``data/pt_pt/dict.tsv`` or fall back to rules.

* Leading zeros (e.g. ``007``) are read digit-by-digit.
* ``0`` → *zero*.
* Hyphenated spans ``\\b\\d+-\\d+\\b`` become two cardinals separated by `` - `` (same spirit as
  Italian / French).
* Integers ``> 999_999`` are left unchanged (returns *s*).
* Teens 16–17 and 19 follow **Brazil** (*dezesseis*, …) vs **Portugal** (*dezasseis*, *dezanove*, …)
  when *variant* is ``pt_pt``.
"""

from __future__ import annotations

import re

_DIGIT_WORD = (
    "zero",
    "um",
    "dois",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
)

_TENS = (
    "",
    "",
    "vinte",
    "trinta",
    "quarenta",
    "cinquenta",
    "sessenta",
    "setenta",
    "oitenta",
    "noventa",
)

_HUNDREDS = (
    "",
    "",
    "duzentos",
    "trezentos",
    "quatrocentos",
    "quinhentos",
    "seiscentos",
    "setecentos",
    "oitocentos",
    "novecentos",
)


def _teens_word(n: int, variant: str) -> str:
    if n < 11 or n > 19:
        raise ValueError(n)
    if n == 18:
        return "dezoito"
    if variant == "pt_pt":
        ep = {
            11: "onze",
            12: "doze",
            13: "treze",
            14: "catorze",
            15: "quinze",
            16: "dezasseis",
            17: "dezassete",
            19: "dezanove",
        }
        return ep[n]
    br = {
        11: "onze",
        12: "doze",
        13: "treze",
        14: "catorze",
        15: "quinze",
        16: "dezesseis",
        17: "dezessete",
        19: "dezenove",
    }
    return br[n]


def _under_100_tokens(n: int, variant: str) -> list[str]:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return [_DIGIT_WORD[n]]
    if n == 10:
        return ["dez"]
    if n < 20:
        return [_teens_word(n, variant)]
    t, u = divmod(n, 10)
    if u == 0:
        return [_TENS[t]]
    return [_TENS[t], "e", _DIGIT_WORD[u]]


def _below_1000_tokens(n: int, variant: str) -> list[str]:
    if n < 0 or n >= 1000:
        raise ValueError(n)
    if n < 100:
        return _under_100_tokens(n, variant)
    h, r = divmod(n, 100)
    if h == 1:
        if r == 0:
            return ["cem"]
        return ["cento", "e", *_under_100_tokens(r, variant)]
    head = _HUNDREDS[h]
    if r == 0:
        return [head]
    return [head, "e", *_under_100_tokens(r, variant)]


def _below_1_000_000_tokens(n: int, variant: str) -> list[str]:
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    if n < 1000:
        return _below_1000_tokens(n, variant)
    q, r = divmod(n, 1000)
    if q == 1:
        left = ["mil"]
    else:
        left = _below_1000_tokens(q, variant) + ["mil"]
    if r == 0:
        return left
    return left + ["e", *_below_1000_tokens(r, variant)]


def expand_cardinal_digits_to_portuguese_words(s: str, *, variant: str = "pt_br") -> str:
    """
    Replace a non-empty digit string with Portuguese cardinal **words** (space-separated).

    * Leading zeros → digit-by-digit (*zero zero sete*).
    * ``> 999_999`` → *s* unchanged.
    """
    v = variant.strip().lower().replace("-", "_")
    if v not in ("pt_br", "pt_pt"):
        raise ValueError("variant must be 'pt_br' or 'pt_pt'")
    if not s.isdigit():
        return s
    if len(s) > 1 and s[0] == "0":
        return " ".join(_DIGIT_WORD[int(c)] for c in s)
    n = int(s)
    if n > 999_999:
        return s
    if n == 0:
        return "zero"
    return " ".join(_below_1_000_000_tokens(n, v))


def expand_digit_tokens_in_text(text: str, *, variant: str = "pt_br") -> str:
    r"""
    Expand ``\b\d+-\d+\b`` as ``A - B`` (two cardinals), then each ``\b\d+\b`` with
    :func:`expand_cardinal_digits_to_portuguese_words`.
    """
    v = variant.strip().lower().replace("-", "_")
    if v not in ("pt_br", "pt_pt"):
        raise ValueError("variant must be 'pt_br' or 'pt_pt'")

    def _range(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return (
            f"{expand_cardinal_digits_to_portuguese_words(a, variant=v)} - "
            f"{expand_cardinal_digits_to_portuguese_words(b, variant=v)}"
        )

    text = re.sub(r"\b(\d+)-(\d+)\b", _range, text)
    return re.sub(
        r"\b\d+\b",
        lambda m: expand_cardinal_digits_to_portuguese_words(m.group(0), variant=v),
        text,
    )
