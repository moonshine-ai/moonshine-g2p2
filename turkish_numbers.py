"""
Turkish cardinal expansion for digit sequences in G2P pipelines.

Produces space-separated Turkish **words** (orthography) for :mod:`turkish_rule_g2p`.

* Leading zeros → digit-by-digit (*sıfır sıfır yedi*).
* ``0`` → *sıfır*.
* Hyphenated spans ``\\b\\d+-\\d+\\b`` become two cardinals separated by `` - `` (Italian/Dutch style).
* Integers ``> 999_999`` are left unchanged.
"""

from __future__ import annotations

import re

_DIGIT_WORD = (
    "sıfır",
    "bir",
    "iki",
    "üç",
    "dört",
    "beş",
    "altı",
    "yedi",
    "sekiz",
    "dokuz",
)

_TENS = (
    "",
    "",
    "yirmi",
    "otuz",
    "kırk",
    "elli",
    "altmış",
    "yetmiş",
    "seksen",
    "doksan",
)


def _under_100(n: int) -> list[str]:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return [_DIGIT_WORD[n]]
    if n == 10:
        return ["on"]
    if n < 20:
        return ["on", _DIGIT_WORD[n - 10]]
    t, u = divmod(n, 10)
    tn = _TENS[t]
    if u == 0:
        return [tn]
    return [tn, _DIGIT_WORD[u]]


def _tokens_0_999(n: int) -> list[str]:
    if n < 0 or n > 999:
        raise ValueError(n)
    if n == 0:
        return ["sıfır"]
    h, r = divmod(n, 100)
    parts: list[str] = []
    if h > 0:
        if h == 1:
            parts.append("yüz")
        else:
            parts.extend([_DIGIT_WORD[h], "yüz"])
        if r == 0:
            return parts
    return parts + _under_100(r)


def _below_1_000_000_tokens(n: int) -> list[str]:
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    if n < 1000:
        return _tokens_0_999(n)
    q, r = divmod(n, 1000)
    if q == 1:
        left = ["bin"]
    else:
        left = _tokens_0_999(q) + ["bin"]
    if r == 0:
        return left
    return left + _tokens_0_999(r)


def expand_cardinal_digits_to_turkish_words(s: str) -> str:
    if not s.isdigit():
        return s
    if len(s) > 1 and s[0] == "0":
        return " ".join(_DIGIT_WORD[int(c)] for c in s)
    n = int(s)
    if n > 999_999:
        return s
    if n == 0:
        return "sıfır"
    return " ".join(_below_1_000_000_tokens(n))


def expand_digit_tokens_in_text(text: str) -> str:
    r"""
    Expand ``\b\d+-\d+\b`` as ``A - B`` (two cardinals), then each ``\b\d+\b`` with
    :func:`expand_cardinal_digits_to_turkish_words`.
    """

    def _range(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return f"{expand_cardinal_digits_to_turkish_words(a)} - {expand_cardinal_digits_to_turkish_words(b)}"

    text = re.sub(r"\b(\d+)-(\d+)\b", _range, text)
    return re.sub(
        r"\b\d+\b",
        lambda m: expand_cardinal_digits_to_turkish_words(m.group(0)),
        text,
    )
