"""
Russian cardinal expansion for ASCII digit sequences in Cyrillic text (Wikipedia, captions).

Produces space-separated Russian **words** so :mod:`russian_rule_g2p` can G2P them. Uses
grammatical forms for **тысяча** (*одна тысяча*, *две тысячи*, *пять тысяч*, …).

* Leading zeros → digit-by-digit (*ноль ноль семь*).
* ``0`` → *ноль*.
* Hyphenated spans ``\\b\\d+-\\d+\\b`` become two cardinals separated by `` - ``.
* Integers ``> 999_999`` are left unchanged.
"""

from __future__ import annotations

import re

_DIGIT = (
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
)

_TEENS = (
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
)

_TENS = (
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
)

_HUNDREDS = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)


def _ones_digit(n: int, *, feminine: bool) -> str:
    if n < 1 or n > 9:
        raise ValueError(n)
    if feminine:
        if n == 1:
            return "одна"
        if n == 2:
            return "две"
    return _DIGIT[n]


def _under_100_tokens(n: int, *, feminine: bool) -> list[str]:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return [_ones_digit(n, feminine=feminine)]
    if n < 20:
        return [_TEENS[n - 10]]
    t, u = divmod(n, 10)
    tens_w = _TENS[t]
    if u == 0:
        return [tens_w]
    return [tens_w, _ones_digit(u, feminine=feminine)]


def _cardinal_1_to_999(n: int, *, feminine: bool) -> list[str]:
    if n < 1 or n > 999:
        raise ValueError(n)
    if n < 100:
        return _under_100_tokens(n, feminine=feminine)
    h, r = divmod(n, 100)
    head = _HUNDREDS[h]
    if r == 0:
        return [head]
    return [head, *_under_100_tokens(r, feminine=feminine)]


def _thousand_suffix(q: int) -> str:
    if 11 <= (q % 100) <= 14:
        return "тысяч"
    k = q % 10
    if k == 1:
        return "тысяча"
    if 2 <= k <= 4:
        return "тысячи"
    return "тысяч"


def _below_1_000_000_tokens(n: int) -> list[str]:
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    if n < 1000:
        return _cardinal_1_to_999(n, feminine=False)
    q, r = divmod(n, 1000)
    left = _cardinal_1_to_999(q, feminine=True) + [_thousand_suffix(q)]
    if r == 0:
        return left
    return left + _cardinal_1_to_999(r, feminine=False)


def expand_cardinal_digits_to_russian_words(s: str) -> str:
    if not s.isdigit():
        return s
    if len(s) > 1 and s[0] == "0":
        return " ".join(_DIGIT[int(c)] for c in s)
    n = int(s)
    if n > 999_999:
        return s
    if n == 0:
        return "ноль"
    return " ".join(_below_1_000_000_tokens(n))


def expand_digit_tokens_in_text(text: str) -> str:
    r"""
    Expand ``\b\d+-\d+\b`` as ``A - B`` (two cardinals), then each ``\b\d+\b`` with
    :func:`expand_cardinal_digits_to_russian_words`.
    """

    def _range(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return (
            f"{expand_cardinal_digits_to_russian_words(a)} - "
            f"{expand_cardinal_digits_to_russian_words(b)}"
        )

    text = re.sub(r"\b(\d+)-(\d+)\b", _range, text)
    return re.sub(
        r"\b\d+\b",
        lambda m: expand_cardinal_digits_to_russian_words(m.group(0)),
        text,
    )
