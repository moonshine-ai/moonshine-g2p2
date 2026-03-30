"""
Ukrainian cardinal expansion for digit sequences in G2P pipelines.

Produces space-separated Ukrainian **words** (orthography) for :mod:`ukrainian_rule_g2p`.

* Leading zeros → digit-by-digit (*нуль нуль сім*).
* ``0`` → *нуль*.
* Hyphenated spans ``\\b\\d+-\\d+\\b`` become two cardinals separated by `` - ``.
* Integers ``> 999_999`` are left unchanged.

Thousand forms follow standard agreement (*одна тисяча*, *дві тисячі*, *п'ять тисяч*, …).
"""

from __future__ import annotations

import re

_DIGIT_WORD = (
    "нуль",
    "один",
    "два",
    "три",
    "чотири",
    "п'ять",
    "шість",
    "сім",
    "вісім",
    "дев'ять",
)

_TEENS = (
    "десять",
    "одинадцять",
    "дванадцять",
    "тринадцять",
    "чотирнадцять",
    "п'ятнадцять",
    "шістнадцять",
    "сімнадцять",
    "вісімнадцять",
    "дев'ятнадцять",
)

_TENS = (
    "",
    "",
    "двадцять",
    "тридцять",
    "сорок",
    "п'ятдесят",
    "шістдесят",
    "сімдесят",
    "вісімдесят",
    "дев'яносто",
)

_HUNDREDS = (
    "",
    "сто",
    "двісті",
    "триста",
    "чотириста",
    "п'ятсот",
    "шістсот",
    "сімсот",
    "вісімсот",
    "дев'ятсот",
)


def _thousand_noun(h: int) -> str:
    if h % 100 in (11, 12, 13, 14):
        return "тисяч"
    m = h % 10
    if m == 1:
        return "тисяча"
    if m in (2, 3, 4):
        return "тисячі"
    return "тисяч"


def _under_100_thousand_mult(n: int) -> list[str]:
    """Expand 1..99 when it is the multiplier immediately before *тисяча* (feminine 1,2)."""
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n == 1:
        return ["одна"]
    if n == 2:
        return ["дві"]
    if n == 3:
        return ["три"]
    if n == 4:
        return ["чотири"]
    if n < 20:
        return [_TEENS[n - 10]]
    t, u = divmod(n, 10)
    tn = _TENS[t]
    if u == 0:
        return [tn]
    if u == 1:
        return [tn, "одна"]
    if u == 2:
        return [tn, "дві"]
    if u == 3:
        return [tn, "три"]
    if u == 4:
        return [tn, "чотири"]
    return [tn, _DIGIT_WORD[u]]


def _under_100_plain(n: int) -> list[str]:
    """Expand 0..99 for ordinary cardinals (e.g. year fragments, under-thousand)."""
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return [_DIGIT_WORD[n]]
    if n < 20:
        return [_TEENS[n - 10]]
    t, u = divmod(n, 10)
    tn = _TENS[t]
    if u == 0:
        return [tn]
    return [tn, _DIGIT_WORD[u]]


def _tokens_thousands_multiplier(h: int) -> list[str]:
    if h <= 0 or h > 999:
        raise ValueError(h)
    tn = _thousand_noun(h)
    if h < 100:
        return _under_100_thousand_mult(h) + [tn]
    hundreds, rem = divmod(h, 100)
    parts: list[str] = []
    parts.append(_HUNDREDS[hundreds])
    if rem == 0:
        return parts + [tn]
    return parts + _under_100_thousand_mult(rem) + [tn]


def _tokens_0_999(n: int) -> list[str]:
    if n < 0 or n > 999:
        raise ValueError(n)
    if n == 0:
        return ["нуль"]
    h, r = divmod(n, 100)
    parts: list[str] = []
    if h:
        parts.append(_HUNDREDS[h])
    if r == 0:
        return parts
    parts.extend(_under_100_plain(r))
    return parts


def _below_1_000_000_tokens(n: int) -> list[str]:
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    if n < 1000:
        return _tokens_0_999(n)
    q, r = divmod(n, 1000)
    left = _tokens_thousands_multiplier(q)
    if r == 0:
        return left
    return left + _tokens_0_999(r)


def expand_cardinal_digits_to_ukrainian_words(s: str) -> str:
    if not s.isdigit():
        return s
    if len(s) > 1 and s[0] == "0":
        return " ".join(_DIGIT_WORD[int(c)] for c in s)
    n = int(s)
    if n > 999_999:
        return s
    if n == 0:
        return "нуль"
    return " ".join(_below_1_000_000_tokens(n))


_DIGIT_TOKEN_RE = re.compile(r"\b\d+\b")
_DIGIT_RANGE_RE = re.compile(r"\b(\d+)-(\d+)\b")


def expand_digit_tokens_in_text(text: str) -> str:
    """Expand ``\\b\\d+-\\d+\\b`` first, then every ``\\b\\d+\\b``."""

    def repl_range(m: re.Match[str]) -> str:
        a = expand_cardinal_digits_to_ukrainian_words(m.group(1))
        b = expand_cardinal_digits_to_ukrainian_words(m.group(2))
        return f"{a} - {b}"

    out = _DIGIT_RANGE_RE.sub(repl_range, text)
    return _DIGIT_TOKEN_RE.sub(lambda m: expand_cardinal_digits_to_ukrainian_words(m.group(0)), out)
