"""
Hindi cardinal expansion for digit sequences in G2P pipelines.

Produces space-separated Hindi **Devanagari** words for :mod:`hindi_rule_g2p`.

* Leading zeros → digit-by-digit (*शून्य शून्य सात*).
* ``0`` → *शून्य*.
* Hyphenated spans ``\\b\\d+-\\d+\\b`` become two cardinals separated by `` - ``.
* Integers ``> 999_999`` are left unchanged.
"""

from __future__ import annotations

import re

_DIGIT_WORD = (
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पाँच",
    "छह",
    "सात",
    "आठ",
    "नौ",
)

_TEENS = (
    "दस",
    "ग्यारह",
    "बारह",
    "तेरह",
    "चौदह",
    "पंद्रह",
    "सोलह",
    "सत्रह",
    "अठारह",
    "उन्नीस",
)

_TENS = (
    "",
    "",
    "बीस",
    "तीस",
    "चालीस",
    "पचास",
    "साठ",
    "सत्तर",
    "अस्सी",
    "नब्बे",
)

# 21–29, 31–39, … 91–99 (spoken Hindi; avoids awkward *बीस एक* for G2P)
_UNITS_AFTER_TEN = (
    ("इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस"),
    ("इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस"),
    ("इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास"),
    ("इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ"),
    ("इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर"),
    ("इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी"),
    ("इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सतासी", "अठासी", "नवासी"),
    ("इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे"),
)


def _under_100(n: int) -> list[str]:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return [_DIGIT_WORD[n]]
    if n < 20:
        return [_TEENS[n - 10]]
    if n % 10 == 0:
        return [_TENS[n // 10]]
    tens = n // 10
    unit = n % 10
    return [_UNITS_AFTER_TEN[tens - 2][unit - 1]]


def _tokens_0_999(n: int) -> list[str]:
    if n < 0 or n > 999:
        raise ValueError(n)
    if n == 0:
        return ["शून्य"]
    h, r = divmod(n, 100)
    parts: list[str] = []
    if h > 0:
        if h == 1:
            parts.append("सौ")
        else:
            parts.extend([_DIGIT_WORD[h], "सौ"])
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
        left = ["हज़ार"]
    else:
        left = _tokens_0_999(q) + ["हज़ार"]
    if r == 0:
        return left
    if r < 100:
        return left + _under_100(r)
    return left + _tokens_0_999(r)


def expand_cardinal_digits_to_hindi_words(s: str) -> str:
    if not s.isdigit():
        return s
    if len(s) > 1 and s[0] == "0":
        return " ".join(_DIGIT_WORD[int(c)] for c in s)
    n = int(s)
    if n > 999_999:
        return s
    if n == 0:
        return "शून्य"
    return " ".join(_below_1_000_000_tokens(n))


def expand_digit_tokens_in_text(text: str) -> str:
    r"""
    Expand ``\b\d+-\d+\b`` as ``A - B`` (two cardinals), then each ``\b\d+\b`` with
    :func:`expand_cardinal_digits_to_hindi_words`.
    """

    def _range(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return f"{expand_cardinal_digits_to_hindi_words(a)} - {expand_cardinal_digits_to_hindi_words(b)}"

    text = re.sub(r"\b(\d+)-(\d+)\b", _range, text)
    return re.sub(
        r"\b\d+\b",
        lambda m: expand_cardinal_digits_to_hindi_words(m.group(0)),
        text,
    )


# Devanagari digits U+0966–U+096F → same expansion as ASCII via int
_DEV_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")


def expand_devanagari_digit_runs_in_text(text: str) -> str:
    """Replace runs of Devanagari digits with Hindi cardinal words (spacing preserved)."""

    def _sub(m: re.Match[str]) -> str:
        ascii_digits = m.group(0).translate(_DEV_DIGIT_MAP)
        return expand_cardinal_digits_to_hindi_words(ascii_digits)

    return re.sub(r"[\u0966-\u096F]+", _sub, text)
