"""
Dutch cardinal expansion for digit sequences (e.g. years, counts), for G2P pipelines.

Produces space-separated Dutch **words** (orthography) so :mod:`dutch_rule_g2p` can look up
each token in ``data/nl/dict.tsv`` or fall back to rules.

* 1100–1999 use *elfhonderd* … *negentienhonderd* + remainder (common year / “nineteen-X” style).
* 1000–1099 use *duizend* + remainder.
* 2000–9999 use *tweeduizend* … + remainder (*driehonderd vijfenveertig* style).
* Up to 999_999, same spirit as :mod:`french_numbers` (larger values left unchanged).

Leading zeros are read digit by digit (*nul nul zeven*).
"""

from __future__ import annotations

import re

_DIGIT_WORD = (
    "nul",
    "een",
    "twee",
    "drie",
    "vier",
    "vijf",
    "zes",
    "zeven",
    "acht",
    "negen",
)

# 11–19 for compounds like elfhonderd, negentienhonderd.
_TEEN = {
    11: "elf",
    12: "twaalf",
    13: "dertien",
    14: "veertien",
    15: "vijftien",
    16: "zestien",
    17: "zeventien",
    18: "achttien",
    19: "negentien",
}

_TENS = (
    "",
    "",
    "twintig",
    "dertig",
    "veertig",
    "vijftig",
    "zestig",
    "zeventig",
    "tachtig",
    "negentig",
)


def _join_unit_tens(u: int, tens_word: str) -> str:
    """
    *eenentwintig*, *tweeentwintig*, … — ASCII *en* joiner to match ipa-dict headwords (no umlaut).
    """
    if u == 0:
        return tens_word
    stem = (
        "een",
        "twee",
        "drie",
        "vier",
        "vijf",
        "zes",
        "zeven",
        "acht",
        "negen",
    )[u - 1]
    return f"{stem}en{tens_word}"


def _below_100(n: int) -> str:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return _DIGIT_WORD[n]
    if n < 13:
        return ("tien", "elf", "twaalf")[n - 10]
    if n < 20:
        return (
            "dertien",
            "veertien",
            "vijftien",
            "zestien",
            "zeventien",
            "achttien",
            "negentien",
        )[n - 13]
    t, u = divmod(n, 10)
    tens_word = _TENS[t]
    return _join_unit_tens(u, tens_word)


def _below_1000_spaced(n: int) -> str:
    """1..999 as words; hundred and remainder space-separated for lexicon friendliness."""
    if n < 0 or n >= 1000:
        raise ValueError(n)
    if n < 100:
        return _below_100(n)
    h, r = divmod(n, 100)
    head = "honderd" if h == 1 else f"{_DIGIT_WORD[h]}honderd"
    if r == 0:
        return head
    return f"{head} {_below_100(r)}"


def _from_1000_to_9999(n: int) -> str:
    if n < 1000 or n > 9999:
        raise ValueError(n)
    if n < 1100:
        if n == 1000:
            return "duizend"
        return f"duizend {_below_100(n - 1000)}"
    if n < 2000:
        c, r = divmod(n, 100)
        if c not in _TEEN:
            raise ValueError(n)
        head = f"{_TEEN[c]}honderd"
        if r == 0:
            return head
        return f"{head} {_below_100(r)}"
    q, r = divmod(n, 1000)
    if q == 1:
        left = "duizend"
    else:
        left = f"{_below_100(q)}duizend"
    if r == 0:
        return left
    return f"{left} {_below_1000_spaced(r)}"


def _fix_thousands_compound(q: int) -> str:
    """
    *2..9*duizend* for single-digit thousands; *tweeduizend* … *negenentwintigduizend* for 10–99;
    for 100–999 use spaced *honderd een duizend* so multi-token lookup works.
    """
    if q == 1:
        return "duizend"
    if q < 10:
        return f"{_DIGIT_WORD[q]}duizend"
    if q < 100:
        return f"{_below_100(q)}duizend"
    return f"{_below_1000_spaced(q)} duizend"


def _below_1_000_000_v2(n: int) -> str:
    """Clearer decomposition for 10_000..999_999."""
    if n < 10_000:
        if n < 1000:
            return _below_1000_spaced(n)
        return _from_1000_to_9999(n)
    q, r = divmod(n, 1000)
    left = "duizend" if q == 1 else _fix_thousands_compound(q)
    if r == 0:
        return left
    return f"{left} {_below_1000_spaced(r)}"


def expand_cardinal_digits_to_dutch_words(s: str) -> str:
    """
    Replace a non-empty digit string with a Dutch cardinal **word** phrase (space-separated tokens).

    * Leading zeros (e.g. ``007``) are read digit by digit.
    * ``0`` → *nul*.
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
        return "nul"
    return _below_1_000_000_v2(n)


def expand_digit_tokens_in_text(text: str) -> str:
    r"""
    Expand ``\b\d+-\d+\b`` as *A tot B* (spoken year/number ranges), then each ``\b\d+\b`` span
    with :func:`expand_cardinal_digits_to_dutch_words`.
    """

    def _range(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return (
            f"{expand_cardinal_digits_to_dutch_words(a)} tot "
            f"{expand_cardinal_digits_to_dutch_words(b)}"
        )

    text = re.sub(r"\b(\d+)-(\d+)\b", _range, text)
    return re.sub(
        r"\b\d+\b",
        lambda m: expand_cardinal_digits_to_dutch_words(m.group(0)),
        text,
    )
