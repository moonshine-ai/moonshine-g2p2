"""
Italian cardinal expansion for digit sequences (counts, years), for G2P pipelines.

Produces space-separated Italian **words** (orthography) so :mod:`italian_rule_g2p` can look up
each token in ``data/it/dict.tsv`` or fall back to rules. Compounds follow standard Italian
(*ventuno*, *ventitré*, *duecentoquarantasette*, *quattrocentododicimila*, …).

* Leading zeros (e.g. ``007``) are read digit-by-digit.
* ``0`` → *zero*.
* Hyphenated spans ``\b\d+-\d+\b`` become two cardinals separated by `` - `` (same spirit as
  Dutch ``tot``, easy to post-process).
* Integers ``> 999_999`` are left unchanged (returns *s*).
"""

from __future__ import annotations

import re

_DIGIT_WORD = (
    "zero",
    "uno",
    "due",
    "tre",
    "quattro",
    "cinque",
    "sei",
    "sette",
    "otto",
    "nove",
)

_SPECIAL_TEENS = {
    11: "undici",
    12: "dodici",
    13: "tredici",
    14: "quattordici",
    15: "quindici",
    16: "sedici",
    17: "diciassette",
    18: "diciotto",
    19: "diciannove",
}

_TENS = (
    "",
    "",
    "venti",
    "trenta",
    "quaranta",
    "cinquanta",
    "sessanta",
    "settanta",
    "ottanta",
    "novanta",
)


def _under_100(n: int) -> str:
    if n < 0 or n >= 100:
        raise ValueError(n)
    if n < 10:
        return _DIGIT_WORD[n]
    if n == 10:
        return "dieci"
    if n < 20:
        return _SPECIAL_TEENS[n]
    t, u = divmod(n, 10)
    tn = _TENS[t]
    if u == 0:
        return tn
    stem = tn[:-1]
    if u == 1:
        return stem + "uno"
    if u == 8:
        return stem + "otto"
    if u == 3:
        if tn.endswith("i"):
            return stem + "itré"
        return stem + "atré"
    uw = _DIGIT_WORD[u]
    if tn.endswith("i"):
        if u == 6:
            return stem + "isei"
        if u == 7:
            return stem + "isette"
        return stem + "i" + uw
    # trenta, quaranta, …
    if u == 6:
        return stem + "asei"
    if u == 7:
        return stem + "asette"
    return stem + "a" + uw


def _hundred_head(h: int) -> str:
    if h < 1 or h > 9:
        raise ValueError(h)
    if h == 1:
        return "cento"
    stems = ("due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove")
    return stems[h - 2] + "cento"


def _spell_1_999_fused(n: int) -> str:
    """One orthographic token for 1..999 (no spaces), e.g. *duecentoquarantasette*."""
    if n < 1 or n > 999:
        raise ValueError(n)
    if n < 100:
        return _under_100(n)
    h, r = divmod(n, 100)
    if h == 1:
        if r == 0:
            return "cento"
        return "cento" + _under_100(r)
    head = _hundred_head(h)
    if r == 0:
        return head
    return head + _under_100(r)


def _tokens_0_999(n: int) -> list[str]:
    """Lexicon-friendly tokens: *duecento* + *quarantasette* rather than one huge OOV."""
    if n < 0 or n > 999:
        raise ValueError(n)
    if n == 0:
        return ["zero"]
    if n < 100:
        return [_under_100(n)]
    h, r = divmod(n, 100)
    if r == 0:
        return [_hundred_head(h) if h > 1 else "cento"]
    if h == 1:
        return ["cento", _under_100(r)]
    return [_hundred_head(h), _under_100(r)]


def _thousands_multiplier_tokens(q: int) -> list[str]:
    """Words for *q* × 1000 (q >= 1)."""
    if q < 1 or q > 999:
        raise ValueError(q)
    if q == 1:
        return ["mille"]
    if q < 10:
        fused = (
            "duemila",
            "tremila",
            "quattromila",
            "cinquemila",
            "seimila",
            "settemila",
            "ottomila",
            "novemila",
        )
        return [fused[q - 2]]
    return [_spell_1_999_fused(q) + "mila"]


def _below_1_000_000_tokens(n: int) -> list[str]:
    if n < 0 or n >= 1_000_000:
        raise ValueError(n)
    if n < 1000:
        return _tokens_0_999(n)
    q, r = divmod(n, 1000)
    left = _thousands_multiplier_tokens(q)
    if r == 0:
        return left
    return left + _tokens_0_999(r)


def expand_cardinal_digits_to_italian_words(s: str) -> str:
    """
    Replace a non-empty digit string with Italian cardinal **words** (space-separated).

    * Leading zeros → digit-by-digit (*zero zero sette*).
    * ``> 999_999`` → *s* unchanged.
    """
    if not s.isdigit():
        return s
    if len(s) > 1 and s[0] == "0":
        return " ".join(_DIGIT_WORD[int(c)] for c in s)
    n = int(s)
    if n > 999_999:
        return s
    if n == 0:
        return "zero"
    return " ".join(_below_1_000_000_tokens(n))


def expand_digit_tokens_in_text(text: str) -> str:
    r"""
    Expand ``\b\d+-\d+\b`` as ``A - B`` (two cardinals), then each ``\b\d+\b`` with
    :func:`expand_cardinal_digits_to_italian_words`.
    """

    def _range(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        return (
            f"{expand_cardinal_digits_to_italian_words(a)} - "
            f"{expand_cardinal_digits_to_italian_words(b)}"
        )

    text = re.sub(r"\b(\d+)-(\d+)\b", _range, text)
    return re.sub(
        r"\b\d+\b",
        lambda m: expand_cardinal_digits_to_italian_words(m.group(0)),
        text,
    )
