"""
Sino-Korean cardinal expansion for ASCII digit tokens in :mod:`korean_rule_g2p`.

Converts tokens like ``42``, ``1,234``, ``007``, ``3.14``, and ``-10`` into Hangul (plus
``마이너스`` for negatives) so the lexicon + rule pipeline can pronounce them.

* Leading-zero integers (e.g. ``007``) are read **digit-by-digit** with ``영`` for zero.
* ``0`` alone → ``영``.
* Integers use **Sino-Korean** grouping (``만`` / ``억`` / ``조`` / ``경``), mirroring the
  usual 10^4 block structure.
* Decimals use ``점`` and read fractional digits one-by-one (e.g. ``3.14`` → ``삼점일사``).
* Magnitudes ``>= 10^16`` fall back to digit-by-digit Hangul.
"""

from __future__ import annotations

import re
import unicodedata

# 0–9 as spoken in digit sequences and as Sino-Korean units (영/일/…/구).
_SINO = ("영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")

_MAX_CARDINAL = 10**16

_TOKEN_RE = re.compile(
    r"^"
    r"(?P<sign>[+-])?"
    r"(?P<whole>\d+)"
    r"(?:(?P<dsep>[.,])(?P<frac>\d+))?"
    r"$"
)

# Thousands separators only (do not erase ``,`` used as decimal comma in ``3,14``).
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=(?:\d{3})+\b)")


def _normalize_numeral_token_string(raw: str) -> str:
    t = unicodedata.normalize("NFC", raw.strip()).replace("_", "").replace(" ", "")
    t = _THOUSANDS_COMMA_RE.sub("", t)
    return t


def _hangul_digits_only(s: str) -> str:
    return "".join(_SINO[int(c)] for c in s if c.isdigit())


def _section_under_10000(n: int) -> str:
    """Write ``1 <= n <= 9999`` as Sino-Korean Hangul (no 만/억).

    Unlike Mandarin, empty tens/hundreds slots do not insert ``영`` (e.g. ``105`` → ``백오``).
    """
    if n <= 0 or n >= 10000:
        raise ValueError(n)
    q, r = divmod(n, 1000)
    parts: list[str] = []
    if q > 0:
        if q == 1:
            parts.append("천")
        else:
            parts.append(_SINO[q] + "천")
    b, r2 = divmod(r, 100)
    if b > 0:
        if b == 1:
            parts.append("백")
        else:
            parts.append(_SINO[b] + "백")
    s, t = divmod(r2, 10)
    if s == 0:
        if t:
            parts.append(_SINO[t])
    elif s == 1:
        if t == 0:
            parts.append("십")
        else:
            parts.append("십" + _SINO[t])
    else:
        parts.append(_SINO[s] + "십" + (_SINO[t] if t else ""))
    return "".join(parts)


def int_to_sino_korean_hangul(n: int) -> str:
    """Non-negative integer → Sino-Korean cardinal Hangul (``영`` … ``경`` …)."""
    if n < 0:
        raise ValueError(n)
    if n == 0:
        return "영"
    if n >= _MAX_CARDINAL:
        return _hangul_digits_only(str(n))
    low_first: list[int] = []
    x = n
    while x > 0:
        low_first.append(x % 10000)
        x //= 10000
    gs = list(reversed(low_first))
    units = ("", "만", "억", "조", "경")
    parts: list[str] = []
    zero_pending = False
    for i, g in enumerate(gs):
        if g == 0:
            if parts:
                zero_pending = True
            continue
        if zero_pending:
            parts.append("영")
            zero_pending = False
        ui = len(gs) - 1 - i
        u = units[ui] if ui < len(units) else units[-1]
        # 10_000 is ``만`` (not ``일만``); larger coefficients use ``이십만``, etc.
        if u == "만" and g == 1:
            parts.append("만")
        else:
            parts.append(_section_under_10000(g) + u)
    return "".join(parts)


def korean_reading_fragments_from_ascii_numeral_token(token: str) -> list[str] | None:
    """
    If *token* is a plain ASCII numeral (optional sign; ``.``/``,`` decimal; strip ``,`` / ``_``),
    return G2P fragments: Hangul strings and optionally the Latin lexeme ``마이너스``.

    Otherwise return ``None``.
    """
    raw = _normalize_numeral_token_string(token)
    if not raw:
        return None
    m = _TOKEN_RE.fullmatch(raw)
    if not m:
        return None

    sign = m.group("sign")
    whole = m.group("whole")
    frac = m.group("frac")
    neg = sign == "-"

    if frac is not None:
        if len(whole) > 1 and whole.startswith("0"):
            return None
        wv = int(whole, 10) if whole else 0
        body = int_to_sino_korean_hangul(wv) + "점" + _hangul_digits_only(frac)
        if neg:
            return ["마이너스", body]
        return [body]

    if len(whole) > 1 and whole.startswith("0"):
        seq = _hangul_digits_only(whole)
        if neg:
            return ["마이너스", seq]
        return [seq]

    wv = int(whole, 10)
    body = int_to_sino_korean_hangul(wv)
    if neg:
        return ["마이너스", body]
    return [body]


def is_ascii_numeral_token(token: str) -> bool:
    return korean_reading_fragments_from_ascii_numeral_token(token) is not None
