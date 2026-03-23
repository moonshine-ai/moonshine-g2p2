"""IPA tokenization and Levenshtein-based matching for heteronym decoder outputs."""

from __future__ import annotations

import unicodedata


def ipa_string_to_phoneme_tokens(s: str) -> list[str]:
    """
    Map an IPA string to a token sequence for edit distance.

    Space-separated IPA (if present) is split on whitespace; otherwise each
    Unicode code point is one token (after NFC normalization).
    """
    t = unicodedata.normalize("NFC", (s or "").strip())
    if not t:
        return []
    if " " in t:
        return [p for p in t.split() if p]
    return list(t)


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein edit distance between two token sequences."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Single-row DP
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i]
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[lb]


def pick_closest_alternative_index(
    predicted_phoneme_tokens: list[str],
    ipa_alternatives: list[str],
    *,
    n_valid: int,
    extra_phonemes: int,
) -> int:
    """
    Choose the alternative index in ``0..n_valid-1`` minimizing Levenshtein distance
    between that alternative's tokens and ``predicted_phoneme_tokens[:L+extra_phonemes]``
    where *L* is the alternative's token length (limits runaway decoder repetition).
    """
    n = min(n_valid, len(ipa_alternatives))
    if n <= 0:
        return 0
    best_i, best_d = 0, 10**9
    for i in range(n):
        cand = ipa_string_to_phoneme_tokens(ipa_alternatives[i])
        lim = len(cand) + max(0, int(extra_phonemes))
        prefix = predicted_phoneme_tokens[:lim]
        d = levenshtein_distance(cand, prefix)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def pick_closest_cmudict_ipa(
    predicted_phoneme_tokens: list[str],
    cmudict_alternatives: list[str],
    *,
    extra_phonemes: int,
) -> str:
    """Return one string from *cmudict_alternatives* using the same rule as training eval."""
    if not cmudict_alternatives:
        return ""
    if len(cmudict_alternatives) == 1:
        return cmudict_alternatives[0]
    i = pick_closest_alternative_index(
        predicted_phoneme_tokens,
        cmudict_alternatives,
        n_valid=len(cmudict_alternatives),
        extra_phonemes=extra_phonemes,
    )
    return cmudict_alternatives[i]
