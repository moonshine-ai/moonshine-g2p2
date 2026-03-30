"""
Merge **single-codepoint Han** LUWs from the char-level tokenizer into runs (e.g. 東+京 → 東京).

The KoichiYasuoka char-LUW model is fed ``tokenize_chinese_chars``-spaced input, so each kanji is
often its own LUW. Lexicon lookup and greedy decomposition need merged surfaces.
"""

from __future__ import annotations

from typing import List, Tuple


def _is_han_cp(cp: int) -> bool:
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0xF900 <= cp <= 0xFAFF)
    )


def _is_single_han(s: str) -> bool:
    if len(s) != 1:
        return False
    return _is_han_cp(ord(s[0]))


def _only_hiragana(s: str) -> bool:
    if not s:
        return False
    for ch in s:
        o = ord(ch)
        if not (0x3040 <= o <= 0x309F):
            return False
    return True


def _only_katakana(s: str) -> bool:
    if not s:
        return False
    for ch in s:
        o = ord(ch)
        if ch == "ー":
            continue
        if not (0x30A0 <= o <= 0x30FF):
            return False
    return True


def _only_han(s: str) -> bool:
    if not s:
        return False
    for ch in s:
        if not _is_han_cp(ord(ch)):
            return False
    return True


def merge_katakana_plus_han(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    e.g. ラテン + 語 → ラテン語 (both NOUN), for lexicon lookup.
    """
    if not pairs:
        return []
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(pairs):
        surf, tag = pairs[i]
        if (
            _only_katakana(surf)
            and tag in ("NOUN", "PROPN")
            and i + 1 < len(pairs)
            and _is_single_han(pairs[i + 1][0])
            and pairs[i + 1][1] in ("NOUN", "PROPN")
        ):
            out.append((surf + pairs[i + 1][0], tag))
            i += 2
            continue
        out.append((surf, tag))
        i += 1
    return out


def merge_verb_adj_okurigana(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Attach hiragana tails to a **Han-only** stem tagged VERB or ADJ (e.g. 行 + きます → 行きます).
    """
    if not pairs:
        return []
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(pairs):
        surf, tag = pairs[i]
        if _only_han(surf) and tag in ("VERB", "ADJ") and i + 1 < len(pairs):
            j = i + 1
            acc = surf
            while j < len(pairs):
                s2, _ = pairs[j]
                if _only_hiragana(s2):
                    acc += s2
                    j += 1
                else:
                    break
            out.append((acc, tag))
            i = j
            continue
        out.append((surf, tag))
        i += 1
    return out


def merge_single_han_luws(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Concatenate adjacent 1-char Han LUWs; keep the first tag."""
    if not pairs:
        return []
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(pairs):
        surf, tag = pairs[i]
        if _is_single_han(surf):
            j = i + 1
            acc = surf
            first_tag = tag
            while j < len(pairs):
                s2, _ = pairs[j]
                if _is_single_han(s2):
                    acc += s2
                    j += 1
                else:
                    break
            out.append((acc, first_tag))
            i = j
            continue
        out.append((surf, tag))
        i += 1
    return out


def merge_for_lexicon_lookup(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Han runs, katakana+kanji compounds, then verb/adjective okurigana."""
    a = merge_single_han_luws(pairs)
    b = merge_katakana_plus_han(a)
    return merge_verb_adj_okurigana(b)
