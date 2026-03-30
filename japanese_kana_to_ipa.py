#!/usr/bin/env python3
"""
Hiragana / katakana → broad IPA aligned with ``data/ja/dict.tsv`` (NHK-style ɯ, ɾ, ɕ, …).

Used by :mod:`japanese_onnx_g2p` and the MeCab reference path so comparisons are apples-to-apples.
"""

from __future__ import annotations

import unicodedata
from typing import List, Tuple

# (onset, nucleus) where full IPA is onset + nucleus; nucleus may be empty for ん.
# Onset "" means vowel-only mora.
_MORA: dict[str, Tuple[str, str]] = {}


def _add_row(
    consonant: str,
    hiras: str,
    nuclei: List[str],
    *,
    special_onset: str | None = None,
) -> None:
    onset = special_onset if special_onset is not None else consonant
    for h, n in zip(hiras, nuclei):
        _MORA[h] = (onset, n)


def _build_mora_table() -> None:
    if _MORA:
        return
    aiueo = "あいうえお"
    vip = ["a", "i", "ɯ", "e", "o"]
    for h, v in zip(aiueo, vip):
        _MORA[h] = ("", v)

    # k/g
    _add_row("k", "かきくけこ", ["a", "i", "ɯ", "e", "o"])
    _add_row("g", "がぎぐげご", ["a", "i", "ɯ", "e", "o"])
    # s/z
    _add_row("s", "さすせそ", ["a", "ɯ", "e", "o"])
    _MORA["し"] = ("ɕ", "i")
    _MORA["ざ"] = ("z", "a")
    _MORA["ず"] = ("z", "ɯ")
    _MORA["ぜ"] = ("z", "e")
    _MORA["ぞ"] = ("z", "o")
    _MORA["じ"] = ("dʑ", "i")
    # t/d
    _add_row("t", "たてと", ["a", "e", "o"])
    _MORA["ち"] = ("tɕ", "i")
    _MORA["つ"] = ("ts", "ɯ")
    _add_row("d", "だでど", ["a", "e", "o"])
    _MORA["ぢ"] = ("dʑ", "i")  # rare
    _MORA["づ"] = ("dz", "ɯ")  # rare
    # n
    _add_row("n", "なにぬねの", ["a", "i", "ɯ", "e", "o"])
    # h/b/p/m
    _add_row("h", "はへほ", ["a", "e", "o"])
    _MORA["ひ"] = ("ç", "i")
    _MORA["ふ"] = ("ɸ", "ɯ")
    _add_row("b", "ばびぶべぼ", ["a", "i", "ɯ", "e", "o"])
    _add_row("p", "ぱぴぷぺぽ", ["a", "i", "ɯ", "e", "o"])
    _add_row("m", "まみむめも", ["a", "i", "ɯ", "e", "o"])
    # y
    _MORA["や"] = ("j", "a")
    _MORA["ゆ"] = ("j", "ɯ")
    _MORA["よ"] = ("j", "o")
    # r
    _add_row("ɾ", "らりるれろ", ["a", "i", "ɯ", "e", "o"], special_onset="ɾ")
    # w (modern)
    _MORA["わ"] = ("ɰ", "a")
    _MORA["を"] = ("", "o")
    _MORA["ん"] = ("", "ɴ")

    # yōon
    yoons = [
        ("きゃきゅきょ", "k", ["ja", "jɯ", "jo"]),
        ("ぎゃぎゅぎょ", "g", ["ja", "jɯ", "jo"]),
        ("しゃしゅしょ", "ɕ", ["a", "ɯ", "o"]),
        ("じゃじゅじょ", "dʑ", ["a", "ɯ", "o"]),
        ("ちゃちゅちょ", "tɕ", ["a", "ɯ", "o"]),
        ("にゃにゅにょ", "n", ["ja", "jɯ", "jo"]),
        ("ひゃひゅひょ", "ç", ["a", "ɯ", "o"]),
        ("びゃびゅびょ", "b", ["ja", "jɯ", "jo"]),
        ("ぴゃぴゅぴょ", "p", ["ja", "jɯ", "jo"]),
        ("みゃみゅみょ", "m", ["ja", "jɯ", "jo"]),
        ("りゃりゅりょ", "ɾ", ["ja", "jɯ", "jo"]),
    ]
    for grp, onset, nucs in yoons:
        for i in range(0, len(grp), 2):
            h = grp[i : i + 2]
            _MORA[h] = (onset, nucs[i // 2])

    # small vowels (foreign)
    _MORA["ぁ"] = ("", "a")
    _MORA["ぃ"] = ("", "i")
    _MORA["ぅ"] = ("", "ɯ")
    _MORA["ぇ"] = ("", "e")
    _MORA["ぉ"] = ("", "o")
    _MORA["ゎ"] = ("ɰ", "a")

    # ふぁ行
    _MORA["ふぁ"] = ("ɸ", "a")
    _MORA["ふぃ"] = ("ɸ", "i")
    _MORA["ふぇ"] = ("ɸ", "e")
    _MORA["ふぉ"] = ("ɸ", "o")
    _MORA["ふゃ"] = ("ɸ", "ja")
    _MORA["ふゅ"] = ("ɸ", "jɯ")
    _MORA["ふょ"] = ("ɸ", "jo")

    # ヴ行 (loan)
    _MORA["ヴぁ"] = ("v", "a")
    _MORA["ヴぃ"] = ("v", "i")
    _MORA["ヴ"] = ("v", "ɯ")
    _MORA["ヴぇ"] = ("v", "e")
    _MORA["ヴぉ"] = ("v", "o")
    _MORA["ヴゃ"] = ("v", "ja")
    _MORA["ヴゅ"] = ("v", "jɯ")
    _MORA["ヴょ"] = ("v", "jo")

    # てぃ/でぃ 等
    _MORA["てぃ"] = ("t", "i")
    _MORA["てゅ"] = ("t", "jɯ")
    _MORA["でぃ"] = ("d", "i")
    _MORA["でゅ"] = ("d", "jɯ")
    _MORA["とぅ"] = ("t", "ɯ")
    _MORA["どぅ"] = ("d", "ɯ")
    _MORA["つぁ"] = ("ts", "a")
    _MORA["つぃ"] = ("ts", "i")
    _MORA["つぇ"] = ("ts", "e")
    _MORA["つぉ"] = ("ts", "o")

    # うぃうぇうぉ (loan)
    _MORA["うぃ"] = ("ɰ", "i")
    _MORA["うぇ"] = ("ɰ", "e")
    _MORA["うぉ"] = ("ɰ", "o")

    # ヲ / を already o
    _MORA["ゐ"] = ("j", "i")  # archaic
    _MORA["ゑ"] = ("j", "e")  # archaic


_build_mora_table()

_HIRA_START = ord("ぁ")
_HIRA_END = ord("ゟ")
_KATA_START = ord("ァ")
_KATA_END = ord("ヿ")


def _katakana_to_hiragana(s: str) -> str:
    out: List[str] = []
    for ch in s:
        o = ord(ch)
        if _KATA_START <= o <= _KATA_END:
            # ー and prolonged are not shifted
            if ch == "ー":
                out.append(ch)
                continue
            out.append(chr(o - (_KATA_START - _HIRA_START)))
        else:
            out.append(ch)
    return "".join(out)


def _mora_keys_longest_first() -> List[str]:
    return sorted(_MORA.keys(), key=len, reverse=True)


_MORA_KEYS = _mora_keys_longest_first()


def _next_mora(s: str, i: int) -> Tuple[str, int] | None:
    if i >= len(s):
        return None
    ch = s[i]
    if ch in ("\u3099", "\u309A"):  # combining dakuten/handakuten — skip (precomposed only)
        return None
    for k in _MORA_KEYS:
        if s.startswith(k, i):
            return k, i + len(k)
    return None


def _geminate(onset: str, nucleus: str) -> str:
    if not onset:
        return onset + nucleus
    return onset + "ː" + nucleus


def _apply_sokuon(prev_out: str, onset: str, nucleus: str) -> str:
    """っ before (onset, nucleus) → lengthen onset closure."""
    return prev_out + _geminate(onset, nucleus)


def _long_mark_extend(ipa: str) -> str:
    """Append length to the last vowel in *ipa* (ー handling)."""
    if not ipa:
        return "ː"
    vowels = "aeiouɯ"
    # walk back to last vowel char (skip tie bar)
    j = len(ipa) - 1
    while j >= 0 and ipa[j] not in vowels and ipa[j] != "ː":
        # skip modifiers that attach to vowels
        if ipa[j] in "̥̃":
            j -= 1
            continue
        break
    if j >= 0 and ipa[j] in vowels:
        return ipa[: j + 1] + "ː" + ipa[j + 1 :]
    return ipa + "ː"


def katakana_hiragana_to_ipa(text: str) -> str:
    """
    Convert a **kana-only** string (after NFKC) to concatenated IPA (no spaces).
    Unknown characters are skipped.
    """
    s = unicodedata.normalize("NFKC", text.strip())
    s = _katakana_to_hiragana(s)
    i = 0
    out: List[str] = []
    while i < len(s):
        ch = s[i]
        if ch == "ー":
            if out:
                out[-1] = _long_mark_extend(out[-1])
            i += 1
            continue
        if ch in "っッ":
            nxt = _next_mora(s, i + 1)
            if not nxt:
                i += 1
                continue
            mora, j = nxt
            onset, nuc = _MORA[mora]
            if onset:
                out.append(_geminate(onset, nuc))
            else:
                out.append(nuc)
            i = j
            continue
        got = _next_mora(s, i)
        if not got:
            i += 1
            continue
        mora, j = got
        onset, nuc = _MORA[mora]
        out.append(onset + nuc)
        i = j
    return "".join(out)


def reading_katakana_to_ipa(katakana: str) -> str:
    """UniDic-style katakana *pron* field → IPA."""
    return katakana_hiragana_to_ipa(katakana)


def is_kana_only(s: str) -> bool:
    t = unicodedata.normalize("NFKC", s.strip())
    for ch in t:
        o = ord(ch)
        if ch.isspace():
            continue
        if ch == "ー":
            continue
        if _HIRA_START <= o <= _HIRA_END:
            continue
        if _KATA_START <= o <= _KATA_END:
            continue
        if ch in "っッ":
            continue
        return False
    return bool(t)


def has_japanese_script(s: str) -> bool:
    t = unicodedata.normalize("NFKC", s)
    for ch in t:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            return True
        if _HIRA_START <= o <= _HIRA_END or _KATA_START <= o <= _KATA_END:
            return True
    return False
