#!/usr/bin/env python3
"""
Korean grapheme-to-phoneme (G2P) — broad IPA for Standard Korean (Seoul), rule-based.

**Pure Python:** no MeCab or other native morphological analyzers. Hangul syllables are
read in NFC codepoint order (one syllable block per U+AC00..U+D7A3). Optional true
morpheme boundaries would need a separate lexicon + segmenter; they are not required for
the current IPA rules (tensification is syllable-based).

Pipeline (high level):
1. **Syllable scan** — collect ``(choseong, jungseong, jongseong)`` per Hangul character.
2. **Ordered phonological passes** on syllable tuples (simplified but ordered):
   - **Resyllabification (연음)** — jongseong before a following ``ㅇ`` onset shifts to the
     next syllable, including compound 받침 splits (Unicode jongseong table).
   - **Lateralization (유음화)** — coda ``ㄴ`` before onset ``ㄹ`` → coda ``ㄹ`` (e.g. 신라).
   - **IPA rendering** with coda neutralization when a coda precedes a consonantal onset
     or phrase boundary.
   - **Nasalization (비음화)** — coda obstruent before a nasal onset → place-matched nasal
     (e.g. 국물 → /kuŋ.mul/).
   - **Aspiration from coda ㅎ (격음화 일부)** — coda ``ㅎ`` lost, following plain stop becomes
     aspirated (e.g. 좋다).
   - **Tensification (경음화, 휴리스틱)** — after a **released** obstruent coda, a following
     plain ``ㄱ/ㄷ/ㅂ/ㅅ/ㅈ`` onset becomes tense at syllable boundaries (e.g. 학교).

Hangul is almost phonemic at the jamo level; most error budget is in post-lexical rules and
lexicalized tensification. Loanwords written in Hangul generally read by the same rules.

Dependencies::
    None (stdlib only).

Limitations::
    Palatalization of verb stems, lexicalized exceptions to tensification, precise coda
    allophones, and English digits/letters are only lightly handled. Output is **broad**
    IPA suitable for lexicon/TTS prototyping, not narrow transcription.

CLI: ``--verbose`` prints a rough Hangul/Latin run split for debugging (not morph analysis).
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Hangul decomposition (Unicode syllable formula)
# ---------------------------------------------------------------------------

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3

# Choseong (19): ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ
CHO_NAMES = tuple("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")

# Jungseong (21)
JUNG_NAMES = tuple("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

# Jongseong (27 + empty). Index 0 = no coda.
JONG_NAMES = (
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)

IDX_O = CHO_NAMES.index("ㅇ")

# When jongseong is non-zero and the next syllable has choseong ㅇ, split 받침 for 연음.
# Maps jong_index -> (remaining_jong_index, moved_choseong_index).
# ㅇ coda (21) does not link to a following vowel as a moved onset (stays /ŋ/).
_JONG_SPLIT_FOR_LINKING: dict[int, tuple[int, int]] = {
    1: (0, CHO_NAMES.index("ㄱ")),
    2: (0, CHO_NAMES.index("ㄲ")),
    3: (1, CHO_NAMES.index("ㅅ")),  # ㄱ remains
    4: (0, CHO_NAMES.index("ㄴ")),
    5: (4, CHO_NAMES.index("ㅈ")),
    6: (4, CHO_NAMES.index("ㅎ")),
    7: (0, CHO_NAMES.index("ㄷ")),
    8: (0, CHO_NAMES.index("ㄹ")),
    9: (8, CHO_NAMES.index("ㄱ")),  # 닭이 → 달기
    10: (8, CHO_NAMES.index("ㅁ")),
    11: (8, CHO_NAMES.index("ㅂ")),  # ㄼ is partly lexicalized; this covers common cases
    12: (8, CHO_NAMES.index("ㅅ")),
    13: (8, CHO_NAMES.index("ㅌ")),
    14: (8, CHO_NAMES.index("ㅍ")),
    15: (8, CHO_NAMES.index("ㅎ")),
    16: (0, CHO_NAMES.index("ㅁ")),
    17: (0, CHO_NAMES.index("ㅂ")),
    18: (17, CHO_NAMES.index("ㅅ")),
    19: (0, CHO_NAMES.index("ㅅ")),
    20: (0, CHO_NAMES.index("ㅆ")),
    # 21: ㅇ coda — no linking table entry
    22: (0, CHO_NAMES.index("ㅈ")),
    23: (0, CHO_NAMES.index("ㅊ")),
    24: (0, CHO_NAMES.index("ㅋ")),
    25: (0, CHO_NAMES.index("ㅌ")),
    26: (0, CHO_NAMES.index("ㅍ")),
    27: (0, CHO_NAMES.index("ㅎ")),
}

# Plain obstruent choseong -> tense counterpart (경음화)
_TENSE_PAIR: dict[int, int] = {
    CHO_NAMES.index("ㄱ"): CHO_NAMES.index("ㄲ"),
    CHO_NAMES.index("ㄷ"): CHO_NAMES.index("ㄸ"),
    CHO_NAMES.index("ㅂ"): CHO_NAMES.index("ㅃ"),
    CHO_NAMES.index("ㅅ"): CHO_NAMES.index("ㅆ"),
    CHO_NAMES.index("ㅈ"): CHO_NAMES.index("ㅉ"),
}

# Jong indices that **release** as obstruent codas and trigger tensification on a following plain onset
_JONG_OBSTRUENT_TENSE_TRIGGER: frozenset[int] = frozenset(
    {
        1,
        2,
        3,  # leaves ㄱ — still obstruent remainder after link in some cases; handled per state
        7,
        17,
        18,
        19,
        20,
        22,
        23,
        24,
        25,
        26,
    }
)

_JONG_NASAL: frozenset[int] = frozenset({4, 16, 21})
_CHO_NASAL: frozenset[int] = frozenset(
    {CHO_NAMES.index("ㄴ"), CHO_NAMES.index("ㅁ"), CHO_NAMES.index("ㅇ")}
)


def decompose_syllable(ch: str) -> tuple[int, int, int] | None:
    """Return (cho, jung, jong) indices for a single Hangul syllable, or None."""
    if len(ch) != 1:
        return None
    o = ord(ch)
    if o < HANGUL_BASE or o > HANGUL_END:
        return None
    code = o - HANGUL_BASE
    jong = code % 28
    jung = (code // 28) % 21
    cho = code // 28 // 21
    return cho, jung, jong


def compose_syllable(cho: int, jung: int, jong: int = 0) -> str:
    if not (0 <= cho < 19 and 0 <= jung < 21 and 0 <= jong < 28):
        raise ValueError(f"invalid jamo indices {(cho, jung, jong)}")
    return chr(HANGUL_BASE + cho * 588 + jung * 28 + jong)


@dataclass
class Syllable:
    cho: int
    jung: int
    jong: int
    source_char: str = ""

    def copy(self) -> Syllable:
        return Syllable(self.cho, self.jung, self.jong, self.source_char)


def text_to_syllables(text: str) -> list[Syllable]:
    """
    Collect syllables in NFC codepoint order (Hangul blocks only; other characters skipped).
    """
    text_nfc = unicodedata.normalize("NFC", text.strip())
    out: list[Syllable] = []
    for ch in text_nfc:
        t = decompose_syllable(ch)
        if t is None:
            continue
        cho, jung, jong = t
        out.append(Syllable(cho, jung, jong, source_char=ch))
    return out


# Back-compat names (previously MeCab vs scan; both paths are now this scan).
text_to_syllables_scan = text_to_syllables
text_to_syllables_from_mecab = text_to_syllables


_TOKEN_DEBUG_RE = re.compile(
    r"[\uac00-\ud7a3]+|"
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"\d+(?:[.,]\d+)*|"
    r"\S",
)


def text_tokenization_debug(text: str) -> str:
    """
    Rough Hangul / Latin / digit runs for CLI ``--verbose``.

    This is **not** morphological analysis — only regex grouping for developer visibility.
    """
    nfc = unicodedata.normalize("NFC", text.strip())
    if not nfc:
        return ""
    lines: list[str] = []
    for m in _TOKEN_DEBUG_RE.finditer(nfc):
        frag = m.group()
        if re.fullmatch(r"[\uac00-\ud7a3]+", frag):
            lines.append(f"{frag}\tHangulRun\tsyllables={len(frag)}")
        else:
            lines.append(f"{frag}\tOther")
    return "\n".join(lines)


def apply_linking(syls: list[Syllable]) -> None:
    """연음: mutate list in place."""
    i = 0
    while i < len(syls) - 1:
        cur, nxt = syls[i], syls[i + 1]
        if cur.jong == 0:
            i += 1
            continue
        if cur.jong == JONG_NAMES.index("ㅇ"):  # 21 — coda /ŋ/, no consonant transfer
            i += 1
            continue
        if nxt.cho != IDX_O:
            i += 1
            continue
        spec = _JONG_SPLIT_FOR_LINKING.get(cur.jong)
        if spec is None:
            i += 1
            continue
        rem_jong, new_cho = spec
        cur.jong = rem_jong
        nxt.cho = new_cho
        i += 1


def apply_lateralization(syls: list[Syllable]) -> None:
    """유음화: coda ㄴ before onset ㄹ → coda ㄹ."""
    for i in range(len(syls) - 1):
        if syls[i].jong == 4 and syls[i + 1].cho == CHO_NAMES.index("ㄹ"):
            syls[i].jong = 8


def _jong_triggers_tense(jong: int) -> bool:
    if jong in _JONG_OBSTRUENT_TENSE_TRIGGER:
        return True
    return False


# --- IPA tables (broad Seoul) ---

def _ipa_onset(cho: int, *, tense: bool = False, aspirate: bool = False) -> str:
    if cho == IDX_O:
        return ""
    # Base (plain) onsets
    base = {
        CHO_NAMES.index("ㄱ"): "k",
        CHO_NAMES.index("ㄲ"): "k͈",
        CHO_NAMES.index("ㄴ"): "n",
        CHO_NAMES.index("ㄷ"): "d",
        CHO_NAMES.index("ㄸ"): "t͈",
        CHO_NAMES.index("ㄹ"): "l",
        CHO_NAMES.index("ㅁ"): "m",
        CHO_NAMES.index("ㅂ"): "p",
        CHO_NAMES.index("ㅃ"): "p͈",
        CHO_NAMES.index("ㅅ"): "s",
        CHO_NAMES.index("ㅆ"): "s͈",
        CHO_NAMES.index("ㅈ"): "tɕ",
        CHO_NAMES.index("ㅉ"): "t͈ɕ",
        CHO_NAMES.index("ㅊ"): "tɕʰ",
        CHO_NAMES.index("ㅋ"): "kʰ",
        CHO_NAMES.index("ㅌ"): "tʰ",
        CHO_NAMES.index("ㅍ"): "pʰ",
        CHO_NAMES.index("ㅎ"): "h",
    }
    if cho not in base:
        return ""
    ip = base[cho]
    if tense and cho in _TENSE_PAIR:
        return _ipa_onset(_TENSE_PAIR[cho], tense=False, aspirate=False)
    if aspirate and cho in (CHO_NAMES.index("ㄱ"), CHO_NAMES.index("ㄷ"), CHO_NAMES.index("ㅂ"), CHO_NAMES.index("ㅈ")):
        if cho == CHO_NAMES.index("ㄱ"):
            return "kʰ"
        if cho == CHO_NAMES.index("ㄷ"):
            return "tʰ"  # aspiration of plain /d/ surfaces as [tʰ]
        if cho == CHO_NAMES.index("ㅂ"):
            return "pʰ"
        if cho == CHO_NAMES.index("ㅈ"):
            return "tɕʰ"
    return ip


def _ipa_nucleus(jung: int, _onset_present: bool) -> str:
    """Map jungseong index to a broad vowel/diphthong (Seoul)."""
    vmap: dict[int, str] = {
        0: "a",
        1: "ɛ",
        2: "ja",
        3: "jɛ",
        4: "ʌ",
        5: "e",
        6: "jʌ",
        7: "je",
        8: "o",
        9: "wa",
        10: "wɛ",
        11: "ø",
        12: "jo",
        13: "u",
        14: "wʌ",
        15: "we",
        16: "ɥi",
        17: "ju",
        18: "ɯ",
        19: "ɰi",
        20: "i",
    }
    return vmap.get(jung, "ə")


def _ipa_coda_simple(jong: int) -> str:
    """Neutralized coda (phrase-final or before obstruent); not nasalized yet."""
    if jong == 0:
        return ""
    # Simplified unreleased stops for obstruent codas
    if jong in (1, 2, 24):  # ㄱ ㄲ ㅋ
        return "k̚"
    if jong in (7, 25, 19, 20, 22, 23):  # ㄷ ㅌ ㅅ ㅆ ㅈ ㅊ
        return "t̚"
    if jong in (17, 26, 18):  # ㅂ ㅍ ㅄ
        return "p̚"
    if jong == 4:
        return "n"
    if jong == 8:
        return "l"
    if jong == 16:
        return "m"
    if jong == 21:
        return "ŋ"
    if jong == 27:
        return "t̚"  # ㅎ — often weak; aspiration handled separately
    # Compound remnants (ㄳ left ㄱ, etc.) map like single
    if jong == 3:
        return "k̚"
    if jong in (5, 6):
        return "n"
    if jong in (9, 10, 11, 12, 13, 14, 15):
        return "l"
    return ""


def _coda_nasal_assimilate(jong: int, next_cho: int | None) -> str:
    """
    Before nasal onset, stem-final obstruent becomes homorganic nasal.
    next_cho None = phrase boundary (use simple coda).
    """
    if next_cho is None or next_cho not in _CHO_NASAL:
        return _ipa_coda_simple(jong)
    if next_cho not in (CHO_NAMES.index("ㅁ"), CHO_NAMES.index("ㄴ")):
        return _ipa_coda_simple(jong)
    if jong in (1, 2, 3, 24, 9):  # velars / clusters ending in velar
        return "ŋ"
    if jong in (7, 19, 20, 22, 23, 25, 27, 12, 13, 14, 15):
        return "n"
    if jong in (17, 18, 26, 11, 14):
        return "m"
    return _ipa_coda_simple(jong)


_PLAIN_ASPIRATED_BY_H: frozenset[int] = frozenset(
    {
        CHO_NAMES.index("ㄱ"),
        CHO_NAMES.index("ㄷ"),
        CHO_NAMES.index("ㅂ"),
        CHO_NAMES.index("ㅈ"),
    }
)


def syllables_to_ipa(
    syls: list[Syllable],
    *,
    syllable_sep: str = ".",
) -> str:
    """
    Render syllable list to a single IPA string with optional syllable separators.
    """
    if not syls:
        return ""

    pieces: list[str] = []

    for i, s in enumerate(syls):
        nxt = syls[i + 1] if i + 1 < len(syls) else None
        prev = syls[i - 1] if i > 0 else None
        cho = s.cho

        onset_ipa = ""
        if cho != IDX_O:
            if prev is not None and prev.jong == 27 and cho in _PLAIN_ASPIRATED_BY_H:
                onset_ipa = _ipa_onset(cho, aspirate=True)
            elif prev is not None and _jong_triggers_tense(prev.jong) and cho in _TENSE_PAIR:
                onset_ipa = _ipa_onset(cho, tense=True)
            else:
                onset_ipa = _ipa_onset(cho)

        if s.jung == 20 and cho in (CHO_NAMES.index("ㅅ"), CHO_NAMES.index("ㅆ")):
            onset_ipa = "ɕ͈" if onset_ipa == "s͈" or cho == CHO_NAMES.index("ㅆ") else "ɕ"

        nucleus = _ipa_nucleus(s.jung, bool(onset_ipa or cho != IDX_O))

        coda_ipa = ""
        if s.jong != 0:
            if nxt is not None and s.jong == 27 and nxt.cho in _PLAIN_ASPIRATED_BY_H:
                coda_ipa = ""
            elif nxt is None:
                coda_ipa = _ipa_coda_simple(s.jong)
            elif nxt.cho != IDX_O:
                if nxt.cho in (CHO_NAMES.index("ㄴ"), CHO_NAMES.index("ㅁ")):
                    coda_ipa = _coda_nasal_assimilate(s.jong, nxt.cho)
                else:
                    coda_ipa = _ipa_coda_simple(s.jong)
            else:
                coda_ipa = _ipa_coda_simple(s.jong)

        pieces.append(onset_ipa + nucleus + coda_ipa)

    return syllable_sep.join(pieces)


def korean_g2p(
    text: str,
    *,
    syllable_sep: str = ".",
    **kwargs: object,
) -> str:
    """
    Full pipeline: Hangul syllable scan → linking → lateralization → IPA.
    Non-Hangul characters are ignored for pronunciation (removed from output).

    For backward compatibility, ``use_mecab=...`` is accepted and ignored (MeCab was removed).
    """
    kwargs.pop("use_mecab", None)
    if kwargs:
        bad = ", ".join(sorted(kwargs))
        raise TypeError(f"korean_g2p() got unexpected keyword arguments: {bad}")
    syls = text_to_syllables(text)
    if not syls:
        return ""
    syls = [s.copy() for s in syls]
    apply_linking(syls)
    apply_lateralization(syls)
    return syllables_to_ipa(syls, syllable_sep=syllable_sep)


def main() -> None:
    ap = argparse.ArgumentParser(description="Korean rule G2P (pure Python phonology heuristics).")
    ap.add_argument("text", nargs="*", help="Korean text (or read stdin if empty)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print rough token / script runs (debug).")
    ap.add_argument("--sep", default=".", help="Syllable separator in IPA output (default: period).")
    args = ap.parse_args()
    if args.text:
        raw = " ".join(args.text)
    else:
        import sys

        raw = sys.stdin.read()
    if args.verbose:
        print(text_tokenization_debug(raw))
        print("---")
    ipa = korean_g2p(raw, syllable_sep=args.sep)
    print(ipa)


if __name__ == "__main__":
    main()
