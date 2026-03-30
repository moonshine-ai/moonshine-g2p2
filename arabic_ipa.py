#!/usr/bin/env python3
"""
MSA Arabic → broad IPA for vocoder-style output (space-separated tokens, ``.`` morpheme boundaries).

Operates on **fully diacritized** NFC Unicode (harakat present). Definite-article assimilation
(``sun letters``) is applied when :func:`apply_al_assimilation` is used before conversion.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

# Sun letters: ال assimilates /l/ to the following coronal.
SUN_LETTERS = frozenset("تثدذرزسشصضطظلن")

_AR_COMBINING = frozenset(
    range(0x064B, 0x065F + 1)
) | frozenset({0x0670})  # superscript alif

# Label names from AbderrahmanSkiredj1/arabertv02_tashkeel_fadel → combining strings
DIAC_LABEL_TO_UTF8: dict[str, str] = {
    "X": "",
    "تطويل": "\u0640",  # tatwīl (lengthening stroke; often with ا/و/ي)
    "سكون": "\u0652",
    "شدة": "\u0651",
    "شدة ضمة": "\u0651\u064F",
    "شدة ضمتان": "\u0651\u064C",
    "شدة فتحة": "\u0651\u064E",
    "شدة فتحتان": "\u0651\u064B",
    "شدة كسرة": "\u0651\u0650",
    "شدة كسرتان": "\u0651\u064D",
    "ضمة": "\u064F",
    "ضمتان": "\u064C",
    "فتحة": "\u064E",
    "فتحتان": "\u064B",
    "كسرة": "\u0650",
    "كسرتان": "\u064D",
}


def strip_arabic_diacritics(s: str) -> str:
    """Remove Arabic combining marks (harakat, shadda, etc.)."""
    out: list[str] = []
    for ch in s:
        o = ord(ch)
        if o in _AR_COMBINING or unicodedata.category(ch) == "Mn" and 0x0600 <= o <= 0x06FF:
            continue
        out.append(ch)
    return "".join(out)


def is_arabic_base_letter(ch: str) -> bool:
    if len(ch) != 1:
        return False
    o = ord(ch)
    if o in _AR_COMBINING:
        return False
    # Main Arabic letters + hamza forms; exclude digits and punctuation
    if 0x0621 <= o <= 0x063A:
        return True
    if 0x0641 <= o <= 0x064A:
        return True
    if 0x0671 <= o <= 0x0673:
        return True
    if o in (0x0679, 0x0686, 0x0698, 0x06A4, 0x06AF):
        return True
    return False


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


@lru_cache(maxsize=4096)
def _letter_to_onset_ipa(base: str) -> str:
    """Consonant onset (broad MSA-style); hamza-aware."""
    # Normalize lam-alif ligatures etc.
    m = {
        "ء": "ʔ",
        "أ": "ʔ",
        "إ": "ʔ",
        "آ": "ʔaː",
        "ؤ": "ʔ",
        "ئ": "ʔ",
        "ا": "",  # handled with vowels / long
        "ب": "b",
        "ت": "t",
        "ث": "θ",
        "ج": "dʒ",
        "ح": "ħ",
        "خ": "x",
        "د": "d",
        "ذ": "ð",
        "ر": "r",
        "ز": "z",
        "س": "s",
        "ش": "ʃ",
        "ص": "sˤ",
        "ض": "dˤ",
        "ط": "tˤ",
        "ظ": "ðˤ",
        "ع": "ʕ",
        "غ": "ɣ",
        "ف": "f",
        "ق": "q",
        "ك": "k",
        "ل": "l",
        "م": "m",
        "ن": "n",
        "ه": "h",
        "و": "w",
        "ي": "j",
        "ى": "",  # alif maqsura — vowel length handled with marks
        "ة": "t",  # often /a/ in pause — default /t/ for formal MSA
        "ﻻ": "l",  # NFC rarely has this; lam-ligature
    }
    if base in m:
        return m[base]
    return ""


def _vowel_from_marks(marks: str) -> tuple[str, bool]:
    """Return (ipa_vowel_or_nothing, is_nasalized_tanwin)."""
    if not marks:
        return "", False
    # Order-independent check for shadda + vowel
    has_shadda = "\u0651" in marks
    body = marks.replace("\u0651", "")
    if "\u064E" in body:
        return ("a", False)
    if "\u064F" in body:
        return ("u", False)
    if "\u0650" in body:
        return ("i", False)
    if "\u064B" in body:
        return ("an", True)
    if "\u064C" in body:
        return ("un", True)
    if "\u064D" in body:
        return ("in", True)
    if "\u0652" in body:
        return ("", False)  # sukun
    if "\u0640" in marks:  # tatwīl
        return ("ː", False)
    return ("", False)


def iter_arabic_syllable_clusters(s: str) -> list[tuple[str, str]]:
    """
    Split logical Arabic into (base_letter, combining_diacritics_utf8) clusters.
    Non-Arabic codepoints become (ch, '').
    """
    s = nfc(s)
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(s):
        ch = s[i]
        o = ord(ch)
        if unicodedata.category(ch) == "Mn" and (o in _AR_COMBINING or 0x064B <= o <= 0x065F):
            if out:
                base, prev = out[-1]
                out[-1] = (base, prev + ch)
            i += 1
            continue
        if is_arabic_base_letter(ch) or (0x0600 <= o <= 0x06FF and unicodedata.category(ch) == "Lo"):
            j = i + 1
            marks = ""
            while j < len(s):
                c2 = s[j]
                o2 = ord(c2)
                if unicodedata.category(c2) == "Mn" and (o2 in _AR_COMBINING or 0x064B <= o2 <= 0x065F):
                    marks += c2
                    j += 1
                else:
                    break
            out.append((ch, marks))
            i = j
            continue
        out.append((ch, ""))
        i += 1
    return out


def _has_arabic_vowel_mark(marks: str) -> bool:
    return any(
        c in marks
        for c in (
            "\u064E",
            "\u064F",
            "\u0650",
            "\u064B",
            "\u064C",
            "\u064D",
            "\u0652",
        )
    )


def strip_spurious_tatweil(word: str) -> str:
    """Remove tatwīl (ـ) when it is the only predicted mark (common partial-model artifact)."""
    w = nfc(word)
    if "\u0640" not in w:
        return w
    tmp = w.replace("\u0640", "")
    if strip_arabic_diacritics(tmp) == strip_arabic_diacritics(w):
        return tmp
    return w


def apply_default_fatha_gaps(word: str) -> str:
    """
    After partial ONNX tashkīl, insert fatḥa (َ) on consonants that still lack a short-vowel mark.
    Skips alif/wāw/yā carriers. Heuristic only.
    """
    w = nfc(word)
    if not w:
        return w
    clusters = iter_arabic_syllable_clusters(w)
    out: list[str] = []
    for base, marks in clusters:
        if len(base) == 1 and is_arabic_base_letter(base):
            if base in ("ا", "و", "ي", "ى", "آ", "ة"):
                out.append(base + marks)
                continue
            marks = marks.replace("\u0640", "")
            if marks and not _has_arabic_vowel_mark(marks) and "\u0651" in marks:
                out.append(base + marks)
                continue
            if not _has_arabic_vowel_mark(marks) and "\u0651" not in marks:
                out.append(base + "\u064E" + marks)
            else:
                out.append(base + marks)
        else:
            out.append(base + marks)
    return nfc("".join(out))


def apply_onnx_partial_postprocess(word: str) -> str:
    """Strip tatwīl-only noise, then fill missing short vowels with fatḥa."""
    return apply_default_fatha_gaps(strip_spurious_tatweil(nfc(word)))


def apply_al_assimilation(word: str) -> str:
    """
    If *word* begins with ``ال`` and the next base letter is a sun letter, drop lam from
    pronunciation-oriented **grapheme** processing by returning a pseudo-word where lam is removed
    for IPA (caller merges article vowel + gemination). Here we only return the original word;
    assimilation is applied in :func:`word_to_ipa_with_assimilation`.
    """
    return word


def _geminate_ipa(onset: str) -> str:
    if not onset:
        return ""
    if onset.endswith("ː"):
        return onset  # long vowel marker from hamza+alif
    parts = onset.split()
    if len(parts) == 1 and len(onset) > 1 and onset[1] in "ʔː":
        return onset  # complex
    return f"{onset}.{onset}" if onset else ""


def word_to_ipa_with_assimilation(word: str) -> str:
    """Single word/cluster to IPA including al-+sun assimilation when applicable."""
    w = nfc(word.strip())
    if not w:
        return ""
    bare = strip_arabic_diacritics(w)
    if bare.startswith("ال") and len(bare) >= 3:
        third = bare[2]
        if third in SUN_LETTERS:
            stem = w[2:]  # drop ا ل — keep diacritics on stem
            onset = _letter_to_onset_ipa(third)
            stem_ipa = diacritized_word_to_ipa(stem)
            if stem_ipa.startswith(onset):
                stem_ipa = stem_ipa[len(onset) :].lstrip(".")
            gem = _geminate_ipa(onset)
            return f"a{gem}.{stem_ipa}".strip(".") if stem_ipa else f"a{gem}"
    return diacritized_word_to_ipa(w)


def diacritized_word_to_ipa(word: str) -> str:
    """Convert one word (diacritized) to broad IPA without cross-word rules."""
    clusters = iter_arabic_syllable_clusters(nfc(word))
    parts: list[str] = []
    prev_onset = ""
    expecting_vowel_after = False

    for base, marks in clusters:
        if base.isspace() or base in "،؛؟!" or ord(base) < 32:
            continue
        if not is_arabic_base_letter(base) and unicodedata.category(base) != "Lo":
            continue

        v, _nas = _vowel_from_marks(marks)
        has_sukun = "\u0652" in marks
        has_shadda = "\u0651" in marks
        onset = _letter_to_onset_ipa(base)

        # ا as long carrier / ā
        if base == "ا" and not marks:
            if parts and parts[-1] in ("a", "i", "u"):
                parts[-1] = parts[-1] + "ː"
            else:
                parts.append("aː")
            prev_onset = ""
            expecting_vowel_after = False
            continue

        if base == "ى" and not marks:
            parts.append("aː")
            prev_onset = ""
            continue

        if base == "ة":
            parts.append("a" if not has_sukun and not v else "t")
            prev_onset = ""
            continue

        # و as consonant /w/ vs long uː
        if base == "و":
            if v == "u":
                parts.append("uː")
            elif not marks:
                parts.append("w")
            else:
                parts.append("w" + v)
            prev_onset = ""
            continue

        if base == "ي":
            if v == "i":
                parts.append("iː")
            elif not marks:
                parts.append("j")
            else:
                parts.append("j" + v)
            prev_onset = ""
            continue

        if onset in ("ʔaː",):
            parts.append("ʔaː")
            prev_onset = ""
            continue

        if not onset and base == "ا":
            continue

        # Default consonant
        if has_shadda and onset:
            seg = _geminate_ipa(onset)
        else:
            seg = onset

        if v:
            parts.append(f"{seg}.{v}" if seg else v)
        elif has_sukun and seg:
            parts.append(seg)
        elif seg:
            # implicit short vowel omitted in orthography only happens if model missed — omit
            parts.append(seg)
        prev_onset = seg
        expecting_vowel_after = bool(seg) and not v and not has_sukun

    out = ".".join(p for p in parts if p)
    return out.replace("..", ".").strip(".")


def line_to_ipa_words(line: str) -> str:
    """Whitespace-split; each Arabic token through :func:`word_to_ipa_with_assimilation`."""
    line = nfc(line.strip())
    if not line:
        return ""
    ipa_tokens: list[str] = []
    for raw in line.split():
        w = raw.strip()
        if not w:
            continue
        if not any(0x0600 <= ord(c) <= 0x06FF for c in w):
            continue
        ipa_tokens.append(word_to_ipa_with_assimilation(w))
    return " ".join(ipa_tokens)
