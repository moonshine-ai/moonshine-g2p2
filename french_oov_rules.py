"""
Rule-based French grapheme → broad IPA for **OOV** tokens (no external engine).

Metropolitan-oriented approximations; many silent letters and vowel contexts are only partly
handled. Intended as a fallback when ``data/fr/dict.tsv`` has no entry.
"""

from __future__ import annotations

import re
import unicodedata

_VOWELS = frozenset("aàâäeéèêëiïîoöôuùûüyœæ")

_PRIMARY = "\u02c8"


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _v(ch: str) -> bool:
    if not ch:
        return False
    base = _nfd(ch)[0]
    return base.lower() in "aàâäeéèêëiïîoöôuùûüyœæ"


def _letters_only(word: str) -> str:
    t = unicodedata.normalize("NFC", word.strip().lower())
    return re.sub(r"[^a-zàâäéèêëïîôùûüÿçœæ]", "", t)


def _insert_stress_final_syllable(ipa: str) -> str:
    if not ipa or _PRIMARY in ipa:
        return ipa
    nasals = ("œ̃", "ɑ̃", "ɛ̃", "ɔ̃")
    best_i = -1
    best_len = 0
    for m in nasals:
        j = ipa.rfind(m)
        if j >= best_i:
            best_i, best_len = j, len(m)
    if best_i >= 0:
        return ipa[:best_i] + _PRIMARY + ipa[best_i :]
    n = len(ipa)
    for j in range(n - 1, -1, -1):
        ch = ipa[j]
        if ch in "aeɛəiouyøœɔɑɜ":
            return ipa[:j] + _PRIMARY + ipa[j:]
    return _PRIMARY + ipa


def _trim_final_by_orthography(ipa: str, ortho: str) -> str:
    """Drop word-final consonants that are usually mute in French, guided by spelling."""
    s = ipa
    o = ortho.rstrip("eE")
    if not o or not s:
        return s

    def prev_is_nucleus(idx: int) -> bool:
        if idx < 0:
            return False
        ch = s[idx]
        if ch in "aeɛəiouyøœɔɑ":
            return True
        if ch in "ʁjwɥ":
            return True
        if ch == "\u0303" and idx > 0:
            return True
        return False

    # Word-final /t d/ often silent after vowel or /ʁ/ (chat, port).
    while len(s) >= 2 and s[-1] in "td":
        if prev_is_nucleus(len(s) - 2):
            s = s[:-1]
        else:
            break
    # Word-final /p b/ often silent after vowel (loup, herbe in some analyses — broad strip).
    while len(s) >= 2 and s[-1] in "pb":
        if prev_is_nucleus(len(s) - 2):
            s = s[:-1]
        else:
            break
    # Plural / mute -s, -x, -z (orthographic).
    lastg = o[-1].lower()
    if lastg in "sxz":
        while len(s) >= 2 and s[-1] in "sz":
            if prev_is_nucleus(len(s) - 2):
                s = s[:-1]
            else:
                break
    return s


def oov_word_to_ipa(word: str, *, with_stress: bool = True) -> str:
    """
    Single French token → broad IPA. Empty string if nothing to pronounce after cleanup.
    Hyphenated forms are split, each part converted, joined with ``-``.
    """
    raw = word.strip()
    if not raw:
        return ""
    if "-" in raw:
        chunks = [c for c in raw.split("-") if c]
        if not chunks:
            return ""
        parts = [oov_word_to_ipa(c, with_stress=with_stress) for c in chunks]
        if any(not p for p in parts):
            return ""
        return "-".join(parts)

    ortho_full = _letters_only(raw)
    if not ortho_full:
        return ""
    ipa = _scan_graphemes(ortho_full)
    ipa = _trim_final_by_orthography(ipa, ortho_full)
    if with_stress:
        ipa = _insert_stress_final_syllable(ipa)
    return ipa


def _scan_graphemes(w: str) -> str:
    i = 0
    n = len(w)
    out: list[str] = []

    def peek(k: int = 1) -> str:
        return w[i : i + k]

    def at_word_end(j: int) -> bool:
        return j >= n

    def next_not_vowel(j: int) -> bool:
        return j >= n or not _v(w[j])

    while i < n:
        ch = w[i]

        if ch == "h":
            i += 1
            continue

        # ---- multi-graph (longest first) ----
        if peek(5) == "aient" and (i == 0 or not _v(w[i - 1])):
            out.append("ɛ")
            i += 5
            continue
        if peek(3) == "ant" and at_word_end(i + 3):
            out.append("ɑ̃")
            i += 3
            continue
        if peek(4) == "eaux":
            out.append("o")
            i += 4
            continue
        if peek(3) == "eau":
            out.append("o")
            i += 3
            continue
        if peek(4) == "tion" and next_not_vowel(i + 4):
            out.append("sjɔ̃")
            i += 4
            continue
        if peek(4) == "sion" and next_not_vowel(i + 4):
            out.append("zjɔ̃")
            i += 4
            continue
        if peek(3) == "oin" and next_not_vowel(i + 3):
            out.append("wɛ̃")
            i += 3
            continue
        if peek(3) == "ien" and next_not_vowel(i + 3):
            out.append("jɛ̃")
            i += 3
            continue
        if peek(3) == "ain" and next_not_vowel(i + 3):
            out.append("ɛ̃")
            i += 3
            continue
        if peek(3) == "eil" and next_not_vowel(i + 3):
            out.append("ɛj")
            i += 3
            continue
        if peek(3) == "ail" and next_not_vowel(i + 3):
            out.append("aj")
            i += 3
            continue
        if peek(3) == "oui":
            out.append("wi")
            i += 3
            continue

        if peek(2) == "ou":
            out.append("u")
            i += 2
            continue
        if peek(2) == "oo":
            out.append("u")
            i += 2
            continue
        if peek(2) == "oi":
            out.append("wa")
            i += 2
            continue
        if peek(2) == "ai" or peek(2) == "ei":
            out.append("ɛ")
            i += 2
            continue
        if peek(2) == "au" and (i + 2 >= n or not _v(w[i + 2])):
            out.append("o")
            i += 2
            continue
        if peek(2) == "eu":
            out.append("ø")
            i += 2
            continue
        if ch == "œ" and i + 1 < n and w[i + 1] == "u":
            out.append("ø")
            i += 2
            continue
        if ch == "œ":
            out.append("œ")
            i += 1
            continue
        if ch == "æ":
            out.append("e")
            i += 1
            continue

        # Nasal vowels (not before a vowel letter — crude denasalization boundary).
        if peek(2) in ("an", "am") and next_not_vowel(i + 2):
            out.append("ɑ̃")
            i += 2
            continue
        if peek(2) in ("en", "em") and next_not_vowel(i + 2):
            if i > 0 and w[i - 1] in "iïy":
                out.append("ɛ̃")
            else:
                out.append("ɑ̃")
            i += 2
            continue
        if peek(2) in ("in", "im", "yn", "ym") and next_not_vowel(i + 2):
            out.append("ɛ̃")
            i += 2
            continue
        if peek(2) in ("on", "om") and next_not_vowel(i + 2):
            out.append("ɔ̃")
            i += 2
            continue
        if peek(2) in ("un", "um") and next_not_vowel(i + 2):
            out.append("œ̃")
            i += 2
            continue

        if peek(2) == "qu" and i + 2 < n and _v(w[i + 2]):
            out.append("k")
            i += 2
            continue
        if (
            ch == "g"
            and i + 1 < n
            and w[i + 1] == "u"
            and i + 2 < n
            and w[i + 2] in "eéèêëiïy"
        ):
            out.append("ɡ")
            i += 2
            continue

        if peek(2) == "ch":
            out.append("ʃ")
            i += 2
            continue
        if peek(2) == "gn":
            out.append("ɲ")
            i += 2
            continue
        if peek(2) == "ph":
            out.append("f")
            i += 2
            continue
        if peek(2) == "th":
            out.append("t")
            i += 2
            continue

        if ch == "c" and i + 1 < n and w[i + 1] == "ç":
            out.append("ks")
            i += 2
            continue

        # ---- single letters & digraphs starting with c/g ----
        if ch == "ç":
            out.append("s")
            i += 1
            continue
        if ch == "c":
            if i + 1 < n and w[i + 1] in "eéèêëiïy":
                out.append("s")
            else:
                out.append("k")
            i += 1
            continue
        if ch == "g":
            if i + 1 < n and w[i + 1] in "eéèêëiïy":
                out.append("ʒ")
            else:
                out.append("ɡ")
            i += 1
            continue
        if ch == "x":
            if not out:
                if i + 1 < n and _v(w[i + 1]):
                    out.append("ɡz")
                else:
                    out.append("ks")
            else:
                last = out[-1]
                if last and last[-1] in "aeɛəiouyøœɔɑ" or last.endswith("̃"):
                    out.append("z")
                else:
                    out.append("ks")
            i += 1
            continue

        if _v(ch):
            if ch in "aàâ":
                out.append("a")
            elif ch in "ä":
                out.append("a")
            elif ch == "é":
                out.append("e")
            elif ch in "èêë":
                out.append("ɛ")
            elif ch == "e":
                if at_word_end(i + 1):
                    i += 1
                    continue
                if i + 1 < n and not _v(w[i + 1]):
                    out.append("ə")
                else:
                    out.append("e")
                i += 1
                continue
            elif ch in "iïî":
                out.append("i")
            elif ch in "oô":
                out.append("o")
            elif ch == "ö":
                out.append("ø")
            elif ch in "uùû":
                out.append("y")
            elif ch == "ü":
                out.append("y")
            elif ch == "y":
                out.append("i")
            elif ch == "œ":
                out.append("œ")
            elif ch == "æ":
                out.append("e")
            else:
                out.append("a")
            i += 1
            continue

        # consonants
        cons = {
            "b": "b",
            "d": "d",
            "f": "f",
            "j": "ʒ",
            "k": "k",
            "l": "l",
            "m": "m",
            "n": "n",
            "p": "p",
            "q": "k",
            "r": "ʁ",
            "s": "s",
            "t": "t",
            "v": "v",
            "w": "w",
            "z": "z",
        }
        if ch in cons:
            out.append(cons[ch])
            i += 1
            continue

        i += 1

    return "".join(out)
