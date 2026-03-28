#!/usr/bin/env python3
"""
Rule- and lexicon-based Italian grapheme-to-phoneme (broad IPA) for vocoding.

* In-vocabulary words are read from ``data/it/dict.tsv`` (word ``\\t`` IPA). When the same
  normalized key appears from a capitalized row and later from an all-lowercase lemma, the
  **lowercase** row wins (same policy as ``german_rule_g2p`` / ``dutch_rule_g2p``).
* Out-of-vocabulary tokens use orthographic syllables, a default stress rule (written accent
  ``àèéìíòóù`` / acute ``é ó``; else penultimate if the word ends in a vowel, otherwise ultimate),
  and a compact letter pass (``c/g`` before ``e/i``, ``sc``, ``gn``, ``gli``, ``gl`` + ``i``,
  ``ch/gh``, geminates, ``z``/``zz``, etc.).

Limitations (intentional):
- Open/closed ``e``/``o`` quality for unaccented letters defaults to mid values; ``è`` ``ò``
  ``é`` ``ó`` disambiguate where written.
- Intervocalic *s* voicing, ``z`` ~ */ts/* vs */dz/*, and full ``gli`` / ``sc`` / ``cc`` rules are
  only partly handled; the lexicon covers high-frequency lemmas.
- Elision (*c'è*, *l'amico*, *po'*, *'ndrangheta*) is one token; typographic apostrophe (U+2019)
  is normalized to ASCII for lexicon keys.
- Digit-only tokens (and ``1933-1945``-style ranges) expand to Italian cardinals via
  :mod:`italian_numbers` before G2P (up to 999_999); disable with ``expand_cardinal_digits=False``
  or CLI ``--no-expand-digits``.

By default, stress from **rule-based** IPA is passed through
:func:`german_rule_g2p.normalize_ipa_stress_for_vocoder` (nuclear, eSpeak-style). Lexicon strings
keep ipa-dict stress placement. Pass ``--syllable-initial-stress`` to disable vocoder shifting.

CLI: prints rule/lexicon IPA, then an eSpeak NG reference line by default (``--no-espeak`` to
disable), same stack as ``german_rule_g2p`` / ``dutch_rule_g2p``.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

from german_rule_g2p import normalize_ipa_stress_for_vocoder
from italian_numbers import expand_cardinal_digits_to_italian_words, expand_digit_tokens_in_text

_DEFAULT_ESPEAK_VOICE = "it"

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT_PATH = _REPO_ROOT / "data" / "it" / "dict.tsv"

_LEXICON_CACHE: dict[str, str] | None = None
_LEXICON_PATH: Path | None = None

_PRIMARY_STRESS = "\u02c8"  # ˈ
_SECONDARY_STRESS = "\u02cc"  # ˌ


def espeak_ng_ipa_line(text: str, *, voice: str = _DEFAULT_ESPEAK_VOICE) -> str | None:
    """
    IPA string from libespeak-ng via ``espeak_phonemizer`` (same separator policy as
    :func:`heteronym.espeak_heteronyms.espeak_phonemize_ipa_raw`).
    """
    try:
        from heteronym.espeak_heteronyms import EspeakPhonemizer, espeak_phonemize_ipa_raw
    except ImportError:
        return None
    t = text.strip()
    if not t:
        return None
    try:
        phon = EspeakPhonemizer(default_voice=voice)
        raw = espeak_phonemize_ipa_raw(phon, t, voice=voice)
    except (AssertionError, OSError, RuntimeError):
        return None
    return raw or None


def normalize_lookup_key(word: str) -> str:
    """Lowercase NFC key for TSV lookup (Italian letters, apostrophe, hyphen)."""
    t = unicodedata.normalize("NFC", word.strip().lower())
    t = t.replace("\u2019", "'")  # ’ → ' (match ipa-dict / ASCII TSV keys)
    return re.sub(r"[^a-zàèéìíîòóùú'`\-]+", "", t)


def load_italian_lexicon(path: Path | None = None) -> dict[str, str]:
    """
    Load ``word\\tIPA`` TSV. Duplicate keys: prefer an all-lowercase surface row over a
    capitalized one (homograph policy).
    """
    p = path or _DEFAULT_DICT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Italian lexicon not found: {p}")
    m: dict[str, tuple[str, bool]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            surf, ipa = parts[0].strip(), parts[1].strip()
            k = normalize_lookup_key(surf)
            if not k:
                continue
            lower_lemma = surf == surf.lower()
            if k not in m:
                m[k] = (ipa, lower_lemma)
            else:
                _prev_ipa, prev_lower = m[k]
                if lower_lemma and not prev_lower:
                    m[k] = (ipa, True)
    return {k: v[0] for k, v in m.items()}


def _get_lexicon(path: Path | None) -> dict[str, str]:
    global _LEXICON_CACHE, _LEXICON_PATH
    p = path or _DEFAULT_DICT_PATH
    if _LEXICON_CACHE is not None and _LEXICON_PATH == p:
        return _LEXICON_CACHE
    if not p.is_file():
        _LEXICON_CACHE = {}
        _LEXICON_PATH = p
        return _LEXICON_CACHE
    _LEXICON_CACHE = load_italian_lexicon(p)
    _LEXICON_PATH = p
    return _LEXICON_CACHE


def _lookup_lexicon(key: str, lexicon: Mapping[str, str]) -> str | None:
    if key in lexicon:
        return lexicon[key]
    return None


# When ``expand_cardinal_digits`` is off: pass digit / simple year-range tokens through unchanged.
_DIGIT_PASS_THROUGH_RE = re.compile(r"^[0-9]+(?:-[0-9]+)*$")


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _strip_accents(ch: str) -> str:
    return "".join(c for c in _nfd(ch) if unicodedata.category(c) != "Mn") or ch


# Vowel letters used for syllabification (Italian + common accents).
_VOWELS_IT = frozenset("aeiouàèéìíîòóùú")


def _is_vowel_ch(ch: str) -> bool:
    return ch.lower() in _VOWELS_IT


def _should_hiatus_it(a: str, b: str) -> bool:
    """Hiatus vs diphthong between adjacent vowel letters (orthographic heuristic)."""
    al, bl = a.lower(), b.lower()
    if al in "íì" or bl in "íì":
        return True
    if al in "úù" or bl in "úù":
        return True
    ba, bb = _strip_accents(al), _strip_accents(bl)
    if ba in "aeo" and bb in "aeo":
        return True
    if ba in "iu" and bb in "aeo":
        return False
    if ba in "aeo" and bb in "iu":
        return False
    if ba == bb:
        return True
    if ba in "iu" and bb in "iu":
        return False
    return True


def _vowel_nucleus_spans(w: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    n = len(w)
    while i < n:
        ch = w[i]
        if not _is_vowel_ch(ch):
            i += 1
            continue
        if i + 1 < n and _is_vowel_ch(w[i + 1]):
            if _should_hiatus_it(ch, w[i + 1]):
                out.append((i, i + 1))
                i += 1
            else:
                out.append((i, i + 2))
                i += 2
        else:
            out.append((i, i + 1))
            i += 1
    return out


_VALID_ONSETS_2 = frozenset({"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "tr", "ch"})


def _split_intervocalic_cluster(cluster: str) -> tuple[str, str]:
    if not cluster:
        return "", ""
    if len(cluster) >= 2 and cluster[-2:] in _VALID_ONSETS_2:
        return cluster[:-2], cluster[-2:]
    return cluster[:-1], cluster[-1:]


def italian_orthographic_syllables(word: str) -> list[str]:
    """
    Rough orthographic syllables (onset maximization between nuclei). Hyphens split compounds
    before syllabifying each piece.
    """
    w = re.sub(r"[^a-zàèéìíîòóùúA-ZÀÈÉÌÍÎÒÓÙÚ\-]", "", word.lower())
    if not w:
        return []
    if "-" in w:
        parts: list[str] = []
        for chunk in w.split("-"):
            if chunk:
                parts.extend(italian_orthographic_syllables(chunk))
        return parts
    spans = _vowel_nucleus_spans(w)
    if not spans:
        return [w]
    syllables: list[str] = []
    first_s = spans[0][0]
    cur = w[:first_s]
    for idx, (s, e) in enumerate(spans):
        cur += w[s:e]
        if idx + 1 < len(spans):
            next_s = spans[idx + 1][0]
            cluster = w[e:next_s]
            coda, onset = _split_intervocalic_cluster(cluster)
            syllables.append(cur + coda)
            cur = onset
        else:
            syllables.append(cur + w[e:])
    return [s for s in syllables if s]


_ACCENTED_VOWELS = frozenset("àèéìíòóùúî")


def default_stressed_syllable_index(syls: list[str], word_lower: str) -> int:
    """0-based stressed syllable index from orthography."""
    if not syls:
        return 0
    w = re.sub(r"[^a-zàèéìíîòóùú\-]", "", word_lower)
    if any(c in _ACCENTED_VOWELS for c in w):
        for i, s in enumerate(syls):
            if any(c in _ACCENTED_VOWELS for c in s):
                return i
    n = len(syls)
    if n == 1:
        return 0
    tail = w.rstrip("-")
    if not tail:
        return 0
    last_letter = _strip_accents(tail[-1].lower())
    if last_letter in "aeiou":
        return max(0, n - 2)
    return n - 1


def _insert_primary_stress_before_vowel(ipa: str) -> str:
    s = ipa.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
    for i, ch in enumerate(s):
        if ch in "aeiouɛɔ":
            return s[:i] + _PRIMARY_STRESS + s[i:]
    return _PRIMARY_STRESS + s


def _next_is_vowel(s: str, j: int) -> bool:
    k = j
    n = len(s)
    while k < n:
        if s[k] == "h":
            k += 1
            continue
        return _is_vowel_ch(s[k])
    return False


def _letters_to_ipa_no_stress(s: str) -> str:
    """Map one syllable chunk (lowercase Italian letters) to IPA without stress."""
    n = len(s)
    i = 0
    out: list[str] = []

    while i < n:
        if s[i] == "-":
            i += 1
            continue

        # geminate zz → /tts/ (broad, matches many ipa-dict entries)
        if i + 1 < n and s[i] == "z" == s[i + 1]:
            out.append("tt͡s")
            i += 2
            continue

        # cc before e,i → /ttʃ/
        if (
            i + 2 < n
            and s[i] == "c"
            and s[i + 1] == "c"
            and s[i + 2].lower() in "eiéè"
        ):
            out.append("tt͡ʃ")
            i += 3
            continue

        # gg before e,i → /ddʒ/
        if (
            i + 2 < n
            and s[i] == "g"
            and s[i + 1] == "g"
            and s[i + 2].lower() in "eiéè"
        ):
            out.append("dd͡ʒ")
            i += 3
            continue

        # gn → /ɲɲ/
        if i + 1 < n and s[i] == "g" and s[i + 1] == "n":
            out.append("ɲɲ")
            i += 2
            continue

        # gli before a,e,o,u → /ʎ/ + vowel; word "gli" → /ʎi/
        if i + 2 < n and s[i : i + 3] == "gli":
            nxt = s[i + 3] if i + 3 < n else ""
            if nxt.lower() in "aeiouàèéìòóù" or nxt == "":
                out.append("ʎ")
                i += 3
                continue
            if nxt.lower() == "i" and (i + 4 >= n or not _is_vowel_ch(s[i + 4])):
                out.append("ʎ")
                i += 3
                continue

        # gl + i (figlio) → /ʎ/, else /ɡl/
        if i + 2 < n and s[i] == "g" and s[i + 1] == "l" and s[i + 2].lower() == "i":
            prev_cons = i == 0 or not _is_vowel_ch(s[i - 1])
            next_after = s[i + 3] if i + 3 < n else ""
            if prev_cons and (next_after == "" or _is_vowel_ch(next_after)):
                out.append("ʎ")
                i += 3
                continue

        # ch → /k/
        if i + 1 < n and s[i : i + 2] == "ch":
            out.append("k")
            i += 2
            continue

        # gh before e,i → /g/, silent h
        if i + 2 < n and s[i : i + 2] == "gh" and s[i + 2].lower() in "eiéè":
            out.append("ɡ")
            i += 3
            continue

        # sc before e,i → /ʃ/
        if i + 2 < n and s[i : i + 2] == "sc" and s[i + 2].lower() in "eiéè":
            out.append("ʃ")
            i += 3
            continue

        # sc before a,o,u → /sk/
        if i + 2 < n and s[i : i + 2] == "sc" and s[i + 2].lower() in "aouàòù":
            out.append("sk")
            i += 3
            continue

        # qu → /kw/
        if i + 1 < n and s[i : i + 2] == "qu":
            out.append("kw")
            i += 2
            continue

        # gu before e,i (silent u): gue, gui, …
        if i + 2 < n and s[i] == "g" and s[i + 1] == "u" and s[i + 2].lower() in "eiéè":
            out.append("ɡ")
            i += 2
            continue

        # ci + vowel → /tʃ/ + vowel (cia, cie, …)
        if i + 2 < n and s[i] == "c" and s[i + 1] == "i" and _is_vowel_ch(s[i + 2]):
            out.append("t͡ʃ")
            i += 2
            continue

        # gi + vowel → /dʒ/ + vowel
        if i + 2 < n and s[i] == "g" and s[i + 1] == "i" and _is_vowel_ch(s[i + 2]):
            out.append("d͡ʒ")
            i += 2
            continue

        # c before e,i → /tʃ/
        if s[i] == "c" and i + 1 < n and s[i + 1].lower() in "eiéè":
            out.append("t͡ʃ")
            i += 2
            continue

        # g before e,i → /dʒ/ (if not already handled)
        if s[i] == "g" and i + 1 < n and s[i + 1].lower() in "eiéè":
            out.append("d͡ʒ")
            i += 2
            continue

        ch = s[i]

        if ch == "h":
            i += 1
            continue

        # double consonants (excluding zz, cc, gg handled)
        if i + 1 < n and ch == s[i + 1] and ch.isalpha() and ch not in "aeiouàèéìíîòóùú":
            if ch in "bcdfglmnpstv":
                out.append(ch + ch)
            else:
                out.append(ch)
            i += 2
            continue

        if ch == "c":
            out.append("k")
            i += 1
            continue

        if ch == "g":
            out.append("ɡ")
            i += 1
            continue

        if ch == "q":
            if i + 1 < n and s[i + 1] == "u" and _next_is_vowel(s, i + 2):
                out.append("k")
                i += 2
                continue
            out.append("k")
            i += 1
            continue

        if ch == "s":
            prev_v = i > 0 and _is_vowel_ch(s[i - 1])
            next_v = i + 1 < n and _next_is_vowel(s, i + 1)
            out.append("z" if prev_v and next_v else "s")
            i += 1
            continue

        if ch == "z":
            prev_v = i > 0 and _is_vowel_ch(s[i - 1])
            next_v = i + 1 < n and _next_is_vowel(s, i + 1)
            out.append("d͡z" if prev_v and next_v else "t͡s")
            i += 1
            continue

        if ch == "x":
            out.append("ks")
            i += 1
            continue

        if ch == "j":
            out.append("j")
            i += 1
            continue

        if ch == "w":
            out.append("w")
            i += 1
            continue

        if ch == "k":
            out.append("k")
            i += 1
            continue

        if _is_vowel_ch(ch):
            cl = ch.lower()
            if i + 1 < n and _is_vowel_ch(s[i + 1]):
                nxt = s[i + 1].lower()
                # Falling diphthongs (broad; aligns with many ipa-dict Italian entries).
                if cl in "aà" and nxt in "uùú":
                    out.append("aw")
                    i += 2
                    continue
                if cl in "aà" and nxt in "iíì":
                    out.append("aj")
                    i += 2
                    continue
                if cl in "eéè" and nxt in "iíì":
                    out.append("ej")
                    i += 2
                    continue
                if cl in "oóò" and nxt in "iíì":
                    out.append("oj")
                    i += 2
                    continue
                if cl in "eéè" and nxt in "uùú":
                    out.append("ɛw")
                    i += 2
                    continue
                if cl in "oóò" and nxt in "uùú":
                    out.append("ow")
                    i += 2
                    continue
            if cl in "aà":
                out.append("a")
            elif cl in "eé":
                out.append("e")
            elif cl in "èê":
                out.append("ɛ")
            elif cl in "iíìî":
                out.append("i")
            elif cl in "oó":
                out.append("o")
            elif cl in "ò":
                out.append("ɔ")
            elif cl in "uùú":
                out.append("u")
            else:
                out.append("a")
            i += 1
            continue

        simple = {
            "b": "b",
            "d": "d",
            "f": "f",
            "l": "l",
            "m": "m",
            "n": "n",
            "p": "p",
            "r": "r",
            "t": "t",
            "v": "v",
        }
        if ch in simple:
            out.append(simple[ch])
            i += 1
            continue

        i += 1

    return "".join(out)


def _rules_word_to_ipa(word: str, *, with_stress: bool) -> str:
    w = re.sub(
        r"[^a-zàèéìíîòóùúA-ZÀÈÉÌÍÎÒÓÙÚ\-]",
        "",
        word.strip(),
    )
    if not w:
        return ""
    wl = w.lower()
    syls = italian_orthographic_syllables(wl)
    if not syls:
        return ""
    stress_idx = default_stressed_syllable_index(syls, wl) if with_stress else -1
    parts: list[str] = []
    for idx, sy in enumerate(syls):
        chunk = _letters_to_ipa_no_stress(sy)
        if with_stress and idx == stress_idx and chunk:
            chunk = _insert_primary_stress_before_vowel(chunk)
        parts.append(chunk)
    return "".join(parts)


def _finalize_word_ipa(
    ipa: str,
    *,
    with_stress: bool,
    vocoder_stress: bool,
    from_lexicon: bool,
) -> str:
    if not with_stress:
        return ipa.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
    if vocoder_stress and not from_lexicon:
        return normalize_ipa_stress_for_vocoder(ipa)
    return ipa


# Very common function words where syllabification + default stress can diverge from usual speech.
_FUNCTION_WORD_IPA: dict[str, str] = {
    "e": "e",
    "ed": "ed",
    "o": "o",
    "a": "a",
    "i": "i",
    "il": "il",
    "lo": "lo",
    "la": "la",
    "le": "le",
    "gli": "ʎi",
    "un": "un",
    "uno": "ˈuno",
    "una": "ˈuna",
    "di": "di",
    "da": "da",
    "in": "in",
    "su": "su",
    "per": "per",
    "tra": "tra",
    "fra": "fra",
    "del": "del",
    "della": "ˈdɛlla",
    "delle": "ˈdɛlle",
    "dei": "ˈdei",
    "degli": "ˈdeʎʎi",
    "al": "al",
    "allo": "ˈallo",
    "alla": "ˈalla",
    "ai": "ai",
    "agli": "ˈaʎʎi",
    "alle": "ˈalle",
    "nel": "nel",
    "nello": "ˈnɛllo",
    "nella": "ˈnɛlla",
    "nell": "nɛll",
    "sul": "sul",
    "sullo": "ˈsullo",
    "sulla": "ˈsulla",
    "col": "kol",
    "coi": "ˈkoi",
    "ci": "t͡ʃi",
    "vi": "vi",
    "si": "si",
    "ti": "ti",
    "mi": "mi",
    "non": "non",
    "che": "ke",
}


def word_to_ipa(
    word: str,
    *,
    lexicon: Mapping[str, str] | None = None,
    dict_path: Path | None = None,
    with_stress: bool = True,
    vocoder_stress: bool = True,
    expand_cardinal_digits: bool = True,
) -> str:
    """
    Single-token G2P: lexicon lookup, optional function-word map, else rules.

    Lexicon IPA keeps stress marks from the TSV; rule-based IPA can be shifted for vocoders
    when *vocoder_stress* is True.

    Pure digit strings expand via :func:`italian_numbers.expand_cardinal_digits_to_italian_words`
    when *expand_cardinal_digits* is True (ranges like ``1933-1945`` are expanded in
    :func:`text_to_ipa` before tokenization).
    """
    if not word or not word.strip():
        return ""
    raw = word.strip()

    if expand_cardinal_digits and raw.isdigit():
        phrase = expand_cardinal_digits_to_italian_words(raw)
        if phrase != raw:
            return text_to_ipa(
                phrase,
                lexicon=lexicon,
                dict_path=dict_path,
                with_stress=with_stress,
                vocoder_stress=vocoder_stress,
                expand_cardinal_digits=False,
            )
        return raw

    if not expand_cardinal_digits and _DIGIT_PASS_THROUGH_RE.fullmatch(raw):
        return raw

    letters_only = normalize_lookup_key(raw)
    if not letters_only:
        return ""

    lex = lexicon if lexicon is not None else _get_lexicon(dict_path)
    ipa = _lookup_lexicon(letters_only, lex)
    if ipa is not None:
        return _finalize_word_ipa(
            ipa,
            with_stress=with_stress,
            vocoder_stress=vocoder_stress,
            from_lexicon=True,
        )

    if "-" in letters_only:
        chunks = [c for c in letters_only.split("-") if c]
        sub = [_lookup_lexicon(c, lex) for c in chunks]
        if chunks and all(x is not None for x in sub):
            merged = "-".join(sub)
            return _finalize_word_ipa(
                merged,
                with_stress=with_stress,
                vocoder_stress=vocoder_stress,
                from_lexicon=True,
            )

    if letters_only in _FUNCTION_WORD_IPA:
        ipa_fw = _FUNCTION_WORD_IPA[letters_only]
        return _finalize_word_ipa(
            ipa_fw,
            with_stress=with_stress,
            vocoder_stress=vocoder_stress,
            from_lexicon=False,
        )

    ipa_rules = _rules_word_to_ipa(raw, with_stress=with_stress)
    return _finalize_word_ipa(
        ipa_rules,
        with_stress=with_stress,
        vocoder_stress=vocoder_stress,
        from_lexicon=False,
    )


# ASCII U+0027 and typographic U+2019 (must not use raw-string \\u2019 — that is literal backslash-u).
_APOSTROPHE_IN_CLASS = "\u0027\u2019"
# c'è, l'amico, dell'arte, po', l', 'ndrangheta — merge so lexicon keys with ' match.
_ITALIAN_WORD_PATTERN = (
    rf"(?:[{_APOSTROPHE_IN_CLASS}]?[\w\-]+(?:[{_APOSTROPHE_IN_CLASS}][\w\-]+)*(?:[{_APOSTROPHE_IN_CLASS}])?)"
)
_TOKEN_RE = re.compile(
    rf"{_ITALIAN_WORD_PATTERN}|[^\w\s{_APOSTROPHE_IN_CLASS}\-]+|[{_APOSTROPHE_IN_CLASS}]|\s+",
    flags=re.UNICODE,
)
_ITALIAN_WORD_FULLMATCH = re.compile(rf"{_ITALIAN_WORD_PATTERN}\Z", flags=re.UNICODE)


def text_to_ipa(
    text: str,
    *,
    lexicon: Mapping[str, str] | None = None,
    dict_path: Path | None = None,
    with_stress: bool = True,
    vocoder_stress: bool = True,
    expand_cardinal_digits: bool = True,
) -> str:
    """Tokenize and G2P each word; preserve punctuation and collapse spaces."""
    if expand_cardinal_digits:
        text = expand_digit_tokens_in_text(text)
    parts: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isspace():
            parts.append(" ")
        elif _ITALIAN_WORD_FULLMATCH.fullmatch(tok):
            parts.append(
                word_to_ipa(
                    tok,
                    lexicon=lexicon,
                    dict_path=dict_path,
                    with_stress=with_stress,
                    vocoder_stress=vocoder_stress,
                    expand_cardinal_digits=False,
                )
            )
        else:
            parts.append(tok)
    out = "".join(parts)
    return re.sub(r" +", " ", out).strip()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Italian text to IPA using data/it/dict.tsv plus rules for OOV."
    )
    p.add_argument("text", nargs="*", help="Italian text (if empty, read stdin)")
    p.add_argument(
        "--dict",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Lexicon TSV (default: {_DEFAULT_DICT_PATH}).",
    )
    p.add_argument("--no-stress", action="store_true", help="Strip stress marks from output.")
    p.add_argument(
        "--syllable-initial-stress",
        action="store_true",
        help="Keep ˈ/ˌ before the first segment of the stressed syllable; default moves them before the nucleus.",
    )
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin.")
    p.add_argument(
        "--no-expand-digits",
        action="store_true",
        help="Leave digit sequences as digits (no spoken Italian cardinal expansion).",
    )
    p.add_argument("--no-espeak", action="store_true", help="Do not print eSpeak NG reference line.")
    p.add_argument(
        "--espeak-voice",
        type=str,
        default=_DEFAULT_ESPEAK_VOICE,
        metavar="VOICE",
        help=f"eSpeak voice (default: {_DEFAULT_ESPEAK_VOICE}).",
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.stdin or not args.text:
        import sys

        raw = sys.stdin.read()
    else:
        raw = " ".join(args.text)
    lex = load_italian_lexicon(args.dict) if args.dict is not None else None
    expand_digits = not args.no_expand_digits
    es_in = expand_digit_tokens_in_text(raw) if expand_digits else raw
    print(
        text_to_ipa(
            raw,
            lexicon=lex,
            dict_path=args.dict,
            with_stress=not args.no_stress,
            vocoder_stress=not args.syllable_initial_stress,
            expand_cardinal_digits=expand_digits,
        )
    )
    if not args.no_espeak:
        es = espeak_ng_ipa_line(es_in, voice=args.espeak_voice)
        if es is not None:
            print(f"{es} (espeak-ng)")


if __name__ == "__main__":
    main()
