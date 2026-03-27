#!/usr/bin/env python3
"""
Rule- and lexicon-based Dutch grapheme-to-phoneme (Standard Dutch / ABN, broad IPA).

* In-vocabulary words are taken from ``data/nl/dict.tsv`` (word ``\\t`` IPA, first line wins
  for duplicate spellings unless a later all-lowercase row overrides a capitalized one —
  same policy as ``german_rule_g2p``). Run ``scripts/download_multilingual_ipa_lexicons.py --only nl``
  to fetch the file (ipa-dict / CC BY).
* Out-of-vocabulary tokens use the rule pass described below, plus a tiny
  :data:`_OOV_SUPPLEMENT_IPA` for compounds missing from ipa-dict (validated vs eSpeak-ng ``nl``).

Lexicon IPA is lightly post-processed (see :func:`_apply_lexicon_ipa_postprocess`) where ipa-dict
systematically disagrees with Wiktionary ABN and eSpeak-ng.

Digit-only tokens (and ``1933-1945``-style ranges) are expanded to Dutch cardinals via
:mod:`dutch_numbers` before G2P (up to 999_999); disable with ``expand_cardinal_digits=False``
or CLI ``--no-expand-digits``.

Uses rough orthographic syllables and simple stress heuristics (prefixes like *ge-* / *ver-*,
suffixes like *-atie* / *-iteit*, else first syllable), with optional written stress on
*á é í ó ú* (rare in modern Dutch except *café*-type words).

Limitations (intentional):
- *ie* is split between *niet*-style /i/ (before *t/s/d* at a morpheme edge) and default /iː/.
- *g* / *ch* are simplified (velar fricatives; no full Northern/Southern split).
- *r* is a generic alveolar /r/; *w* is /ʋ/.
- Loanword *c* (*cent* vs *club*) uses *ce/ci* → /s/, else /k/.
- Schwa syncope and voicing assimilation are not modeled.

Primary stress is written immediately before the syllable nucleus (eSpeak-style). Pass
``--syllable-initial-stress`` to keep dictionary-style marks at the first segment of the
stressed syllable.

CLI: prints rule IPA, then an eSpeak NG reference line by default (``--no-espeak`` to disable),
same stack as ``german_rule_g2p`` / ``spanish_rule_g2p``.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

from dutch_numbers import expand_cardinal_digits_to_dutch_words, expand_digit_tokens_in_text

_DEFAULT_ESPEAK_VOICE = "nl"

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT_PATH = _REPO_ROOT / "data" / "nl" / "dict.tsv"

_LEXICON_CACHE: dict[str, str] | None = None
_LEXICON_PATH: Path | None = None

# Primary / secondary stress (modifier letters, NFC)
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


def normalize_lexicon_key(word: str) -> str:
    """
    Lowercase NFC key for TSV lookup: *ij* ligature → ``ij``, combining accents stripped,
    ``a-z`` and hyphen only (ipa-dict entries are mostly unaccented ASCII).
    """
    t = unicodedata.normalize("NFC", word.strip().lower())
    t = t.replace("\u0133", "ij")  # ĳ
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = unicodedata.normalize("NFC", t)
    return re.sub(r"[^a-z-]+", "", t)


def load_dutch_lexicon(path: Path | None = None) -> dict[str, str]:
    """
    Load ``word\\tIPA`` TSV. Keys from :func:`normalize_lexicon_key` (surface column).

    When the same key appears from a capitalized row and later from an all-lowercase lemma,
    the **lowercase** row wins (homograph disambiguation, same as German).
    """
    p = path or _DEFAULT_DICT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Dutch lexicon not found: {p}")
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
            k = normalize_lexicon_key(surf)
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
    _LEXICON_CACHE = load_dutch_lexicon(p)
    _LEXICON_PATH = p
    return _LEXICON_CACHE


def _lookup_lexicon(key: str, lexicon: Mapping[str, str]) -> str | None:
    if key in lexicon:
        return lexicon[key]
    return None


# When ``expand_cardinal_digits`` is off: pass digit / simple year-range tokens through unchanged.
_DIGIT_PASS_THROUGH_RE = re.compile(r"^[0-9]+(?:-[0-9]+)*$")


def _apply_lexicon_ipa_postprocess(ipa: str, word_key: str) -> str:
    """
    Correct systematic ipa-dict biases checked against Wiktionary (Dutch) and eSpeak-ng ``nl``.

    * ``architect`` family: ipa-dict uses /ʒ/; standard Dutch is /x/ or /ʃ/ (Wiktionary
      ``/ˌɑrxiˈtɛkt/, /ˌɑrʃiˈtɛkt/``), and eSpeak-ng uses a velar fricative cluster.
    * ⟨g⟩ as /x/: ipa-dict uses /x/ for many native words; ABN and eSpeak-ng use /ɣ/
      (e.g. *groot* ``/ɣroːt/``, *gevangenisstraf* with syllable-initial ``xə`` from ⟨ge⟩).
    """
    s = ipa.replace("ɑr.ʒiː", "ɑr.xi")
    if word_key[:1] == "g":
        if s.startswith("x"):
            s = "ɣ" + s[1:]
        elif len(s) >= 2 and s[0] in (_PRIMARY_STRESS, _SECONDARY_STRESS) and s[1] == "x":
            s = s[0] + "ɣ" + s[2:]
    return s


# Compounds / inflections absent from ipa-dict but attested in running text (checked vs eSpeak-ng ``nl``).
_OOV_SUPPLEMENT_IPA: dict[str, str] = {
    # Plural of *architect* (Wiktionary); TSV has singular only with the old /ʒ/ digraph.
    "architecten": "ɑr.xiˈtɛktən",
    # TSV has *rijksarchief* etc.; this headword is missing.
    "rijksarchitect": "ˈrɛiks.ɑr.xi.ˈtɛkt",
    # Long compound missing from TSV.
    "naziheerschappij": "ˈnaː.tsiː.heːr.sxɑ.ˈpɛi",
    "stedenbouwkundige": "stˈeːdənbʌʊkˈɵndɪɣə",
}


# Very common tokens where the letter rules mis-stress or mis-read vowel length.
_FUNCTION_WORD_IPA: dict[str, str] = {
    "de": "də",
    "het": "ɦət",
    "een": "ən",
    "te": "tə",
    "je": "jə",
    "ze": "zə",
    "we": "ʋə",
    "me": "mə",
    "mijn": "mɛin",
    "zijn": "zɛin",
    "hij": "ɦɛi",
    "wij": "ʋɛi",
    "jij": "jɛi",
}


def normalize_grapheme_key(word: str) -> str:
    """Lowercase NFC key for Dutch letters (ij ligature → *ij*)."""
    t = unicodedata.normalize("NFC", word.strip().lower())
    t = t.replace("\u0133", "ij")  # ĳ
    return re.sub(r"[^a-záéíóúàèêëïöü-]+", "", t)


_VOWEL_PLAIN = frozenset("aeiouy")
_ACCENT_VOWELS = frozenset("áéíóúàèêëïöü")


def _is_vowel_letter(c: str) -> bool:
    return c in _VOWEL_PLAIN or c in _ACCENT_VOWELS


def _strip_to_plain_vowel(c: str) -> str:
    return {
        "á": "a",
        "à": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ï": "i",
        "ó": "o",
        "ö": "o",
        "ú": "u",
        "ü": "u",
    }.get(c, c)


def _word_has_written_stress(word: str) -> bool:
    return any(c in "áéíóú" for c in word)


def _stressed_syllable_from_acute(word: str, syllables: list[str]) -> int | None:
    """Index of syllable containing á/é/í/ó/ú, else None."""
    for i, syl in enumerate(syllables):
        if any(c in "áéíóú" for c in syl):
            return i
    return None


# Longest-first nucleus tags for syllabification (inclusive start, exclusive end).
_NUCLEUS_PATTERNS: tuple[tuple[str, int], ...] = tuple(
    sorted(
        (
            ("aai", 3),
            ("eeu", 3),
            ("oei", 3),
            ("ieu", 3),
            ("ij", 2),
            ("ei", 2),
            ("au", 2),
            ("ou", 2),
            ("ui", 2),
            ("eu", 2),
            ("aa", 2),
            ("ee", 2),
            ("oo", 2),
            ("uu", 2),
            ("oe", 2),
            ("ai", 2),
            ("ie", 2),
        ),
        key=lambda x: len(x[0]),
        reverse=True,
    )
)


def _dutch_vowel_nucleus_spans(w: str) -> list[tuple[int, int]]:
    """Inclusive-exclusive spans of vowel nuclei (digraphs / trigraphs as one)."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(w)
    while i < n:
        if w[i] == "-":
            i += 1
            continue
        if not _is_vowel_letter(w[i]):
            i += 1
            continue
        matched = False
        for pat, ln in _NUCLEUS_PATTERNS:
            if i + ln <= n and w[i : i + ln] == pat:
                spans.append((i, i + ln))
                i += ln
                matched = True
                break
        if matched:
            continue
        spans.append((i, i + 1))
        i += 1
    return spans


def dutch_orthographic_syllables(word: str) -> list[str]:
    """
    Rough orthographic syllables (maximal onset: consonants between nuclei start the next syllable).
    Hyphen splits compounds before syllabifying each piece.
    """
    w = normalize_grapheme_key(word)
    w = re.sub(r"-+", "-", w).strip("-")
    if not w:
        return []
    if "-" in w:
        parts: list[str] = []
        for chunk in w.split("-"):
            if chunk:
                parts.extend(dutch_orthographic_syllables(chunk))
        return parts
    spans = _dutch_vowel_nucleus_spans(w)
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
            syllables.append(cur)
            cur = cluster
        else:
            syllables.append(cur + w[e:])
    return [s for s in syllables if s]


def _unstressed_prefix_len(w: str) -> int:
    for p in (
        "tegen",
        "tussen",
        "door",
        "voor",
        "ver",
        "her",
        "ont",
        "in",
        "op",
        "af",
        "uit",
        "aan",
        "be",
        "ge",
        "er",
        "te",
    ):
        if w.startswith(p) and len(w) > len(p) and w[len(p)] not in "-":
            return len(p)
    return 0


def _default_stress_syllable_index(syls: list[str], w: str) -> int:
    if not syls:
        return 0
    n = len(syls)
    if n == 1:
        return 0
    if _word_has_written_stress(w):
        idx = _stressed_syllable_from_acute(w, syls)
        if idx is not None:
            return idx
    wl = w.replace("-", "")
    for suf in ("atie", "iteit", "isme", "eerd", "eren"):
        if wl.endswith(suf) and len(wl) > len(suf) + 1:
            return n - 1
    plen = _unstressed_prefix_len(wl)
    if plen > 0:
        # *geiten*-style: first orthographic syllable is *gei* / *geu* …, not separable *ge* + *i*.
        first = syls[0].lower()
        if first.startswith("ge") and len(first) > 2:
            return 0
        acc = 0
        for idx, sy in enumerate(syls):
            acc += len(sy)
            if acc >= plen:
                return min(idx + 1, n - 1)
    return 0


def _insert_primary_stress_before_vowel(ipa: str) -> str:
    s = ipa.replace("ˈ", "")
    m = re.search(
        r"ɛi|ʌu|ʌy|øː|aɪ̯|iː|eː|aː|oː|uː|yː|ɪ|ʏ|y|ø|[aɛəioɔuɑ]",
        s,
    )
    if not m:
        return "ˈ" + s
    return s[: m.start()] + "ˈ" + s[m.start() :]


# Longest first: nuclei for vocoder stress placement (overlap with German where possible).
_IPA_NUCLEUS_PREFIXES: tuple[str, ...] = tuple(
    sorted(
        (
            "ɛi",
            "ʌu",
            "ʌy",
            "aɪ̯",
            "iː",
            "eː",
            "aː",
            "oː",
            "uː",
            "yː",
            "øː",
            "ŋ̩",
            "n̩",
            "m̩",
            "l̩",
            "ə",
            "ɛ",
            "ɪ",
            "ʏ",
            "y",
            "ø",
            "ɔ",
            "ɑ",
            "a",
            "i",
            "e",
            "o",
            "u",
        ),
        key=len,
        reverse=True,
    )
)

_IPA_CONS_CLUSTER_PREFIXES: tuple[str, ...] = tuple(
    sorted(
        (
            "t͡ʃ",
            "tʃ",
            "t͡s",
        ),
        key=len,
        reverse=True,
    )
)


def _ipa_starts_with_nucleus(s: str, j: int) -> bool:
    if j >= len(s):
        return False
    rest = s[j:]
    return any(rest.startswith(p) for p in _IPA_NUCLEUS_PREFIXES)


def _ipa_skip_pre_nucleus(s: str, j: int) -> int:
    if j >= len(s) or s[j] in (_PRIMARY_STRESS, _SECONDARY_STRESS):
        return j
    if _ipa_starts_with_nucleus(s, j):
        return j
    for p in _IPA_CONS_CLUSTER_PREFIXES:
        if s.startswith(p, j):
            return j + len(p)
    return j + 1


def normalize_ipa_stress_for_vocoder(ipa: str) -> str:
    """
    Move ˈ and ˌ to immediately before the next syllable nucleus (vowel / syllabic consonant).
    Idempotent on already nuclear placement.
    """
    if not ipa or (_PRIMARY_STRESS not in ipa and _SECONDARY_STRESS not in ipa):
        return ipa
    out: list[str] = []
    i = 0
    n = len(ipa)
    while i < n:
        ch = ipa[i]
        if ch not in (_PRIMARY_STRESS, _SECONDARY_STRESS):
            j = i
            while j < n and ipa[j] not in (_PRIMARY_STRESS, _SECONDARY_STRESS):
                j += 1
            out.append(ipa[i:j])
            i = j
            continue
        mark = ch
        i += 1
        j = i
        while j < n and ipa[j] not in (_PRIMARY_STRESS, _SECONDARY_STRESS) and not _ipa_starts_with_nucleus(
            ipa, j
        ):
            j2 = _ipa_skip_pre_nucleus(ipa, j)
            if j2 == j:
                break
            j = j2
        out.append(ipa[i:j])
        out.append(mark)
        i = j
    return "".join(out)


def _final_devoice_obstruents(ipa: str) -> str:
    if not ipa:
        return ipa
    repl = {"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s", "ɣ": "x", "ʒ": "ʃ"}
    last = ipa[-1]
    return ipa[:-1] + repl.get(last, last)


def _letters_to_ipa_no_stress(
    letters: str,
    *,
    full_word: str,
    hyphen_word: str,
    span_start: int,
) -> str:
    """
    Map *letters* (one syllable chunk, lowercase with accents) to IPA without stress.
    *full_word* has no hyphens; *hyphen_word* retains ``-``; *span_start* indexes *letters[0]* in *full_word*.
    """
    s = letters.lower()
    n = len(s)
    i = 0
    out: list[str] = []

    while i < n:
        if s[i] == "-":
            i += 1
            continue

        ch = s[i]

        if i + 2 < n and s[i : i + 3] == "sch":
            out.append("sx")
            i += 3
            continue

        if i + 1 < n and s[i : i + 2] == "ch":
            out.append("x")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "ng":
            out.append("ŋ")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "nk":
            out.append("ŋk")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "sj":
            out.append("ʃ")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "tj":
            out.append("tʃ")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "ij":
            out.append("ɛi")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "ei":
            out.append("ɛi")
            i += 2
            continue

        if i + 2 < n and s[i : i + 3] == "aai":
            out.append("aːi")
            i += 3
            continue

        if i + 2 < n and s[i : i + 3] == "eeu":
            out.append("eːʏ")
            i += 3
            continue

        if i + 2 < n and s[i : i + 3] == "oei":
            out.append("ʌi")
            i += 3
            continue

        if i + 2 < n and s[i : i + 3] == "ieu":
            out.append("ʌu")
            i += 3
            continue

        if i + 1 < n and s[i : i + 2] in {"au", "ou"}:
            out.append("ʌu")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "ui":
            out.append("ʌy")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "eu":
            out.append("øː")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "oe":
            out.append("u")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "ai":
            out.append("aɪ̯")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "aa":
            out.append("aː")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "ee":
            out.append("eː")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "oo":
            out.append("oː")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "uu":
            out.append("y")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "ie":
            nxt = s[i + 2] if i + 2 < n else ""
            if nxt in "tsd" and (i + 3 >= n or not _is_vowel_letter(s[i + 3])):
                out.append("i")
            else:
                out.append("iː")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "qu":
            out.append("kv")
            i += 2
            continue

        if ch == "h":
            out.append("ɦ")
            i += 1
            continue

        if ch == "x":
            out.append("ks")
            i += 1
            continue

        if ch == "c" and i + 1 < n and s[i + 1] in "eiéèêë":
            # ⟨c⟩ is /s/; the following vowel letter is still pronounced (e.g. *cent* /sɛnt/).
            out.append("s")
            i += 1
            continue

        if ch == "c":
            out.append("k")
            i += 1
            continue

        if ch == "q" and (i + 1 >= n or s[i + 1] != "u"):
            out.append("k")
            i += 1
            continue

        if ch == "j":
            out.append("j")
            i += 1
            continue

        if ch == "y":
            pv = i > 0 and _is_vowel_letter(s[i - 1])
            nv = i + 1 < n and _is_vowel_letter(s[i + 1])
            if not pv and nv:
                out.append("j")
            else:
                out.append("i")
            i += 1
            continue

        if ch == "w":
            out.append("ʋ")
            i += 1
            continue

        if ch == "v":
            out.append("v")
            i += 1
            continue

        if ch == "z":
            out.append("z")
            i += 1
            continue

        if ch == "g":
            out.append("ɣ")
            i += 1
            continue

        if _is_vowel_letter(ch):
            plain = _strip_to_plain_vowel(ch)
            if ch == "é":
                out.append("eː")
            elif ch in "èê":
                out.append("ɛ")
            elif ch == "ë":
                out.append("ə")
            elif ch == "ï" or ch == "ü":
                out.append("y")
            elif ch == "ö":
                out.append("ø")
            elif plain == "a":
                out.append("ɑ")
            elif plain == "e":
                out.append("ə" if i == n - 1 else "ɛ")
            elif plain == "i":
                out.append("ɪ")
            elif plain == "o":
                out.append("ɔ")
            elif plain == "u":
                out.append("ʏ")
            else:
                out.append("i")
            i += 1
            continue

        if ch == "r":
            out.append("r")
            i += 1
            continue

        if ch == "s" and i + 1 < n and s[i + 1] == "s":
            out.append("s")
            i += 2
            continue

        if ch == "s":
            # Same syllable only (mirrors ``german_rule_g2p``); avoids *Amsterdam*-style /s/→/z/.
            prev_v = i > 0 and _is_vowel_letter(s[i - 1])
            next_v = i + 1 < n and _is_vowel_letter(s[i + 1])
            out.append("z" if prev_v and next_v else "s")
            i += 1
            continue

        simple = {
            "b": "b",
            "d": "d",
            "f": "f",
            "k": "k",
            "l": "l",
            "m": "m",
            "n": "n",
            "p": "p",
            "t": "t",
        }
        if ch in simple:
            out.append(simple[ch])
            i += 1
            continue

        if ch == "p" and i + 1 < n and s[i + 1] == "h":
            out.append("f")
            i += 2
            continue

        if ch == "t" and i + 1 < n and s[i + 1] == "h":
            out.append("t")
            i += 2
            continue

        i += 1

    ipa = "".join(out)
    stem = letters.replace("-", "")
    if stem.endswith("ig") and len(stem) >= 3 and not stem.endswith("lijk"):
        if ipa.endswith("ɣ"):
            ipa = ipa[:-1] + "x"
        elif ipa.endswith("ɡ"):
            ipa = ipa[:-1] + "x"
    return _final_devoice_obstruents(ipa)


def _rules_word_to_ipa(word: str, *, with_stress: bool) -> str:
    w = normalize_grapheme_key(word)
    if not w:
        return ""
    wl = w
    wl_nh = wl.replace("-", "")
    syls = dutch_orthographic_syllables(wl)
    if not syls:
        return ""
    stress_idx = _default_stress_syllable_index(syls, wl) if with_stress else -1
    offset = 0
    parts: list[str] = []
    for idx, sy in enumerate(syls):
        chunk = _letters_to_ipa_no_stress(
            sy,
            full_word=wl_nh,
            hyphen_word=wl,
            span_start=offset,
        )
        if with_stress and idx == stress_idx and chunk:
            chunk = _insert_primary_stress_before_vowel(chunk)
        parts.append(chunk)
        offset += len(sy)
    return "".join(parts)


def _finalize_word_ipa(
    ipa: str,
    *,
    with_stress: bool,
    vocoder_stress: bool,
    from_lexicon: bool = False,
) -> str:
    if not with_stress:
        return ipa.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
    # ipa-dict stress is dictionary-style; nuclear shifting breaks both dotted entries
    # (*bourgeoisie*) and short lemmas (*goed*). Keep TSV stress for all lexicon hits.
    if vocoder_stress and not from_lexicon:
        return normalize_ipa_stress_for_vocoder(ipa)
    return ipa


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
    Single-token G2P: lexicon lookup (normalized key), else a small function-word map, else rules.

    If *vocoder_stress* is True (default), :func:`normalize_ipa_stress_for_vocoder` is applied
    only to **rule-based** IPA (lexicon strings keep ipa-dict stress).

    Pure digit strings are expanded via :func:`dutch_numbers.expand_cardinal_digits_to_dutch_words`
    when *expand_cardinal_digits* is True (ranges like ``1933-1945`` are expanded in
    :func:`text_to_ipa` before tokenization).
    """
    if not word or not word.strip():
        return ""
    raw = word.strip()

    if expand_cardinal_digits and raw.isdigit():
        phrase = expand_cardinal_digits_to_dutch_words(raw)
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

    # German surname: lexicon *speer* is the noun “spear” (/speːr/); the person’s name is /spɪːr/
    # (eSpeak-ng ``nl``).
    if raw == "Speer":
        return _finalize_word_ipa(
            "spˈɪːr",
            with_stress=with_stress,
            vocoder_stress=vocoder_stress,
            from_lexicon=False,
        )

    letters_only = normalize_lexicon_key(raw)
    if not letters_only:
        return ""

    lex = lexicon if lexicon is not None else _get_lexicon(dict_path)
    ipa = _lookup_lexicon(letters_only, lex)
    if ipa is None:
        ipa = _OOV_SUPPLEMENT_IPA.get(letters_only)
    if ipa is not None:
        ipa = _apply_lexicon_ipa_postprocess(ipa, letters_only)
        return _finalize_word_ipa(
            ipa,
            with_stress=with_stress,
            vocoder_stress=vocoder_stress,
            from_lexicon=True,
        )

    if "-" in letters_only:
        chunks = [c for c in letters_only.split("-") if c]
        sub_raw = [_lookup_lexicon(c, lex) for c in chunks]
        if chunks and all(sub_raw):
            sub = [_apply_lexicon_ipa_postprocess(ipa_i, ck) for ipa_i, ck in zip(sub_raw, chunks)]
            merged = "-".join(sub)
            return _finalize_word_ipa(
                merged,
                with_stress=with_stress,
                vocoder_stress=vocoder_stress,
                from_lexicon=True,
            )

    if letters_only in _FUNCTION_WORD_IPA:
        ipa = _FUNCTION_WORD_IPA[letters_only]
        return _finalize_word_ipa(
            ipa,
            with_stress=with_stress,
            vocoder_stress=vocoder_stress,
            from_lexicon=False,
        )

    ipa = _rules_word_to_ipa(raw, with_stress=with_stress)
    return _finalize_word_ipa(
        ipa,
        with_stress=with_stress,
        vocoder_stress=vocoder_stress,
        from_lexicon=False,
    )


_TOKEN_RE = re.compile(r"[\w\-]+|[^\w\s\-]+|\s+", flags=re.UNICODE)


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
        elif re.fullmatch(r"[\w\-]+", tok, flags=re.UNICODE):
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
        description="Dutch text to IPA using data/nl/dict.tsv plus rules for OOV."
    )
    p.add_argument("text", nargs="*", help="Dutch text (if empty, read stdin)")
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
        help="Keep ˈ before the first segment of the stressed syllable; default moves it before the nucleus.",
    )
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin.")
    p.add_argument(
        "--no-expand-digits",
        action="store_true",
        help="Leave digit sequences as digits (no spoken Dutch cardinal expansion).",
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
    lex = load_dutch_lexicon(args.dict) if args.dict is not None else None
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
