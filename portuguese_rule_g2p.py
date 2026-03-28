#!/usr/bin/env python3
"""
Rule- and lexicon-based Portuguese grapheme-to-phoneme (broad IPA).

* Brazilian Portuguese uses ``data/pt_br/dict.tsv`` (ipa-dict); European Portuguese uses
  ``data/pt_pt/dict.tsv`` (WikiPron-style entries). Run
  ``scripts/download_multilingual_ipa_lexicons.py`` to fetch them.
* In-vocabulary words are looked up with the same duplicate-key policy as
  :mod:`german_rule_g2p` / :mod:`italian_rule_g2p` (all-lowercase surface rows override
  capitalized homographs).
* Out-of-vocabulary tokens use orthographic syllables, a simplified Portuguese stress rule,
  and a letter pass with a small ``pt_br`` vs ``pt_pt`` preset (chiefly unstressed vowels
  and intervocalic ⟨s⟩).

Lexicon entries in ipa-dict use ASCII periods as syllable boundaries (e.g. ``as.tɾo.no.ˈmi.ə``).
By default those dots are **stripped** so output matches the continuous IPA style of eSpeak NG;
pass ``keep_syllable_dots=True`` (or CLI ``--keep-syllable-dots``) to preserve the TSV form.
Rule-based IPA can be passed through :func:`german_rule_g2p.normalize_ipa_stress_for_vocoder`
(eSpeak-style nuclear stress), matching :mod:`italian_rule_g2p`.

Limitations (intentional):
- Nasalization before coda *n/m* (e.g. *bom*, *bem*) is not fully spelled out in the rules path.
- ⟨x⟩ is only partly disambiguated; the lexicon covers high-frequency lemmas.
- Default stress heuristics miss some patterns (e.g. some *-mente* forms); prefer the lexicon.
- Brazilian palatalization of ⟨d⟩/⟨t⟩ before ⟨i⟩ (*dividiu*) is not modeled in the letter pass; the lexicon usually wins.
- When ⟨s⟩+⟨c⟩ falls across orthographic syllables (*des-cer*, *pis-cina*), use :data:`_OOV_SC_STRADDLE` or the lexicon.
- European Portuguese (``pt_pt``) rule path maps common plural word-final ⟨…V+s⟩ to ``…Vʃ`` (see :func:`_pt_pt_apply_rules_final_s_to_esh`); disable with ``apply_pt_pt_final_esh=False``.

CLI: prints G2P IPA, then an eSpeak NG reference line by default (``--no-espeak`` to disable),
same stack as ``german_rule_g2p`` / ``spanish_rule_g2p``.

Wiki verification: ``scripts/verify_portuguese_g2p_wiki.py`` reads ``data/{pt_br,pt_pt}/wiki-text.txt``
(this repository exports wiki samples as ``.txt``, not ``.tsv``).

Digit-only tokens (and ``1933-1945``-style ranges) expand to Portuguese cardinals via
:mod:`portuguese_numbers` before G2P (up to 999_999); disable with ``expand_cardinal_digits=False``
or CLI ``--no-expand-digits``.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

from german_rule_g2p import normalize_ipa_stress_for_vocoder
from portuguese_numbers import expand_cardinal_digits_to_portuguese_words, expand_digit_tokens_in_text

_DEFAULT_ESPEAK_VOICE_BR = "pt-br"
_DEFAULT_ESPEAK_VOICE_PT = "pt"

_REPO_ROOT = Path(__file__).resolve().parent

_LEXICON_CACHES: dict[tuple[str, Path], dict[str, str]] = {}

# When ``expand_cardinal_digits`` is off: pass digit / simple year-range tokens through unchanged.
_DIGIT_PASS_THROUGH_RE = re.compile(r"^[0-9]+(?:-[0-9]+)*$")

_PRIMARY_STRESS = "\u02c8"  # ˈ
_SECONDARY_STRESS = "\u02cc"  # ˌ


def espeak_ng_ipa_line(text: str, *, voice: str) -> str | None:
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


def default_dict_path(variant: str) -> Path:
    """``pt_br`` / ``pt_pt`` → default ``dict.tsv`` under ``data/``."""
    v = variant.strip().lower().replace("-", "_")
    if v not in ("pt_br", "pt_pt"):
        raise ValueError("variant must be 'pt_br' or 'pt_pt'")
    return _REPO_ROOT / "data" / v / "dict.tsv"


def default_espeak_voice(variant: str) -> str:
    if variant == "pt_pt":
        return _DEFAULT_ESPEAK_VOICE_PT
    return _DEFAULT_ESPEAK_VOICE_BR


def normalize_lookup_key(word: str) -> str:
    """Lowercase NFC key for TSV lookup (Portuguese letters, apostrophe, hyphen)."""
    t = unicodedata.normalize("NFC", word.strip().lower())
    t = t.replace("\u2019", "'")
    return re.sub(r"[^a-záàâãçéêíóôõúü'`\-]+", "", t)


def load_portuguese_lexicon(path: Path | None = None, *, variant: str = "pt_br") -> dict[str, str]:
    """
    Load ``word\\tIPA`` TSV. Duplicate keys: prefer an all-lowercase surface row over a
    capitalized one (homograph policy).
    """
    p = path or default_dict_path(variant)
    if not p.is_file():
        raise FileNotFoundError(f"Portuguese lexicon not found: {p}")
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


def _get_lexicon(variant: str, path: Path | None) -> dict[str, str]:
    p = path or default_dict_path(variant)
    key = (variant, p.resolve())
    if key in _LEXICON_CACHES:
        return _LEXICON_CACHES[key]
    if not p.is_file():
        _LEXICON_CACHES[key] = {}
        return _LEXICON_CACHES[key]
    _LEXICON_CACHES[key] = load_portuguese_lexicon(p, variant=variant)
    return _LEXICON_CACHES[key]


def _lookup_lexicon(key: str, lexicon: Mapping[str, str]) -> str | None:
    return lexicon[key] if key in lexicon else None


# Common clitics / function words where syllable heuristics are weak (broad IPA hints).
_FUNCTION_WORD_IPA_BR: dict[str, str] = {
    "a": "ɐ",
    "o": "u",
    "os": "ʊs",
    "as": "ɐs",
    "e": "i",
    "ou": "ow",
    "em": "ɐ̃j̃",
    "no": "nʊ",
    "na": "nɐ",
    "nos": "nʊs",
    "nas": "nɐs",
    "de": "dʒɪ",
    "do": "dʊ",
    "da": "dɐ",
    "dos": "dʊs",
    "das": "dɐs",
    "dum": "dũ",
    "duma": "ˈdumɐ",
    "num": "nũ",
    "numa": "ˈnumɐ",
    "pelo": "ˈpɛlʊ",
    "pela": "ˈpɛlɐ",
    "pelos": "ˈpɛlʊs",
    "pelas": "ˈpɛlɐs",
    "com": "kõ",
    "sem": "sɐ̃j̃",
    "por": "poɾ",
    "para": "ˈpaɾɐ",
    "que": "ki",
    "não": "ˈnɐ̃w̃",
    "um": "ũ",
    "uma": "ˈumɐ",
    "uns": "ũs",
    "umas": "ˈumɐs",
    "ao": "aw",
    "aos": "awʃ",
    "à": "a",
    "às": "ɐʃ",
}

_FUNCTION_WORD_IPA_PT: dict[str, str] = {
    "a": "ɐ",
    "o": "u",
    "os": "uʃ",
    "as": "ɐʃ",
    "e": "ɨ",
    "ou": "ow",
    "em": "ɐ̃j̃",
    "no": "nu",
    "na": "nɐ",
    "nos": "nuʃ",
    "nas": "nɐʃ",
    "de": "dɨ",
    "do": "du",
    "da": "dɐ",
    "dos": "duʃ",
    "das": "dɐʃ",
    "dum": "dũ",
    "duma": "ˈdumɐ",
    "num": "nũ",
    "numa": "ˈnumɐ",
    "pelo": "ˈpɛlu",
    "pela": "ˈpɛlɐ",
    "pelos": "ˈpɛluʃ",
    "pelas": "ˈpɛlɐʃ",
    "com": "kõ",
    "sem": "sɐ̃j̃",
    "por": "puɾ",
    "para": "ˈpɐɾɐ",
    "que": "kɨ",
    "não": "ˈnɐ̃w̃",
    "um": "ũ",
    "uma": "ˈumɐ",
    "uns": "ũʃ",
    "umas": "ˈumɐʃ",
    "ao": "aw",
    "aos": "awʃ",
    "à": "a",
    "às": "aʃ",
}


def _function_words_for(variant: str) -> dict[str, str]:
    return _FUNCTION_WORD_IPA_PT if variant == "pt_pt" else _FUNCTION_WORD_IPA_BR


_VOWELS_PT = frozenset("aeiouáàâãéêíóôõúüý")


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _strip_accents(ch: str) -> str:
    return "".join(c for c in _nfd(ch) if unicodedata.category(c) != "Mn") or ch


def _is_vowel_pt(ch: str) -> bool:
    return ch.lower() in _VOWELS_PT


def _should_hiatus_pt(a: str, b: str) -> bool:
    """Hiatus vs diphthong between adjacent vowel letters (Portuguese orthography, heuristic)."""
    al, bl = a.lower(), b.lower()
    if al in "íúý" or bl in "íúý":
        return True
    ba, bb = _strip_accents(al), _strip_accents(bl)
    if ba == bb:
        return True
    # Nasal vowel letters still participate in hiatus with another strong vowel.
    if al in "ãõ" or bl in "ãõ":
        if {ba, bb} <= {"a", "e", "i", "o", "u"} and (ba in "aeo" and bb in "aeo"):
            return True
        return False
    sa, sb = ba in "aeo", bb in "aeo"
    if sa and sb:
        if al in "áéóâêô" or bl in "áéóâêô":
            return True
        if ba + bb in {"ae", "ea"}:
            return False
        return True
    return False


def _vowel_nucleus_spans(w: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    n = len(w)
    while i < n:
        ch = w[i]
        if not _is_vowel_pt(ch):
            i += 1
            continue
        # ⟨ão⟩ / ⟨ãe⟩ are single nuclei in Portuguese orthography.
        if ch == "ã" and i + 1 < n and w[i + 1] == "o":
            out.append((i, i + 2))
            i += 2
            continue
        if ch == "ã" and i + 1 < n and w[i + 1] == "e":
            out.append((i, i + 2))
            i += 2
            continue
        if i + 1 < n and _is_vowel_pt(w[i + 1]):
            if _should_hiatus_pt(ch, w[i + 1]):
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
    if cluster == "rr":
        return "", "rr"
    if len(cluster) >= 2 and cluster[-2:] == "lh":
        return cluster[:-2], cluster[-2:]
    if len(cluster) >= 2 and cluster[-2:] == "nh":
        return cluster[:-2], cluster[-2:]
    if len(cluster) >= 2 and cluster[-2:] in _VALID_ONSETS_2:
        return cluster[:-2], cluster[-2:]
    return cluster[:-1], cluster[-1:]


def portuguese_orthographic_syllables(word: str) -> list[str]:
    """Rough orthographic syllables (onset maximization); hyphen splits compounds first."""
    w = re.sub(r"[^a-záàâãçéêíóôõúüýA-ZÁÀÂÃÇÉÊÍÓÔÕÚÜÝ\-]", "", word.lower())
    if not w:
        return []
    if "-" in w:
        parts: list[str] = []
        for chunk in w.split("-"):
            if chunk:
                parts.extend(portuguese_orthographic_syllables(chunk))
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


_ACCENTED = frozenset("áàâéêíóôú")


def default_stressed_syllable_index(syls: list[str], word_lower: str) -> int:
    """0-based stressed syllable index from Portuguese orthography (heuristic)."""
    if not syls:
        return 0
    w = re.sub(r"[^a-záàâãçéêíóôõúüý\-]", "", word_lower)
    for i, s in enumerate(syls):
        if any(c in _ACCENTED for c in s):
            return i
    n = len(syls)
    if n == 1:
        return 0
    if w.endswith("ões") or w.endswith("ãos") or w.endswith("ão"):
        return n - 1
    if w.endswith("ã") or w.endswith("ãs"):
        return n - 1
    last = w[-1]
    if last == "s" and len(w) >= 2:
        prev = w[-2]
        if prev in "aeiouáéíóúãõâêô":
            return max(0, n - 2)
    if last in "aeoáéó":
        return max(0, n - 2)
    if w.endswith("em") or w.endswith("ens") or w.endswith("am"):
        return max(0, n - 2)
    if last in "iuíú":
        return n - 1
    if last in "rlzx":
        return n - 1
    if last == "n" and not w.endswith("em"):
        return n - 1
    if last == "e":
        return max(0, n - 2)
    if last == "m":
        return max(0, n - 2)
    return max(0, n - 2)


def _insert_primary_stress_before_vowel(ipa: str) -> str:
    s = ipa.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
    # Walk codepoints; treat common IPA vowels (incl. ɐ ɨ) as nuclei.
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in "aeiouɛɔɐɪʊɨəæ":
            return s[:i] + _PRIMARY_STRESS + s[i:]
        if ch == "ɐ" or ch == "ɔ":
            return s[:i] + _PRIMARY_STRESS + s[i:]
        i += 1
    return _PRIMARY_STRESS + s


_X_EXCEPTIONS: dict[str, str] = {
    "táxi": "ˈtaksi",
    "taxi": "ˈtaksi",
    "máximo": "ˈmaksimu",
    "fênix": "ˈfɛniks",
    "fénix": "ˈfɛniks",
}

# Orthographic ⟨s⟩ + ⟨c⟩ straddles syllables (e.g. es|cola, pis|cina); rules work per syllable.
_OOV_SC_STRADDLE: dict[str, str] = {
    "escola": "ɪskˈɔlɐ",
    "piscina": "piʃˈkinɐ",
    "descer": "dɪʃˈseɾ",
}

# Singular (or fixed) lemmas; do not map word-final ⟨s⟩ → /ʃ/ for European Portuguese rules.
_PT_PT_FINAL_S_EXCLUDE: frozenset[str] = frozenset(
    {
        "caos",
        "cosmos",
        "bônus",
        "vírus",
        "lápis",
        "país",
        "cais",
        "mês",
        "três",
        "inglês",
        "francês",
        "português",
        "anís",
        "fénix",
        "tórax",
    }
)


def _roman_to_int(s: str) -> int | None:
    """Parse a Roman numeral (ASCII I V X L C D M) or return None."""
    u = s.upper().strip()
    if not u or not re.fullmatch(r"[IVXLCDM]+", u):
        return None
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    i = 0
    n = len(u)
    while i < n:
        v = vals.get(u[i])
        if v is None:
            return None
        if i + 1 < n:
            nv = vals.get(u[i + 1])
            if nv is not None and nv > v:
                total += nv - v
                i += 2
                continue
        total += v
        i += 1
    return total if 0 < total < 4000 else None


# Broad IPA for spoken Portuguese cardinals (Wikipedia-style Roman numerals, e.g. século XX).
_CARDINAL_IPA_BY_NUMBER: dict[int, str] = {
    1: "ˈũ",
    2: "ˈdɔjs",
    3: "ˈtɾɛjs",
    4: "ˈkwatɾʊ",
    5: "ˈsĩkʊ",
    6: "ˈsejs",
    7: "ˈsɛtʃi",
    8: "ˈɔjtʊ",
    9: "ˈnɔvi",
    10: "ˈdɛjs",
    11: "ˈɔ̃zi",
    12: "ˈdɔzi",
    13: "ˈtɾɛzi",
    14: "kaɪˈɔɾzi",
    15: "ˈkĩzi",
    16: "dɛˈzesejs",
    17: "dɛˈzesɛtʃi",
    18: "dɛˈzejzj",
    19: "dɛzenˈɔvi",
    20: "ˈvĩtʃi",
    21: "vĩˈtʃiˈeũ",
    30: "ˈtɾĩtʃi",
    40: "kwɐˈɾẽtɐ",
    50: "ˈsĩkwẽtɐ",
    60: "ˈsessẽtʃi",
    70: "sɛˈtẽtʃi",
    80: "ˈojtẽtʃi",
    90: "ˈnɔvẽtʃi",
    100: "ˈsẽtʃi",
}


def _roman_numeral_token_to_ipa(wl: str, variant: str) -> str | None:
    """
    If *wl* is an all-Roman token (e.g. ``XX``), return broad IPA for the spoken cardinal.
    Covers 1–21, 30, 40, …, 100 used in encyclopedic Portuguese.
    """
    if "-" in wl or "'" in wl:
        return None
    n = _roman_to_int(wl)
    if n is None:
        return None
    ipa = _CARDINAL_IPA_BY_NUMBER.get(n)
    if ipa is None:
        return None
    if variant == "pt_pt":
        ipa = ipa.replace("ˈvĩtʃi", "ˈvĩtʃɨ")
    return ipa


def _prev_global_vowel(full_word: str, gidx: int) -> bool:
    j = gidx - 1
    while j >= 0:
        if _is_vowel_pt(full_word[j]):
            return True
        if full_word[j] == "-":
            break
        j -= 1
    return False


def _next_global_vowel(full_word: str, gidx: int) -> bool:
    j = gidx + 1
    n = len(full_word)
    while j < n:
        if _is_vowel_pt(full_word[j]):
            return True
        if full_word[j] == "-":
            break
        j += 1
    return False


def _letters_to_ipa_no_stress(
    s: str,
    *,
    variant: str,
    full_word: str,
    span_start: int,
    stressed_syllable: bool,
) -> str:
    """Map one syllable chunk to IPA (no stress mark)."""
    n = len(s)
    i = 0
    out: list[str] = []

    def unstressed_vowel(base: str) -> str:
        if stressed_syllable:
            return base
        if variant == "pt_pt":
            m = {
                "a": "ɐ",
                "e": "ɨ",
                "i": "i",
                "o": "u",
                "u": "u",
            }
            return m.get(base, base)
        m = {
            "a": "ɐ",
            "e": "ɪ",
            "i": "i",
            "o": "ʊ",
            "u": "u",
        }
        return m.get(base, base)

    def map_vowel_char(ch: str) -> str:
        cl = ch.lower()
        if cl in "áàâ":
            return "a"
        if cl == "é":
            return "ɛ"
        if cl == "ê":
            return "ɛ"
        if cl == "í":
            return "i"
        if cl == "ó":
            return "ɔ"
        if cl == "ô":
            return "ɔ"
        if cl == "ú":
            return "u"
        if cl == "ã":
            return "ɐ̃"
        if cl == "õ":
            return "o\u0303"
        if cl in "aeiou":
            if cl == "a":
                return "a" if stressed_syllable else unstressed_vowel("a")
            if cl == "e":
                return "ɛ" if stressed_syllable and "ê" in s else ("e" if stressed_syllable else unstressed_vowel("e"))
            if cl == "i":
                return unstressed_vowel("i")
            if cl == "o":
                return "ɔ" if stressed_syllable and "ô" in s else ("o" if stressed_syllable else unstressed_vowel("o"))
            if cl == "u":
                return unstressed_vowel("u")
        if cl == "ü":
            return "w"
        if cl in "ýy":
            return "i"
        return ""

    while i < n:
        if s[i] == "-":
            i += 1
            continue

        gi = span_start + i

        if s[i] == "ã" and i + 1 < n and s[i + 1] == "o":
            out.append("ɐ̃w̃")
            i += 2
            continue
        if s[i] == "ã" and i + 1 < n and s[i + 1] == "e":
            out.append("ɐ̃j̃")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "ch":
            out.append("ʃ")
            i += 2
            continue
        if i + 1 < n and s[i : i + 2] == "nh":
            out.append("ɲ")
            i += 2
            continue
        if i + 1 < n and s[i : i + 2] == "lh":
            out.append("ʎ")
            i += 2
            continue
        if i + 1 < n and s[i : i + 2] == "rr":
            out.append("ʁ")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "qu" and i + 2 < n and s[i + 2].lower() in "eéêií":
            out.append("k")
            i += 2
            continue
        if i + 1 < n and s[i : i + 2] == "gu" and i + 2 < n and s[i + 2].lower() in "eéêií":
            out.append("ɡ")
            i += 2
            continue
        if i + 1 < n and s[i : i + 2] == "qu":
            out.append("kw")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "ss":
            out.append("s")
            i += 2
            continue

        if s[i] == "ç":
            out.append("s")
            i += 1
            continue

        if (
            s[i] == "c"
            and i > 0
            and s[i - 1] == "s"
            and i + 1 < n
            and s[i + 1].lower() in "aáâeéêiíoóôuúãõ"
        ):
            v = s[i + 1].lower()
            if v in "eéêií":
                out.append("ʃ")
            else:
                out.append("sk")
            i += 1
            continue

        if s[i] == "c" and i + 1 < n and s[i + 1].lower() in "eéêií":
            out.append("s")
            i += 1
            continue
        if s[i] == "c":
            out.append("k")
            i += 1
            continue

        if s[i] == "g" and i + 1 < n and s[i + 1].lower() in "eéêií":
            out.append("ʒ")
            i += 1
            continue
        if s[i] == "g":
            out.append("ɡ")
            i += 1
            continue

        if s[i] == "x":
            if gi == 0 and i + 1 < n and s[i + 1].lower() in "eéií":
                out.append("ʒ")
                i += 2
                continue
            pv = _prev_global_vowel(full_word, gi)
            nv = _next_global_vowel(full_word, gi + 1)
            if pv and nv:
                out.append("ʒ" if variant == "pt_br" else "ʃ")
            else:
                out.append("ks")
            i += 1
            continue

        if s[i] == "h":
            i += 1
            continue

        if s[i] == "s":
            pv = gi > 0 and _prev_global_vowel(full_word, gi - 1)
            nv = i + 1 < n and _next_global_vowel(full_word, gi + 1)
            if pv and nv:
                if variant == "pt_br":
                    out.append("z")
                else:
                    out.append("ʒ")
            else:
                out.append("s")
            i += 1
            continue

        if s[i] == "z":
            out.append("z")
            i += 1
            continue

        if s[i] == "j":
            out.append("ʒ")
            i += 1
            continue

        if s[i] in "wW":
            out.append("w")
            i += 1
            continue

        if s[i] == "r":
            gidx = span_start + i
            at_word = gidx == 0
            prev_ch = full_word[gidx - 1] if gidx > 0 else ""
            after_cons = gidx > 0 and not _is_vowel_pt(prev_ch) and prev_ch != "'"
            intervocal = (
                gidx > 0
                and _is_vowel_pt(prev_ch)
                and gidx + 1 < len(full_word)
                and _is_vowel_pt(full_word[gidx + 1])
            )
            if at_word or after_cons or (i + 1 < n and s[i + 1] == "r"):
                out.append("ʁ")
            elif intervocal:
                out.append("ɾ")
            else:
                out.append("ɾ")
            i += 1
            continue

        ch = s[i]
        if _is_vowel_pt(ch):
            # Diphthongs (stressed syllable heuristics).
            if i + 1 < n and _is_vowel_pt(s[i + 1]) and not _should_hiatus_pt(ch, s[i + 1]):
                a, b = ch.lower(), s[i + 1].lower()
                if a in "aáàâ" and b in "ií":
                    out.append("aj")
                    i += 2
                    continue
                if a in "aáàâ" and b in "uú":
                    out.append("aw")
                    i += 2
                    continue
                if a in "eéê" and b in "ií":
                    out.append("ej")
                    i += 2
                    continue
                if a in "oóô" and b in "ií":
                    out.append("oj")
                    i += 2
                    continue
                if a in "eéê" and b in "uú":
                    out.append("ew")
                    i += 2
                    continue
                if a in "oóô" and b in "uú":
                    out.append("ow")
                    i += 2
                    continue
            seg = map_vowel_char(ch)
            if seg:
                out.append(seg)
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
            "t": "t",
            "v": "v",
            "k": "k",
        }
        if ch.lower() in simple:
            out.append(simple[ch.lower()])
            i += 1
            continue

        i += 1

    return "".join(out)


def _rules_word_to_ipa_single(wl: str, variant: str, *, with_stress: bool) -> str:
    """
    Rule G2P for one hyphen-free Portuguese word (lowercase letters + accents).
    *wl* must not contain ``-`` so *span_start* / *full_word* indices stay aligned.
    """
    syls = portuguese_orthographic_syllables(wl)
    if not syls:
        return ""
    stress_idx = default_stressed_syllable_index(syls, wl) if with_stress else -1
    offset = 0
    parts: list[str] = []
    for idx, sy in enumerate(syls):
        chunk = _letters_to_ipa_no_stress(
            sy,
            variant=variant,
            full_word=wl,
            span_start=offset,
            stressed_syllable=(idx == stress_idx),
        )
        if with_stress and idx == stress_idx and chunk:
            chunk = _insert_primary_stress_before_vowel(chunk)
        parts.append(chunk)
        offset += len(sy)
    return "".join(parts)


def _rules_word_to_ipa(word: str, variant: str, *, with_stress: bool) -> str:
    w = re.sub(
        r"[^a-záàâãçéêíóôõúüýA-ZÁÀÂÃÇÉÊÍÓÔÕÚÜÝ\-]",
        "",
        word.strip(),
    )
    if not w:
        return ""
    wl = w.lower()
    wkey = normalize_lookup_key(wl)
    if wkey in _X_EXCEPTIONS:
        ipa_x = _X_EXCEPTIONS[wkey]
        if not with_stress:
            return ipa_x.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
        return ipa_x

    if wkey in _OOV_SC_STRADDLE:
        ipa_sc = _OOV_SC_STRADDLE[wkey]
        if not with_stress:
            return ipa_sc.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
        return ipa_sc

    if "-" in wl:
        chunks = [c for c in wl.split("-") if c]
        if len(chunks) > 1:
            return "-".join(_rules_word_to_ipa_single(c, variant, with_stress=with_stress) for c in chunks)

    return _rules_word_to_ipa_single(wl, variant, with_stress=with_stress)


_VOWEL_GRAPHEME_PT = frozenset("aeiouáàâãéêíóôõúü")


def _pt_pt_apply_rules_final_s_to_esh(ipa: str, letters_key: str) -> str:
    """
    European Portuguese: word-final /s/ → [ʃ] after a vowel for typical plural endings
    (⟨as⟩, ⟨es⟩, ⟨os⟩), aligning better with eSpeak ``pt`` and spoken EP.

    Lexicon and hand-mapped OOV strings are not passed through here.
    """
    if not ipa or not letters_key.endswith("s") or letters_key.endswith("ss"):
        return ipa
    if len(letters_key) < 4:
        return ipa
    lk = letters_key.lower()
    if lk in _PT_PT_FINAL_S_EXCLUDE:
        return ipa
    if lk.endswith(("ês", "ás", "ís", "ús")):
        return ipa
    if not (lk.endswith("as") or lk.endswith("os") or lk.endswith("es")):
        return ipa
    pen = lk[-2]
    if pen not in _VOWEL_GRAPHEME_PT:
        return ipa
    if not ipa.endswith("s"):
        return ipa
    return ipa[:-1] + "ʃ"


def _finalize_word_ipa(
    ipa: str,
    *,
    with_stress: bool,
    vocoder_stress: bool,
    from_lexicon: bool,
    keep_syllable_dots: bool,
) -> str:
    if not with_stress:
        out = ipa.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
        if not keep_syllable_dots:
            out = out.replace(".", "")
        return out
    out = ipa
    if vocoder_stress and not from_lexicon:
        out = normalize_ipa_stress_for_vocoder(out)
    if not keep_syllable_dots:
        out = out.replace(".", "")
    return out


_APOSTROPHE_IN_CLASS = "\u0027\u2019"
_PORT_WORD_PATTERN = (
    rf"(?:[{_APOSTROPHE_IN_CLASS}]?[\w\-]+(?:[{_APOSTROPHE_IN_CLASS}][\w\-]+)*(?:[{_APOSTROPHE_IN_CLASS}])?)"
)
_TOKEN_RE = re.compile(
    rf"{_PORT_WORD_PATTERN}|[^\w\s{_APOSTROPHE_IN_CLASS}\-]+|[{_APOSTROPHE_IN_CLASS}]|\s+",
    flags=re.UNICODE,
)
_PORT_WORD_FULLMATCH = re.compile(rf"{_PORT_WORD_PATTERN}\Z", flags=re.UNICODE)


def word_to_ipa(
    word: str,
    *,
    variant: str = "pt_br",
    lexicon: Mapping[str, str] | None = None,
    dict_path: Path | None = None,
    with_stress: bool = True,
    vocoder_stress: bool = True,
    keep_syllable_dots: bool = False,
    apply_pt_pt_final_esh: bool = True,
    expand_cardinal_digits: bool = True,
) -> str:
    """
    Single-token G2P: lexicon lookup (normalized key), optional hyphen merge, function-word map,
    else rules. *variant* is ``pt_br`` or ``pt_pt``.

    *keep_syllable_dots*: when False (default), remove ``.`` syllable-boundary markers from
    lexicon IPA (ipa-dict style) so output is one continuous string like eSpeak NG.

    *apply_pt_pt_final_esh*: for ``pt_pt`` rule-based IPA only, map word-final ``...Vs`` to
    ``...Vʃ`` when the ending matches common plural patterns (see :func:`_pt_pt_apply_rules_final_s_to_esh`).

    Pure digit strings expand via :func:`portuguese_numbers.expand_cardinal_digits_to_portuguese_words`
    when *expand_cardinal_digits* is True (ranges like ``1933-1945`` are expanded in
    :func:`text_to_ipa` before tokenization).
    """
    if not word or not word.strip():
        return ""
    raw = word.strip()
    v = variant.strip().lower().replace("-", "_")
    if v not in ("pt_br", "pt_pt"):
        raise ValueError("variant must be 'pt_br' or 'pt_pt'")

    if expand_cardinal_digits and raw.isdigit():
        phrase = expand_cardinal_digits_to_portuguese_words(raw, variant=v)
        if phrase != raw:
            return text_to_ipa(
                phrase,
                variant=v,
                lexicon=lexicon,
                dict_path=dict_path,
                with_stress=with_stress,
                vocoder_stress=vocoder_stress,
                keep_syllable_dots=keep_syllable_dots,
                apply_pt_pt_final_esh=apply_pt_pt_final_esh,
                expand_cardinal_digits=False,
            )
        return raw  # e.g. integer > 999_999: leave digits unchanged

    if not expand_cardinal_digits and _DIGIT_PASS_THROUGH_RE.fullmatch(raw):
        return raw

    letters_only = normalize_lookup_key(raw)
    if not letters_only:
        return ""

    rom = _roman_numeral_token_to_ipa(letters_only, v)
    if rom is not None:
        return _finalize_word_ipa(
            rom,
            with_stress=with_stress,
            vocoder_stress=False,
            from_lexicon=True,
            keep_syllable_dots=keep_syllable_dots,
        )

    lex = lexicon if lexicon is not None else _get_lexicon(v, dict_path)
    ipa = _lookup_lexicon(letters_only, lex)
    if ipa is not None:
        return _finalize_word_ipa(
            ipa,
            with_stress=with_stress,
            vocoder_stress=False,
            from_lexicon=True,
            keep_syllable_dots=keep_syllable_dots,
        )

    if "-" in letters_only:
        chunks = [c for c in letters_only.split("-") if c]
        sub = [_lookup_lexicon(c, lex) for c in chunks]
        if chunks and all(x is not None for x in sub):
            merged = "-".join(sub)
            return _finalize_word_ipa(
                merged,
                with_stress=with_stress,
                vocoder_stress=False,
                from_lexicon=True,
                keep_syllable_dots=keep_syllable_dots,
            )

    fw = _function_words_for(v)
    if letters_only in fw:
        return _finalize_word_ipa(
            fw[letters_only],
            with_stress=with_stress,
            vocoder_stress=vocoder_stress,
            from_lexicon=False,
            keep_syllable_dots=keep_syllable_dots,
        )

    ipa_rules = _rules_word_to_ipa(raw, v, with_stress=with_stress)
    # ⟨s⟩+⟨c⟩ straddle entries use hand-tuned stress; skip vocoder normalization.
    oov_sc_hand = letters_only in _OOV_SC_STRADDLE
    if v == "pt_pt" and apply_pt_pt_final_esh and not oov_sc_hand:
        ipa_rules = _pt_pt_apply_rules_final_s_to_esh(ipa_rules, letters_only)
    return _finalize_word_ipa(
        ipa_rules,
        with_stress=with_stress,
        vocoder_stress=vocoder_stress,
        from_lexicon=oov_sc_hand,
        keep_syllable_dots=keep_syllable_dots,
    )


def text_to_ipa(
    text: str,
    *,
    variant: str = "pt_br",
    lexicon: Mapping[str, str] | None = None,
    dict_path: Path | None = None,
    with_stress: bool = True,
    vocoder_stress: bool = True,
    keep_syllable_dots: bool = False,
    apply_pt_pt_final_esh: bool = True,
    expand_cardinal_digits: bool = True,
) -> str:
    """Tokenize and G2P each word; preserve punctuation and collapse spaces."""
    v = variant.strip().lower().replace("-", "_")
    if expand_cardinal_digits:
        text = expand_digit_tokens_in_text(text, variant=v)
    parts: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isspace():
            parts.append(" ")
        elif _PORT_WORD_FULLMATCH.fullmatch(tok):
            parts.append(
                word_to_ipa(
                    tok,
                    variant=variant,
                    lexicon=lexicon,
                    dict_path=dict_path,
                    with_stress=with_stress,
                    vocoder_stress=vocoder_stress,
                    keep_syllable_dots=keep_syllable_dots,
                    apply_pt_pt_final_esh=apply_pt_pt_final_esh,
                    expand_cardinal_digits=False,
                )
            )
        else:
            parts.append(tok)
    out = "".join(parts)
    return re.sub(r" +", " ", out).strip()


def coarse_ipa_for_compare(s: str) -> str:
    """
    Strip stress, syllable dots, and most combining marks for loose agreement checks vs eSpeak.
    """
    t = unicodedata.normalize("NFD", s)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
    t = t.replace(".", "").replace("͡", "")
    return t.lower()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Portuguese text to IPA (Brazil or Portugal) using data/pt_*/dict.tsv plus OOV rules."
    )
    p.add_argument("text", nargs="*", help="Portuguese text (if empty, read stdin)")
    p.add_argument(
        "--variant",
        choices=("pt_br", "pt_pt"),
        default="pt_br",
        help="Lexicon + rule preset (default: pt_br).",
    )
    p.add_argument(
        "--dict",
        type=Path,
        default=None,
        metavar="PATH",
        help="Lexicon TSV (default: data/<variant>/dict.tsv).",
    )
    p.add_argument("--no-stress", action="store_true", help="Strip stress marks from output.")
    p.add_argument(
        "--syllable-initial-stress",
        action="store_true",
        help="For rule-based IPA, keep syllable-edge ˈ/ˌ; default shifts to nuclear (eSpeak-style).",
    )
    p.add_argument(
        "--keep-syllable-dots",
        action="store_true",
        help="Keep '.' syllable boundaries from the lexicon (ipa-dict); default strips them (eSpeak-like).",
    )
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin.")
    p.add_argument(
        "--no-expand-digits",
        action="store_true",
        help="Leave digit sequences as digits (no spoken Portuguese cardinal expansion).",
    )
    p.add_argument("--no-espeak", action="store_true", help="Do not print eSpeak NG reference line.")
    p.add_argument(
        "--espeak-voice",
        type=str,
        default=None,
        metavar="VOICE",
        help="eSpeak voice (default: pt-br or pt from --variant).",
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.stdin or not args.text:
        import sys

        raw = sys.stdin.read()
    else:
        raw = " ".join(args.text)
    v = args.variant
    lex = load_portuguese_lexicon(args.dict, variant=v) if args.dict is not None else None
    expand_digits = not args.no_expand_digits
    es_in = expand_digit_tokens_in_text(raw, variant=v) if expand_digits else raw
    print(
        text_to_ipa(
            raw,
            variant=v,
            lexicon=lex,
            dict_path=args.dict,
            with_stress=not args.no_stress,
            vocoder_stress=not args.syllable_initial_stress,
            keep_syllable_dots=args.keep_syllable_dots,
            expand_cardinal_digits=expand_digits,
        )
    )
    if not args.no_espeak:
        voice = args.espeak_voice or default_espeak_voice(v)
        es = espeak_ng_ipa_line(es_in, voice=voice)
        if es is not None:
            print(f"{es} (espeak-ng)")


if __name__ == "__main__":
    main()
