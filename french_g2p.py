#!/usr/bin/env python3
"""
French grapheme-to-phoneme helpers with **lexicon lookup**, **shallow POS hints**, and **liaison**.

The large IPA lexicon in ``data/fr/dict.tsv`` (word ``\\t`` IPA) provides citation forms; final
latent consonants are usually absent (e.g. *les* → ``le``, *nous* → ``nu``). Liaison inserts a
linking consonant when two tokens meet under grammatical conditions inferred from the CSV
word lists in ``data/fr`` (determiners, pronouns, nouns, verbs, etc.).

This is a **practical TTS-oriented approximation**, not a full phonological parser:

- **Obligatory** liaison is approximated for tight groups such as determiner + noun/adjective,
  pronoun + verb, and *et* + vowel-initial word.
- **Forbidden** liaison is blocked for noun + verb and before a small **h aspiré** set.
- **Optional** liaison (e.g. adjective + noun) follows a register flag.
- **Enchaînement** (resyllabification of already pronounced codas) is not rewritten; consecutive
  words are joined with spaces. Citation IPA that already ends in an audible consonant is not
  given a second liaison segment from orthography.
- **Schwa** (*e caduc*) and narrow regional variation are whatever the lexicon encodes.

By default, each word gets a **nuclear primary stress** (ˈ) before its **last vowel** via
:func:`ensure_french_nuclear_stress`, matching eSpeak-style output that many vocoders expect; the
lexicon itself usually omits stress marks.

**OOV** tokens fall back to :mod:`french_oov_rules` (pure-Python digraphs + mute finals + final
syllable stress) when the lexicon has no entry; disable with ``FrenchG2PConfig(oov_rules=False)``.

Digit-only tokens (e.g. ``1891``) are expanded to French cardinals via :mod:`french_numbers` before
G2P (up to 999_999); disable with ``FrenchG2PConfig(expand_cardinal_digits=False)``.

CLI: prints lexicon/rules IPA first, then an eSpeak NG reference line by default (``--no-espeak`` to
disable), same stack as ``german_rule_g2p`` / ``spanish_rule_g2p`` / ``heteronym.espeak_heteronyms``.
Default voice is ``fr`` (use ``--espeak-voice fr-fr`` etc. when your build supports it).
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from french_numbers import expand_cardinal_digits_to_french_words, cardinal_compound_ipa, expand_digit_tokens_in_text
from french_oov_rules import oov_word_to_ipa

_DEFAULT_ESPEAK_VOICE = "fr"

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT_PATH = _REPO_ROOT / "data" / "fr" / "dict.tsv"
_DEFAULT_CSV_DIR = _REPO_ROOT / "data" / "fr"

_LEXICON_CACHE: dict[str, str] | None = None
_LEXICON_PATH: Path | None = None
_POS_CACHE: tuple[dict[str, frozenset[str]], ...] | None = None
_POS_DIR: Path | None = None

LiaisonStrength = Literal["none", "optional", "obligatory"]


def espeak_ng_ipa_line(text: str, *, voice: str = _DEFAULT_ESPEAK_VOICE) -> str | None:
    """
    IPA string from libespeak-ng via ``espeak_phonemizer`` (same separator policy as
    :func:`heteronym.espeak_heteronyms.espeak_phonemize_ipa_raw`).

    Returns ``None`` if the optional dependency or engine is unavailable, or phonemization fails.
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


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def normalize_lookup_key(word: str) -> str:
    """Lowercase NFC key for lexicon lookup (letters including French diacritics, apostrophe, hyphen)."""
    t = _nfc(word.strip().lower())
    return re.sub(r"[^a-zàâäéèêëïîôùûüÿçœæ'-]+", "", t)


def load_french_lexicon(path: Path | None = None) -> dict[str, str]:
    """
    Load ``word\\tIPA`` TSV. Duplicate keys: prefer an all-lowercase surface row over a
    capitalized one (same policy as ``german_rule_g2p.load_german_lexicon``).
    """
    p = path or _DEFAULT_DICT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"French lexicon not found: {p}")
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
    if _LEXICON_CACHE is not None and (_LEXICON_PATH is None or _LEXICON_PATH == p):
        return _LEXICON_CACHE
    _LEXICON_CACHE = load_french_lexicon(p)
    _LEXICON_PATH = p
    return _LEXICON_CACHE


def _parse_tags(cell: str) -> list[str]:
    cell = cell.strip()
    if not cell:
        return []
    try:
        v = ast.literal_eval(cell)
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
    except (ValueError, SyntaxError, TypeError):
        pass
    return []


def load_french_pos_inventory(csv_dir: Path | None = None) -> dict[str, frozenset[str]]:
    """
    Map category name (from ``*.csv`` stem, uppercased: DET, NOUN, VERB, …) to a frozenset of
    **single-token** lowercase forms. Multi-word rows are skipped (phrase tagging is out of scope).
    """
    d = csv_dir or _DEFAULT_CSV_DIR
    if not d.is_dir():
        raise FileNotFoundError(f"French morphology directory not found: {d}")
    out: dict[str, set[str]] = {}
    for path in sorted(d.glob("*.csv")):
        cat = path.stem.upper()
        forms: set[str] = set()
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "form" not in (reader.fieldnames or ()):
                continue
            for row in reader:
                form = (row.get("form") or "").strip()
                if not form or form == "-" or " " in form:
                    continue
                forms.add(form.lower())
        if forms:
            out[cat] = forms
    return {k: frozenset(v) for k, v in out.items()}


def _get_pos_inventory(csv_dir: Path | None) -> dict[str, frozenset[str]]:
    global _POS_CACHE, _POS_DIR
    d = csv_dir or _DEFAULT_CSV_DIR
    if _POS_CACHE is not None and (_POS_DIR is None or _POS_DIR == d):
        return _POS_CACHE[0]
    inv = load_french_pos_inventory(d)
    _POS_CACHE = (inv,)
    _POS_DIR = d
    return inv


# Order for: (a) scanning membership, (b) default disambiguation among multiple categories
_POS_SCAN_ORDER: tuple[str, ...] = (
    "DET",
    "PRON",
    "PREP",
    "CONJ",
    "ADJ",
    "ADV",
    "VERB",
    "NOUN",
)


def _categories_for_form(word: str, inv: Mapping[str, frozenset[str]]) -> list[str]:
    k = word.lower()
    found: list[str] = []
    for cat in _POS_SCAN_ORDER:
        if k in inv.get(cat, frozenset()):
            found.append(cat)
    return found


def classify_pos(
    word: str,
    inv: Mapping[str, frozenset[str]],
    *,
    prev_pos: str | None = None,
) -> str | None:
    """
    Pick one POS for *word* using inventory membership and light context *prev_pos*
    (previous word's POS) to break noun/verb ambiguity after determiners and pronouns.
    """
    cands = _categories_for_form(word, inv)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    if prev_pos == "DET":
        if "ADJ" in cands:
            return "ADJ"
        if "NOUN" in cands:
            return "NOUN"
    if prev_pos == "PRON":
        if "VERB" in cands:
            return "VERB"
    for cat in _POS_SCAN_ORDER:
        if cat in cands:
            return cat
    return cands[0]


# H aspiré (no liaison / liaison vocalique onto the following word)
_H_ASPIRE: frozenset[str] = frozenset(
    {
        "hareng",
        "harpagon",
        "harpe",
        "hargneux",
        "hargneusement",
        "hautain",
        "haut",
        "hâte",
        "haïr",
        "haï",
        "haïe",
        "haïes",
        "haïs",
        "héros",
        "héroïne",
        "hérisson",
        "hérésie",
        "hiérarchie",
        "hollande",
        "honte",
        "honteux",
        "huit",
        "huitième",
        "humble",
        "humour",
        "hurler",
        "hutte",
    }
)

_PRIMARY = "\u02c8"
_SECONDARY = "\u02cc"

# IPA vowel symbols (single-char bases) and common nasals / semi-vowels at phrase edge
_VOWEL_BASES = frozenset("aeiouyɑɛəɜɪʊɔøœʏɐ")
_NASAL_MARK = "\u0303"  # combining tilde


def _strip_stress(ipa: str) -> str:
    return ipa.replace(_PRIMARY, "").replace(_SECONDARY, "")


# Longest first: syllable nuclei for placing ˈ before the **last** nucleus (French word stress).
_FRENCH_NUCLEUS_PREFIXES: tuple[str, ...] = tuple(
    sorted(
        (
            "ɑ̃",
            "ɛ̃",
            "ɔ̃",
            "œ̃",
            "ə",
            "ɛ",
            "œ",
            "ø",
            "ɔ",
            "ɑ",
            "æ",
            "ɜ",
            "a",
            "e",
            "i",
            "o",
            "u",
            "y",
            "ɪ",
            "ʊ",
        ),
        key=len,
        reverse=True,
    )
)


def _french_nucleus_spans(ipa_no_stress: str) -> list[tuple[int, int]]:
    """Inclusive start, exclusive end for each vowel / nasal nucleus, left to right."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(ipa_no_stress)
    while i < n:
        matched = False
        for p in _FRENCH_NUCLEUS_PREFIXES:
            if ipa_no_stress.startswith(p, i):
                spans.append((i, i + len(p)))
                i += len(p)
                matched = True
                break
        if not matched:
            i += 1
    return spans


def ensure_french_nuclear_stress(ipa: str) -> str:
    """
    Ensure a single primary stress mark (ˈ) sits **immediately before the last syllable nucleus**.

    The French lexicon often omits stress; eSpeak NG and many vocoders expect nuclear ˈ similar to
    :func:`german_rule_g2p.normalize_ipa_stress_for_vocoder`. Idempotent when already correct.
    Hyphenated IPA is split, each piece stressed, then rejoined.
    """
    if not ipa or not ipa.strip():
        return ipa
    if "-" in ipa:
        return "-".join(ensure_french_nuclear_stress(chunk) for chunk in ipa.split("-") if chunk)
    s = _strip_stress(ipa)
    if not s:
        return ipa
    spans = _french_nucleus_spans(s)
    if not spans:
        return _PRIMARY + s
    start = spans[-1][0]
    return s[:start] + _PRIMARY + s[start:]


def _ipa_starts_with_vowel_sound(ipa: str) -> bool:
    """True if *ipa* begins with a vowel, nasal vowel, or mute-h context (no h in IPA)."""
    s = _strip_stress(ipa)
    if not s:
        return False
    # Digraph-style starts common in French lexicons
    if s.startswith(("ɥ", "w", "j")) and len(s) > 1 and s[1] in _VOWEL_BASES | {"ə"}:
        return True
    ch0 = s[0]
    if ch0 in _VOWEL_BASES:
        return True
    if ch0 in "œɶ":
        return True
    if len(s) >= 2 and s[1] == _NASAL_MARK and ch0 in "aɔoœeɛiɑu":
        return True
    return False


def _ipa_ends_with_audible_consonant(ipa: str) -> bool:
    """Rough check: citation ends in a realized obstruent/sonorant (liaison C not latent)."""
    s = _strip_stress(ipa)
    if not s:
        return False
    last = s[-1]
    if last == _NASAL_MARK and len(s) >= 2:
        return False
    if last in _VOWEL_BASES or last in "œɶ":
        return False
    if last in "bdfɡɟhjklmnpstvzʃʒɲŋʁwɥc":
        return True
    return False


def _replace_suffix_once(ipa: str, old: str, new: str) -> str:
    i = ipa.rfind(old)
    if i < 0:
        return ipa
    return ipa[:i] + new + ipa[i + len(old) :]


def _nasal_liaison_transform(word: str, ipa: str) -> str | None:
    """
    Determiners / adjectives with word-final nasal vowel in isolation: realize /n/ before a vowel.
    Returns full **replacement IPA** for the left word when liaison applies, else None.
    """
    w = word.lower()
    s = _strip_stress(ipa)
    if w in {"mon", "ton", "son", "bon"} and s.endswith("ɔ̃"):
        return _replace_suffix_once(ipa, "ɔ̃", "ɔn")
    if w == "un" and s.endswith("œ̃"):
        return _replace_suffix_once(ipa, "œ̃", "œn")
    if w in {"aucun", "aucune"} and s.endswith("œ̃"):
        return _replace_suffix_once(ipa, "œ̃", "œn")
    if w == "en" and s.endswith("ɑ̃"):
        return _replace_suffix_once(ipa, "ɑ̃", "ɑn")
    return None


def _orthographic_liaison_consonant(word: str) -> str | None:
    """
    Map a **final latent** consonant letter to its liaison realization (single IPA string).
    Crude ``e``-stripping handles many feminine / mute-e endings.
    """
    w = re.sub(r"[^a-zàâäéèêëïîôùûüÿçœæ-]", "", word.lower())
    if not w:
        return None
    if w.endswith(("ent", "ont")):
        return "t"
    # One word-final mute ⟨e⟩ only (``rstrip("e")`` would mangle *des* → *d*).
    if len(w) > 1 and w.endswith("e"):
        w = w[:-1]
    if not w:
        return None
    last = w[-1]
    cmap = {
        "s": "z",
        "x": "z",
        "z": "z",
        "d": "t",
        "t": "t",
        "n": "n",
        "r": "ʁ",
        "l": "l",
        "f": "v",
        "c": "k",
        "p": "p",
        "g": "ɡ",
        "m": "m",
        "b": "b",
    }
    return cmap.get(last)


# Articles / partitives often missing from ``det.csv`` but still trigger liaison.
_CLOSED_LIAISON_DETERMINERS: frozenset[str] = frozenset(
    {
        "les",
        "des",
        "ces",
        "mes",
        "tes",
        "ses",
        "nos",
        "vos",
        "leurs",
        "aux",
        "quelques",
        "plusieurs",
        "certains",
        "certaines",
    }
)


def liaison_strength(
    pos_left: str | None,
    pos_right: str | None,
    wleft: str,
    wright: str,
    *,
    optional_register_formal: bool,
) -> LiaisonStrength:
    """
    Grammatical liaison strength between two **words** (already lowercased tokens).

    Unknown POS on either side is mostly **none**, except closed-class determiners
    (``_CLOSED_LIAISON_DETERMINERS``) before noun/adjective.
    """
    del wright  # reserved for future h / negation patterns
    wl = wleft.lower()
    if pos_left == "CONJ" and wl == "et":
        return "obligatory"
    if pos_left is None and wl in _CLOSED_LIAISON_DETERMINERS and pos_right in {"NOUN", "ADJ"}:
        return "obligatory"
    # Lexicon misses POS but determiner / pronoun + likely nominal (vowel check in _apply_liaison_pair).
    if pos_right is None and pos_left in {"PRON", "DET"}:
        return "obligatory"
    if pos_left is None and pos_right is None:
        return "none"
    if pos_left is None or pos_right is None:
        return "none"
    # Forbidden across major boundaries
    if pos_left == "NOUN" and pos_right == "VERB":
        return "none"
    if pos_left == "VERB" and pos_right == "NOUN":
        return "none"
    # Pronoun + verb; pronoun + noun (e.g. *les* is tagged PRON in some lexicons)
    if pos_left == "PRON" and pos_right == "VERB":
        return "obligatory"
    if pos_left == "PRON" and pos_right == "NOUN":
        return "obligatory"
    # Determiner + noun/adjective
    if pos_left == "DET" and pos_right in {"NOUN", "ADJ"}:
        return "obligatory"
    if pos_left == "DET" and pos_right == "ADV":
        return "optional"
    # Preposition + … (en, dans, sur …): often phrase-specific; allow optional
    if pos_left == "PREP":
        return "optional" if optional_register_formal else "none"
    # Adjective + noun
    if pos_left == "ADJ" and pos_right == "NOUN":
        return "optional" if optional_register_formal else "none"
    if pos_left == "CONJ":
        return "none"
    return "none"


def _apply_liaison_pair(
    ipa_left: str,
    ipa_right: str,
    wleft: str,
    wright: str,
    strength: LiaisonStrength,
) -> tuple[str, str]:
    """Return (modified_left, right) IPA after liaison, or originals if no liaison."""
    if strength == "none":
        return ipa_left, ipa_right
    if not ipa_left.strip():
        return ipa_left, ipa_right
    if wright.lower() in _H_ASPIRE:
        return ipa_left, ipa_right
    if not _ipa_starts_with_vowel_sound(ipa_right):
        return ipa_left, ipa_right

    nasal = _nasal_liaison_transform(wleft, ipa_left)
    if nasal is not None:
        return nasal, ipa_right

    if _ipa_ends_with_audible_consonant(ipa_left):
        return ipa_left, ipa_right

    c = _orthographic_liaison_consonant(wleft)
    if not c:
        return ipa_left, ipa_right
    if ipa_left.rstrip().endswith(c):
        return ipa_left, ipa_right

    return ipa_left + c, ipa_right


@dataclass(frozen=True)
class FrenchG2PConfig:
    """Runtime configuration for :func:`word_to_ipa` / :func:`text_to_ipa`."""

    dict_path: Path | None = None
    csv_dir: Path | None = None
    with_stress: bool = True
    liaison: bool = True
    liaison_optional: bool = True  # if True, optional liaisons are realized; if False, only obligatory
    oov_rules: bool = True  # rule-based IPA when lexicon misses (see :mod:`french_oov_rules`)
    expand_cardinal_digits: bool = True  # ``1891`` → *mille huit cent quatre-vingt-onze* before G2P


def _lookup_lexicon(key: str, lexicon: Mapping[str, str]) -> str | None:
    if key in lexicon:
        return lexicon[key]
    if key.endswith("'"):
        k2 = key.rstrip("'")
        if k2 in lexicon:
            return lexicon[k2]
    return None


# Context-sensitive spelling → default citation IPA
_HETERONYM_DEFAULT_IPA: dict[str, str] = {
    "est": "ɛ",  # copula *être* (cardinal *est* /ɛst/ is rare in prose)
    "a": "a",  # auxiliary / avoir — lexicon should win if present
}


def word_to_ipa(
    word: str,
    *,
    lexicon: Mapping[str, str] | None = None,
    dict_path: Path | None = None,
    with_stress: bool = True,
    use_oov_rules: bool = True,
) -> str:
    """Single-token G2P: lexicon lookup (normalized key), else rule-based OOV IPA if enabled."""
    if not word or not word.strip():
        return ""
    raw = word.strip()
    key = normalize_lookup_key(raw)
    if not key:
        return ""

    lex = lexicon if lexicon is not None else _get_lexicon(dict_path)

    if raw.isdigit():
        phrase = expand_cardinal_digits_to_french_words(raw)
        if phrase != raw:
            return globals()["text_to_ipa"](
                phrase,
                lexicon=lex,
                dict_path=dict_path,
                config=FrenchG2PConfig(
                    with_stress=with_stress,
                    liaison=True,
                    liaison_optional=True,
                    oov_rules=use_oov_rules,
                    expand_cardinal_digits=False,
                ),
            )

    ipa = _lookup_lexicon(key, lex)
    if ipa is None and key in _HETERONYM_DEFAULT_IPA:
        ipa = _HETERONYM_DEFAULT_IPA[key]
    from_compound = False
    if ipa is None:
        cipa = cardinal_compound_ipa(raw.lower())
        if cipa is not None:
            ipa = cipa
            from_compound = True
    if ipa is None and use_oov_rules:
        ipa = oov_word_to_ipa(raw, with_stress=with_stress)
    elif ipa is None:
        ipa = ""
    if not ipa:
        return ""
    if not with_stress:
        ipa = _strip_stress(ipa)
    elif not from_compound:
        ipa = ensure_french_nuclear_stress(ipa)
    return ipa


_TOKEN_RE = re.compile(r"[\w'-]+|[^\w\s'-]+|\s+", flags=re.UNICODE)


def text_to_ipa(
    text: str,
    *,
    lexicon: Mapping[str, str] | None = None,
    dict_path: Path | None = None,
    pos_inventory: Mapping[str, frozenset[str]] | None = None,
    csv_dir: Path | None = None,
    config: FrenchG2PConfig | None = None,
) -> str:
    """
    Tokenize *text*, G2P each word, optionally apply **liaison** between adjacent word tokens.

    Punctuation and whitespace are preserved; inter-word spaces collapse to a single space between
    word IPA spans (each word's IPA is one token in the output).
    """
    cfg = config or FrenchG2PConfig()
    if cfg.expand_cardinal_digits:
        text = expand_digit_tokens_in_text(text)
    lex = lexicon if lexicon is not None else _get_lexicon(dict_path or cfg.dict_path)
    inv = pos_inventory if pos_inventory is not None else _get_pos_inventory(csv_dir or cfg.csv_dir)

    words: list[str] = []
    tokens: list[tuple[str, bool]] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.isspace():
            tokens.append((tok, False))
        elif re.fullmatch(r"[\w'-]+", tok, flags=re.UNICODE):
            tokens.append((tok, True))
        else:
            tokens.append((tok, False))

    for tok, is_w in tokens:
        if is_w:
            words.append(tok)

    ipas: list[str | None] = []
    poses: list[str | None] = []
    prev_pos: str | None = None
    for w in words:
        k = normalize_lookup_key(w)
        ipa = word_to_ipa(
            w,
            lexicon=lex,
            dict_path=dict_path,
            with_stress=cfg.with_stress,
            use_oov_rules=cfg.oov_rules,
        )
        if not ipa and k:
            ipa = ""  # still classify POS for structure
        ipas.append(ipa if ipa else None)
        pos = classify_pos(w, inv, prev_pos=prev_pos)
        poses.append(pos)
        if pos is not None:
            prev_pos = pos
        elif w.lower() in _CLOSED_LIAISON_DETERMINERS or w.lower() in inv.get("DET", frozenset()):
            prev_pos = "DET"
        else:
            prev_pos = pos

    if not cfg.liaison:
        out_words = [
            word_to_ipa(
                w,
                lexicon=lex,
                dict_path=dict_path,
                with_stress=cfg.with_stress,
                use_oov_rules=cfg.oov_rules,
            )
            for w in words
        ]
        ipa_by_idx = [o for o in out_words]
    else:
        ipa_by_idx: list[str] = []
        for i in range(len(words)):
            left = ipas[i] or ""
            if i + 1 < len(words):
                right = ipas[i + 1] or ""
                wl, wr = words[i], words[i + 1]
                pl, pr = poses[i], poses[i + 1]
                st = liaison_strength(
                    pl,
                    pr,
                    wl,
                    wr,
                    optional_register_formal=cfg.liaison_optional,
                )
                if st == "none" and wl.lower() == "et" and pl is None:
                    st = "obligatory"
                if st == "optional" and not cfg.liaison_optional:
                    st = "none"
                if st != "none":
                    left, _ = _apply_liaison_pair(left, right, wl, wr, st)
            if cfg.with_stress and left.count(_PRIMARY) <= 1:
                left = ensure_french_nuclear_stress(left)
            ipa_by_idx.append(left)

    wi = 0
    parts: list[str] = []
    prev_was_word = False
    for tok, is_w in tokens:
        if is_w:
            ipa_w = ipa_by_idx[wi] if wi < len(ipa_by_idx) else ""
            # Space between consecutive words when the source had no explicit whitespace
            # between them (rare); normal spaces come from whitespace tokens below.
            if prev_was_word:
                parts.append(" ")
            parts.append(ipa_w)
            wi += 1
            prev_was_word = True
        elif tok.isspace():
            # Always preserve source whitespace; otherwise `!` / `?` leave no gap before
            # the next word (prev_was_word is False after punctuation).
            parts.append(" ")
            prev_was_word = False
        else:
            parts.append(tok)
            prev_was_word = False
    out = "".join(parts)
    return re.sub(r" +", " ", out).strip()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="French text to IPA (lexicon + liaison + OOV rules), with optional eSpeak NG "
        "reference line."
    )
    p.add_argument("text", nargs="*", help="French text (if empty, read stdin)")
    p.add_argument(
        "--dict",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Lexicon TSV (default: {_DEFAULT_DICT_PATH}).",
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=f"Directory of form,tags CSV files (default: {_DEFAULT_CSV_DIR}).",
    )
    p.add_argument("--no-stress", action="store_true", help="Strip ˈ / ˌ from lexicon IPA.")
    p.add_argument("--no-liaison", action="store_true", help="Do not apply liaison between words.")
    p.add_argument(
        "--no-oov-rules",
        action="store_true",
        help="Do not use built-in grapheme rules for unknown words (lexicon-only).",
    )
    p.add_argument(
        "--no-optional-liaison",
        action="store_true",
        help="Only obligatory liaisons (skip adj+noun, prep+…, etc.).",
    )
    p.add_argument(
        "--no-expand-digits",
        action="store_true",
        help="Leave digit sequences as digits (no cardinal French word expansion).",
    )
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin.")
    p.add_argument(
        "--no-espeak",
        action="store_true",
        help="Do not print a second IPA line from eSpeak NG (requires espeak-phonemizer + libespeak-ng).",
    )
    p.add_argument(
        "--espeak-voice",
        type=str,
        default=_DEFAULT_ESPEAK_VOICE,
        metavar="VOICE",
        help=f"eSpeak voice for the reference line (default: {_DEFAULT_ESPEAK_VOICE}; try fr-fr if installed).",
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    if args.stdin or not args.text:
        import sys

        raw = sys.stdin.read()
    else:
        raw = " ".join(args.text)
    cfg = FrenchG2PConfig(
        dict_path=args.dict,
        csv_dir=args.csv_dir,
        with_stress=not args.no_stress,
        liaison=not args.no_liaison,
        liaison_optional=not args.no_optional_liaison,
        oov_rules=not args.no_oov_rules,
        expand_cardinal_digits=not args.no_expand_digits,
    )
    print(text_to_ipa(raw, config=cfg))
    if not args.no_espeak:
        es_in = expand_digit_tokens_in_text(raw) if cfg.expand_cardinal_digits else raw
        es = espeak_ng_ipa_line(es_in, voice=args.espeak_voice)
        if es is not None:
            print(f"{es} (espeak-ng)")


if __name__ == "__main__":
    main()
