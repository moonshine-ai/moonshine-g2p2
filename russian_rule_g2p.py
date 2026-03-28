#!/usr/bin/env python3
"""
Rule- and lexicon-based Russian grapheme-to-phoneme (broad IPA).

* In-vocabulary words are read from ``data/ru/dict.tsv`` (word ``\\t`` IPA). When the same
  normalized key appears from a capitalized row and later from an all-lowercase lemma, the
  **lowercase** row wins (same policy as ``german_rule_g2p`` / ``italian_rule_g2p``).
* Out-of-vocabulary tokens use a compact Cyrillic letter pass: palatalization before
  ``е ё и ю я`` and ``ь``, optional ``ъ`` blocking jot, rough vowel reduction on unstressed
  syllables, and orthographic syllables with **maximal onset** between vowels (typical for
  Russian consonant clusters). Stress comes from ``ё``, from a combining acute on a vowel
  (``́`` U+0301), or defaults to the **first** syllable (lexicon should cover high-frequency
  lemmas where stress is unpredictable).

Limitations (intentional):
- Voicing assimilation (всё, сделать), optional vowel reduction (especially ``е``/``а`` in
  prefixes), and ``тс``/``тьс`` realizations are not fully modeled in the rules path.
- eSpeak NG ``ru`` uses its own phone inventory (e.g. ``ɭ`` for palatal ``л``); use
  :func:`coarse_ipa_for_compare` for loose agreement checks.

By default, ˈ / ˌ are shifted to sit before the syllable nucleus via
:func:`german_rule_g2p.normalize_ipa_stress_for_vocoder` (eSpeak-style). Pass
``--syllable-initial-stress`` to keep dictionary-style marks at the first segment of the
stressed syllable.

CLI: prints lexicon/rules IPA, then an eSpeak NG reference line by default (``--no-espeak``
to disable), same stack as ``german_rule_g2p`` / ``italian_rule_g2p``.

ASCII digit-only tokens (and ``1933-1945``-style ranges) expand to Russian cardinals in Cyrillic
via :mod:`russian_numbers` before G2P (up to 999_999); disable with ``expand_cardinal_digits=False``
or CLI ``--no-expand-digits``.

Wiki checks: ``scripts/verify_russian_g2p_wiki.py`` samples ``data/ru/wiki-text.txt``.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

from german_rule_g2p import normalize_ipa_stress_for_vocoder
from russian_numbers import expand_cardinal_digits_to_russian_words, expand_digit_tokens_in_text

_DEFAULT_ESPEAK_VOICE = "ru"

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT_PATH = _REPO_ROOT / "data" / "ru" / "dict.tsv"


def default_dict_path() -> Path:
    """Default ``data/ru/dict.tsv`` path."""
    return _DEFAULT_DICT_PATH

_LEXICON_CACHE: dict[str, str] | None = None
_LEXICON_PATH: Path | None = None

_PRIMARY_STRESS = "\u02c8"  # ˈ
_SECONDARY_STRESS = "\u02cc"  # ˌ
_ACUTE = "\u0301"  # combining acute (orthographic stress in running text)

# When ``expand_cardinal_digits`` is off: pass digit / simple year-range tokens through unchanged.
_DIGIT_PASS_THROUGH_RE = re.compile(r"^[0-9]+(?:-[0-9]+)*$")


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
    """
    Lowercase key for TSV lookup: strip combining marks (stress, rare diacritics), keep
    Cyrillic letters and hyphen.
    """
    t = unicodedata.normalize("NFC", word.strip().lower())
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = unicodedata.normalize("NFC", t)
    return re.sub(r"[^а-яё\-]+", "", t)


def load_russian_lexicon(path: Path | None = None) -> dict[str, str]:
    """
    Load ``word\\tIPA`` TSV. Duplicate keys: prefer an all-lowercase surface row over a
    capitalized one (homograph policy).
    """
    p = path or _DEFAULT_DICT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Russian lexicon not found: {p}")
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
    _LEXICON_CACHE = load_russian_lexicon(p)
    _LEXICON_PATH = p
    return _LEXICON_CACHE


def _lookup_lexicon(key: str, lexicon: Mapping[str, str]) -> str | None:
    if key in lexicon:
        return lexicon[key]
    return None


# --- OOV rules: orthographic syllables (maximal onset between vowel letters) ---

_VOWELS = frozenset("аеёиоуыэюя")


def _is_vowel_letter(ch: str) -> bool:
    return ch.lower() in _VOWELS


def _vowel_nucleus_spans(w: str) -> list[tuple[int, int]]:
    """One nucleus per vowel letter (``ё`` is a single letter)."""
    out: list[tuple[int, int]] = []
    i = 0
    n = len(w)
    while i < n:
        ch = w[i]
        if _is_vowel_letter(ch):
            out.append((i, i + 1))
            i += 1
        else:
            i += 1
    return out


def russian_orthographic_syllables(word: str) -> list[str]:
    """
    Rough orthographic syllables: consonants between vowels start the **next** syllable
    (maximal onset). Hyphen splits compounds before syllabifying each piece.
    """
    w = re.sub(r"[^а-яёA-ZА-ЯЁ\-]", "", word.lower())
    if not w:
        return []
    if "-" in w:
        parts: list[str] = []
        for chunk in w.split("-"):
            if chunk:
                parts.extend(russian_orthographic_syllables(chunk))
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
            syllables.append(cur)
            cur = cluster
        else:
            syllables.append(cur + w[e:])
    return [s for s in syllables if s]


def _strip_grapheme_diacritics(word: str) -> str:
    """NFC lowercase, strip combining marks (stress), keep letters and hyphen."""
    t = unicodedata.normalize("NFC", word.strip().lower())
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", t)


def _filter_russian_graphemes_keep_stress(s: str) -> str:
    """Lowercase NFC: keep Cyrillic letters, hyphen, and combining marks (e.g. stress)."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", s.strip().lower()):
        if unicodedata.combining(ch):
            out.append(ch)
        elif re.match(r"[а-яё\-]", ch):
            out.append(ch)
    return "".join(out)


def _acute_stressed_vowel_ordinal(w_nfc: str) -> int | None:
    """
    If a vowel bears combining acute (``́``), return its 0-based index among vowel letters;
    else ``None``.
    """
    i = 0
    n = len(w_nfc)
    v_ord = 0
    while i < n:
        if w_nfc[i].lower() in _VOWELS:
            j = i + 1
            while j < n and unicodedata.combining(w_nfc[j]):
                if w_nfc[j] == _ACUTE:
                    return v_ord
                j += 1
            v_ord += 1
            i = j
        else:
            i += 1
    return None


def _vowel_ordinal_to_syllable(syls: list[str], nucleus_ord: int) -> int:
    """Map vowel ordinal (0-based) to orthographic syllable index."""
    v = 0
    for si, sy in enumerate(syls):
        for ch in sy:
            if ch.lower() in _VOWELS:
                if v == nucleus_ord:
                    return si
                v += 1
    return 0


def _stress_syllable_index(syls: list[str], word_with_stress_marks: str) -> int:
    """
    Stressed syllable index: ``ё`` wins; else combining acute on a vowel; else ``0``.

    *word_with_stress_marks* may include ``́`` (NFC); syllables in *syls* must match the
    stress-stripped grapheme string.
    """
    if not syls:
        return 0
    w_marked = unicodedata.normalize("NFC", word_with_stress_marks.lower())
    for si, sy in enumerate(syls):
        if "ё" in sy:
            return si
    ord_acute = _acute_stressed_vowel_ordinal(w_marked)
    if ord_acute is not None:
        return _vowel_ordinal_to_syllable(syls, ord_acute)
    return 0


def _char_syllable_indices(w: str) -> list[int]:
    """Parallel to *w*: syllable index for each character (hyphen-free word)."""
    syls = russian_orthographic_syllables(w)
    if not syls or len(w) == 0:
        return [0] * len(w)
    pos_map: list[int] = []
    for si, sy in enumerate(syls):
        for _ in sy:
            pos_map.append(si)
    if len(pos_map) != len(w):
        return [0] * len(w)
    return pos_map


# Consonants that receive IPA palatalization marker ``ʲ`` before front vowels / ``ь``.
_PALATALIZABLE = frozenset("бвгдзклмнпрстфх")

_CONS_PHONE = {
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "д": "d",
    "ж": "ʐ",
    "з": "z",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ф": "f",
    "х": "x",
}

def _emit_consonant(ch: str, *, palatal: bool) -> str:
    if ch == "ч":
        return "tɕ"
    if ch == "щ":
        return "ɕː"
    if ch == "ц":
        return "ts"
    if ch in "жш":
        return _CONS_PHONE[ch]
    base = _CONS_PHONE.get(ch, "")
    if not base:
        return ""
    if palatal and ch in _PALATALIZABLE:
        return base + "ʲ"
    return base


def _vowel_ipa(
    ch: str,
    *,
    stressed: bool,
    after_palatal: bool,
    after_hard_consonant: bool,
    word_initial_jot: bool,
) -> str:
    """Map single vowel letter to IPA (no jot prefix — handle ``е``/``ю``/``я`` separately)."""
    cl = ch.lower()
    if cl == "а":
        return "a" if stressed else "ə"
    if cl == "о":
        return "o" if stressed else "ə"
    if cl == "у":
        return "u"
    if cl == "ы":
        return "ɨ"
    if cl == "э":
        return "ɛ"
    if cl == "и":
        return "i" if stressed else "ɪ"
    if cl == "ё":
        return "o" if stressed else "ə"
    if cl == "е":
        if word_initial_jot:
            return "e"
        if after_palatal:
            return "e" if stressed else "ɪ"
        if after_hard_consonant:
            return "ɛ" if stressed else "ɪ"
        return "je" if stressed else "jɪ"
    if cl == "ю":
        if word_initial_jot:
            return "u"
        if after_palatal:
            return "u"
        return "ʊ" if not stressed else "u"
    if cl == "я":
        if word_initial_jot:
            return "a"
        if after_palatal:
            return "a" if stressed else "ə"
        if after_hard_consonant:
            return "a" if stressed else "ə"
        return "a" if stressed else "jə"
    return ""


def _letters_to_ipa_rules(word: str, *, stress_syl: int) -> str:
    """
    Map a stress-mark-free lowercase Cyrillic word (no hyphen) to IPA without stress marks.
    """
    w = re.sub(r"[^а-яё]", "", word.lower())
    if not w:
        return ""
    pos_syl = _char_syllable_indices(w)
    if len(pos_syl) != len(w):
        pos_syl = [0] * len(w)

    out: list[str] = []
    i = 0
    n = len(w)
    after_vowel = False

    def after_palatal_consonant() -> bool:
        if not out:
            return False
        last = out[-1]
        if last in ("tɕ", "ɕː", "ts", "ʐ", "ʂ"):
            return False
        return last.endswith("ʲ")

    def after_hard_consonant_ipa() -> bool:
        if not out:
            return False
        last = out[-1]
        if last.endswith("ʲ"):
            return False
        if last[-1] in "aeiouɛəɨɪʊ":
            return False
        return True

    while i < n:
        ch = w[i]
        syl_here = pos_syl[i] if i < len(pos_syl) else 0
        stressed = syl_here == stress_syl

        if ch in "ъ":
            i += 1
            continue

        if ch in "ь":
            i += 1
            continue

        if ch == "й":
            out.append("j")
            i += 1
            after_vowel = False
            continue

        if _is_vowel_letter(ch):
            jot = len(out) == 0 or after_vowel
            after_palatal = after_palatal_consonant()
            after_hard = after_hard_consonant_ipa()
            ve = ch.lower()
            if ve == "е" and jot:
                out.append("je" if stressed else "jɪ")
                i += 1
                after_vowel = True
                continue
            if ve == "ю" and jot:
                out.append("ju")
                i += 1
                after_vowel = True
                continue
            if ve == "я" and jot:
                out.append("ja" if stressed else "jə")
                i += 1
                after_vowel = True
                continue
            ipa_v = _vowel_ipa(
                ch,
                stressed=stressed,
                after_palatal=after_palatal,
                after_hard_consonant=after_hard,
                word_initial_jot=jot,
            )
            out.append(ipa_v)
            i += 1
            after_vowel = True
            continue

        if ch.lower() not in _CONS_PHONE and ch not in "цчщ":
            i += 1
            continue

        palatal = False
        j = i + 1
        if j < n and w[j] == "ь":
            palatal = ch in _PALATALIZABLE
            out.append(_emit_consonant(ch, palatal=palatal))
            i = j + 1
            after_vowel = False
            continue

        if j < n and w[j] == "ъ":
            out.append(_emit_consonant(ch, palatal=False))
            i = j + 1
            after_vowel = False
            continue

        if j < n and _is_vowel_letter(w[j]):
            v = w[j]
            if ch in "жцш" and v.lower() in "еёиюя":
                palatal = False
            elif ch in "чщ":
                palatal = False
            elif v.lower() in "еёиюя":
                palatal = ch in _PALATALIZABLE
            out.append(_emit_consonant(ch, palatal=palatal))
            i += 1
            after_vowel = False
            continue

        out.append(_emit_consonant(ch, palatal=False))
        i += 1
        after_vowel = False

    return "".join(out)


def _insert_primary_stress_before_vowel(ipa: str) -> str:
    s = ipa.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
    for j, ch in enumerate(s):
        if ch in "aeiouɛəɨɪʊøɵ":
            return s[:j] + _PRIMARY_STRESS + s[j:]
    return _PRIMARY_STRESS + s


def _rules_word_to_ipa(word: str, *, with_stress: bool) -> str:
    raw = word.strip()
    stress_src = _filter_russian_graphemes_keep_stress(raw)
    w = _strip_grapheme_diacritics(stress_src)
    w = re.sub(r"[^а-яё\-]", "", w)
    if not w:
        return ""
    if "-" in w:
        w_chunks = [c for c in w.split("-") if c]
        stress_chunks = stress_src.split("-")
        parts: list[str] = []
        if len(stress_chunks) == len(w_chunks):
            for wc, sc in zip(w_chunks, stress_chunks):
                parts.append(_rules_word_to_ipa_single(wc, sc, with_stress=with_stress))
        else:
            for wc in w_chunks:
                parts.append(_rules_word_to_ipa_single(wc, stress_src, with_stress=with_stress))
        return "-".join(parts)
    return _rules_word_to_ipa_single(w, stress_src, with_stress=with_stress)


def _rules_word_to_ipa_single(w_clean: str, stress_source: str, *, with_stress: bool) -> str:
    syls = russian_orthographic_syllables(w_clean)
    stress_syl = _stress_syllable_index(syls, stress_source)
    body = _letters_to_ipa_rules(w_clean, stress_syl=stress_syl)
    if with_stress and body:
        body = _insert_primary_stress_before_vowel(body)
    return body


def _finalize_word_ipa(
    ipa: str,
    *,
    with_stress: bool,
    vocoder_stress: bool,
) -> str:
    if not with_stress:
        return ipa.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
    if vocoder_stress:
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
    Single-token G2P: lexicon lookup (normalized key), else rules fallback.

    If *vocoder_stress* is True (default), :func:`normalize_ipa_stress_for_vocoder` is applied
    whenever stress marks are kept.

    Pure digit strings expand via :func:`russian_numbers.expand_cardinal_digits_to_russian_words`
    when *expand_cardinal_digits* is True.
    """
    if not word or not word.strip():
        return ""
    raw = word.strip()

    if expand_cardinal_digits and raw.isdigit():
        phrase = expand_cardinal_digits_to_russian_words(raw)
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
        return _finalize_word_ipa(ipa, with_stress=with_stress, vocoder_stress=vocoder_stress)

    if "-" in letters_only:
        chunks = letters_only.split("-")
        sub = [_lookup_lexicon(c, lex) for c in chunks if c]
        if all(sub):
            merged = "-".join(sub)
            return _finalize_word_ipa(merged, with_stress=with_stress, vocoder_stress=vocoder_stress)

    return _finalize_word_ipa(
        _rules_word_to_ipa(raw, with_stress=with_stress),
        with_stress=with_stress,
        vocoder_stress=vocoder_stress,
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
        elif re.fullmatch(r"[\w\-]+", tok, flags=re.UNICODE) and re.search(r"[а-яА-ЯёЁ]", tok):
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
        elif re.fullmatch(r"[\w\-]+", tok, flags=re.UNICODE):
            parts.append(tok)
        else:
            parts.append(tok)
    out = "".join(parts)
    return re.sub(r" +", " ", out).strip()


def coarse_ipa_for_compare(s: str) -> str:
    """
    Strip stress, syllable dots, and combining marks; normalize eSpeak ``ru`` vs ipa-dict
    conventions for loose agreement (verification / regression checks).
    """
    t = unicodedata.normalize("NFD", s)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = t.replace(_PRIMARY_STRESS, "").replace(_SECONDARY_STRESS, "")
    t = t.replace(".", "").replace("͡", "")
    t = t.replace("ɭ", "l")
    t = t.replace("ɑ", "a")
    t = t.replace("ʌ", "ə")
    t = t.replace("ɐ", "ə")
    t = t.replace("ɫ", "l")
    t = t.replace("ɪ", "i")
    t = t.replace("ɨ", "i")
    t = t.replace("ɵ", "o")
    t = t.replace("ʂ", "ʃ")
    t = t.replace("ʐ", "ʒ")
    return t.lower()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Russian text to IPA using data/ru/dict.tsv plus rules for OOV."
    )
    p.add_argument("text", nargs="*", help="Russian text (if empty, read stdin)")
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
        help="Keep ˈ/ˌ before the first segment of the stressed syllable (lexicon-style); "
        "default moves them before the vowel for vocoders.",
    )
    p.add_argument("--stdin", action="store_true", help="Read full text from stdin.")
    p.add_argument(
        "--no-expand-digits",
        action="store_true",
        help="Leave digit sequences as digits (no spoken Russian cardinal expansion).",
    )
    p.add_argument(
        "--no-espeak",
        action="store_true",
        help="Do not print eSpeak NG reference line.",
    )
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
    expand_digits = not args.no_expand_digits
    es_in = expand_digit_tokens_in_text(raw) if expand_digits else raw
    print(
        text_to_ipa(
            raw,
            lexicon=load_russian_lexicon(args.dict) if args.dict is not None else None,
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
