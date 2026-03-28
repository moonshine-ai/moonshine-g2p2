#!/usr/bin/env python3
"""
Rule- and lexicon-based German grapheme-to-phoneme (Standard / High German, broad IPA).

* In-vocabulary words are taken from ``models/de/dict.tsv`` (word ``\\t`` IPA, first line wins
  for duplicate spellings).
* Out-of-vocabulary tokens use a compact letter-guessing pass (digraphs, *ch* context,
  final devoicing, *ig*-ending, word-initial *st*/*sp*, etc.) plus rough orthographic
  syllables for default stress.

Limitations (intentional):
- OOV stress is heuristic (suffix *-ung* / *-schaft* / *-tion* / *-ismus* → last syllable;
  else first syllable, with optional stripping of common unstressed verbal prefixes).
- Morpheme-internal *st*/*sp* (e.g. *verstehen*) are not fully modeled; the lexicon covers
  common lemmas.
- *r* is rendered as /ʁ/; *-er* is not forced to [ɐ] in the rules path.
- Loanword *ch* (*Chor*, *Charakter*) is only partly special-cased.

By default, ˈ / ˌ are **shifted to sit before the syllable nucleus** (vocoder- and eSpeak-style).
Pass ``--syllable-initial-stress`` to keep dictionary-style marks at the first segment of the
stressed syllable (strict IPA syllable edge).

CLI: prints rule/lexicon IPA, then an eSpeak NG reference line (``--no-espeak`` to disable),
same stack as ``spanish_rule_g2p`` / ``heteronym.espeak_heteronyms``.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

_DEFAULT_ESPEAK_VOICE = "de"

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT_PATH = _REPO_ROOT / "models" / "de" / "dict.tsv"

_LEXICON_CACHE: dict[str, str] | None = None
_LEXICON_PATH: Path | None = None


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


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def normalize_lookup_key(word: str) -> str:
    """Lowercase NFC key for lexicon lookup (letters, umlauts, ß only)."""
    t = unicodedata.normalize("NFC", word.strip().lower())
    return re.sub(r"[^a-zäöüß]+", "", t)


def load_german_lexicon(path: Path | None = None) -> dict[str, str]:
    """
    Load ``word\\tIPA`` TSV. Keys are NFC-lowercase grapheme keys (see :func:`normalize_lookup_key`).

    When the same key appears from a capitalized row (e.g. English ``Die`` /daɪ/) and later from an
    all-lowercase German lemma (``die`` /diː/), the **lowercase** row wins so common tokens are not
    stuck on the wrong homograph.
    """
    p = path or _DEFAULT_DICT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"German lexicon not found: {p}")
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
    _LEXICON_CACHE = load_german_lexicon(p)
    _LEXICON_PATH = p
    return _LEXICON_CACHE


def _lookup_lexicon(key: str, lexicon: Mapping[str, str]) -> str | None:
    if key in lexicon:
        return lexicon[key]
    if "ß" in key:
        alt = key.replace("ß", "ss")
        if alt in lexicon:
            return lexicon[alt]
    else:
        if "ss" in key:
            alt = key.replace("ss", "ß", 1)
            if alt in lexicon:
                return lexicon[alt]
    return None


_VOWEL = frozenset("aeiouyäöü")
_VOWEL_OR_H = frozenset("aeiouyäöüh")


def _char_before(s: str, i: int) -> str | None:
    j = i - 1
    while j >= 0:
        if s[j] == "-":
            return None
        if s[j] in _VOWEL:
            return s[j]
        if s[j] == "h" and j > 0 and s[j - 1] in _VOWEL:
            return s[j - 1]
        j -= 1
    return None


def _ch_ipa(s: str, i: int) -> str:
    """*ch* after a,o,u (incl. *au*) → /x/, else /ç/ (simplified)."""
    if i > 1 and s[i - 2 : i] == "au":
        return "x"
    prev = _char_before(s, i)
    if prev is None:
        return "ç"
    if prev in "aou":
        return "x"
    return "ç"


def _final_devoice(ipa: str) -> str:
    if not ipa:
        return ipa
    # Word-final obstruent devoicing (German)
    repl = {"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s"}
    last = ipa[-1]
    return ipa[:-1] + repl.get(last, last)


def _st_sp_at_morpheme_start(hyphen_word: str, compact_index: int) -> bool:
    """True at absolute word start or at first letter of a segment after ``-`` (compact index, no hyphens)."""
    if compact_index == 0:
        return True
    pos = 0
    for part in hyphen_word.lower().split("-"):
        if not part:
            continue
        if compact_index == pos:
            return True
        pos += len(part)
    return False


def _unstressed_prefix_len(w: str) -> int:
    for p in (
        "wider",
        "entgegen",
        "ver",
        "zer",
        "miss",
        "ent",
        "emp",
        "ge",
        "be",
        "er",
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
    wl = w.replace("-", "")
    if wl.endswith(("ung", "schaft", "tion", "ismus")):
        return n - 1
    plen = _unstressed_prefix_len(wl)
    if plen > 0:
        acc = 0
        for idx, sy in enumerate(syls):
            acc += len(sy)
            if acc >= plen:
                return min(idx + 1, n - 1)
    return 0


def _german_vowel_nucleus_spans(w: str) -> list[tuple[int, int]]:
    """Inclusive-exclusive spans of vowel nuclei (diphthongs / long vowels as one)."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(w)
    while i < n:
        ch = w[i]
        if ch == "-":
            i += 1
            continue
        if ch not in _VOWEL:
            i += 1
            continue
        start = i
        if i + 1 < n:
            pair = w[i : i + 2]
            if pair in {"au", "ei", "eu", "ai", "äu", "ey", "oi"}:
                spans.append((start, i + 2))
                i += 2
                continue
            if pair == "ie":
                if i + 2 >= n or w[i + 2] == "-" or w[i + 2] not in _VOWEL:
                    spans.append((start, i + 2))
                    i += 2
                    continue
            if ch in "aoeiuäöü" and w[i + 1] == ch:
                spans.append((start, i + 2))
                i += 2
                continue
        spans.append((start, start + 1))
        i += 1
    return spans


def german_orthographic_syllables(word: str) -> list[str]:
    """
    Rough orthographic syllables (maximal onset: consonants between nuclei start the next syllable).
    Hyphen splits compounds before syllabifying each piece.
    """
    w = re.sub(r"[^a-zäöüß-]", "", word.lower())
    if not w:
        return []
    if "-" in w:
        parts: list[str] = []
        for chunk in w.split("-"):
            if chunk:
                parts.extend(german_orthographic_syllables(chunk))
        return parts
    spans = _german_vowel_nucleus_spans(w)
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


def _insert_primary_stress_before_vowel(ipa: str) -> str:
    s = ipa.replace("ˈ", "")
    m = re.search(
        r"aɪ̯|aʊ̯|ɔʏ̯|iː|eː|aː|oː|uː|ɪ|ʊ|[aɛəioɔuyøʏɐ]",
        s,
    )
    if not m:
        return "ˈ" + s
    return s[: m.start()] + "ˈ" + s[m.start() :]


# Primary / secondary stress (modifier letters, NFC)
_PRIMARY_STRESS = "\u02c8"  # ˈ
_SECONDARY_STRESS = "\u02cc"  # ˌ

# Longest first: nuclei (vowel, diphthong, syllabic consonant) for vocoder stress placement.
_IPA_NUCLEUS_PREFIXES: tuple[str, ...] = tuple(
    sorted(
        (
            "aɪ̯",
            "aʊ̯",
            "ɔʏ̯",
            "ɛɪ̯",
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
            "r̩",
            "ə",
            "ɛ",
            "ɜ",
            "ɪ",
            "ʊ",
            "ɐ̯",
            "ɐ",
            "ɨ",
            "ɵ",
            "ø",
            "œ",
            "ʏ",
            "y",
            "ɔ",
            "ɑ",
            "æ",
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

# Consonant letters / affricates (longest first) skipped when moving ˈ/ˌ to the nucleus.
_IPA_CONS_CLUSTER_PREFIXES: tuple[str, ...] = tuple(
    sorted(
        (
            "t͡s",
            "p͡f",
            "d͡ʒ",
            "t͡ʃ",
            "tʃ",
            "ts",
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
    """Advance *j* past one consonantal segment, or return *j* if already at nucleus / stress / end."""
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

    Lexicons often use IPA syllable-edge stress (e.g. ``ˈkɔmə``); many vocoders and eSpeak-style
    IPA use nuclear stress (``kˈɔmə``). Idempotent on already nuclear placement.
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


def _letters_to_ipa_no_stress(
    letters: str,
    *,
    full_word: str,
    hyphen_word: str,
    span_start: int,
) -> str:
    """
    Map *letters* (one syllable chunk, lowercase) to IPA without stress.
    *full_word* is the word without hyphens (for *ch* context). *hyphen_word* retains ``-``
    for compound boundaries. *span_start* is the index of *letters[0]* in *full_word*.
    """
    s = letters
    n = len(s)
    i = 0
    out: list[str] = []

    def global_index(local: int) -> int:
        return span_start + local

    while i < n:
        gi = global_index(i)
        ch = s[i]

        if ch == "-":
            i += 1
            continue

        if i + 3 <= n and s[i : i + 4] == "tsch":
            out.append("tʃ")
            i += 4
            continue

        if i + 2 <= n and s[i : i + 3] == "sch":
            out.append("ʃ")
            i += 3
            continue

        if i + 2 <= n and s[i : i + 3] == "chs":
            out.append("ks")
            i += 3
            continue

        if i + 1 < n and s[i : i + 2] == "ch":
            out.append(_ch_ipa(full_word, gi))
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

        if i + 1 < n and s[i : i + 2] == "pf":
            out.append("pf")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "qu":
            out.append("kv")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "st" and _st_sp_at_morpheme_start(hyphen_word, gi):
            out.append("ʃt")
            i += 2
            continue

        if i + 1 < n and s[i : i + 2] == "sp" and _st_sp_at_morpheme_start(hyphen_word, gi):
            out.append("ʃp")
            i += 2
            continue

        if ch == "h" and i + 1 < n and s[i + 1] in _VOWEL:
            i += 1
            continue

        if ch == "h":
            i += 1
            continue

        if ch == "ß":
            out.append("s")
            i += 1
            continue

        if i + 1 < n and s[i : i + 2] == "tz":
            out.append("ts")
            i += 2
            continue

        if ch == "z":
            out.append("ts")
            i += 1
            continue

        if ch == "c" and i + 1 < n and s[i + 1] == "k":
            out.append("k")
            i += 2
            continue

        if ch == "c" and i + 1 < n and s[i + 1] in "ei":
            out.append("ts")
            i += 2
            continue

        if ch == "c":
            out.append("k")
            i += 1
            continue

        if ch == "x":
            out.append("ks")
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

        if ch == "v":
            out.append("f")
            i += 1
            continue

        if ch == "w":
            out.append("v")
            i += 1
            continue

        if ch == "y" and (i + 1 >= n or s[i + 1] not in _VOWEL):
            out.append("ʏ")
            i += 1
            continue

        if i + 1 < n and s[i : i + 2] == "au":
            out.append("aʊ̯")
            i += 2
            continue
        if i + 1 < n and s[i : i + 2] in {"ei", "ai", "ey"}:
            out.append("aɪ̯")
            i += 2
            continue
        if i + 1 < n and s[i : i + 2] in {"eu", "äu"}:
            out.append("ɔʏ̯")
            i += 2
            continue
        if i + 1 < n and s[i : i + 2] == "ie" and (i + 2 >= n or s[i + 2] not in _VOWEL):
            out.append("iː")
            i += 2
            continue

        if i + 1 < n and s[i] in _VOWEL and s[i + 1] == s[i] and s[i] in "aoeiu":
            long_map = {"a": "aː", "e": "eː", "i": "iː", "o": "oː", "u": "uː"}
            out.append(long_map.get(s[i], s[i]))
            i += 2
            continue

        if ch in _VOWEL:
            if ch == "a":
                out.append("a")
            elif ch == "e":
                out.append("ə" if i == n - 1 else "ɛ")
            elif ch == "i":
                out.append("ɪ")
            elif ch == "o":
                out.append("ɔ")
            elif ch == "u":
                out.append("ʊ")
            elif ch == "ä":
                out.append("ɛ")
            elif ch == "ö":
                out.append("ø")
            elif ch == "ü":
                out.append("ʏ")
            elif ch == "y":
                out.append("ʏ")
            else:
                out.append(ch)
            i += 1
            continue

        if ch == "r":
            out.append("ʁ")
            i += 1
            continue

        if ch == "s" and i + 1 < n and s[i + 1] == "s":
            out.append("s")
            i += 2
            continue

        if ch == "s":
            prev_v = i > 0 and s[i - 1] in _VOWEL
            next_v = i + 1 < n and s[i + 1] in _VOWEL
            out.append("z" if prev_v and next_v else "s")
            i += 1
            continue

        simple = {
            "b": "b",
            "d": "d",
            "f": "f",
            "g": "ɡ",
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

        i += 1

    ipa = "".join(out)
    stem = letters.rstrip("-")
    if stem.endswith("ig") and len(stem) >= 2 and not stem.endswith("lich"):
        if ipa.endswith("ɡ"):
            ipa = ipa[:-1] + "ç"
    return _final_devoice(ipa)


def _rules_word_to_ipa(word: str, *, with_stress: bool) -> str:
    w = re.sub(r"[^a-zäöüßA-ZÄÖÜß-]", "", word)
    if not w:
        return ""
    wl = w.lower()
    wl_nh = wl.replace("-", "")
    syls = german_orthographic_syllables(wl)
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
) -> str:
    """
    Single-token G2P: lexicon lookup (normalized key), else rules fallback.

    If *vocoder_stress* is True (default), :func:`normalize_ipa_stress_for_vocoder` is applied
    whenever stress marks are kept.
    """
    if not word or not word.strip():
        return ""
    raw = word.strip()
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
) -> str:
    """Tokenize and G2P each word; preserve punctuation and collapse spaces."""
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
                )
            )
        else:
            parts.append(tok)
    out = "".join(parts)
    return re.sub(r" +", " ", out).strip()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="German text to IPA using models/de/dict.tsv plus rules for OOV."
    )
    p.add_argument("text", nargs="*", help="German text (if empty, read stdin)")
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
    lex = load_german_lexicon(args.dict) if args.dict is not None else None
    print(
        text_to_ipa(
            raw,
            lexicon=lex,
            dict_path=args.dict,
            with_stress=not args.no_stress,
            vocoder_stress=not args.syllable_initial_stress,
        )
    )
    if not args.no_espeak:
        es = espeak_ng_ipa_line(raw, voice=args.espeak_voice)
        if es is not None:
            print(f"{es} (espeak-ng)")


if __name__ == "__main__":
    main()
