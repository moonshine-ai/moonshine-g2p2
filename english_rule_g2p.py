#!/usr/bin/env python3
"""
Rule- and lexicon-based **English (US) grapheme-to-phoneme** (broad IPA).

* **In-vocabulary** words use ``models/en_us/dict_filtered_heteronyms.tsv``
  (``word<TAB>ipa``). For **single-word** :meth:`~EnglishLexiconRuleG2p.g2p`, the
  default is the first IPA after merging CMUdict with
  ``heteronym/homograph_index.json`` candidate order (when present), else sorted
  CMU order. Use :meth:`~EnglishLexiconRuleG2p.g2p_span` with sentence context to
  apply lightweight heteronym heuristics (``english_heteronym_heuristics``).
* **Out-of-vocabulary** words use :class:`moonshine_onnx_g2p.OnnxOovG2p` when
  ``models/en_us/oov/model.onnx`` is present and ``onnxruntime`` loads (same default
  path as ``MoonshineOnnxG2P``). If the model is missing or inference fails, falls
  back to :func:`english_oov_rules_ipa` (greedy grapheme rules + primary stress).
  Pass ``use_onnx_oov=False`` to keep the hand-rule path only.

Compare with eSpeak NG (``en-us``) and run ``scripts/eval_english_g2p_metrics.py`` /
``scripts/eval_english_oov_handling.py`` for benchmarks.

CLI prints rule/lexicon IPA, then an eSpeak reference line when available.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from cmudict_ipa import normalize_word_for_lookup

from english_heteronym_heuristics import (
    context_neighbor_words,
    disambiguate_heteronym_ipa,
    load_homograph_ordered_ipa,
    merge_tsv_and_homograph_candidates,
)

_DEFAULT_ESPEAK_VOICE = "en-us"
_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_DICT_PATH = _REPO_ROOT / "models" / "en_us" / "dict_filtered_heteronyms.tsv"
_DEFAULT_OOV_PHONEME_VOCAB = _REPO_ROOT / "models" / "en_us" / "oov" / "phoneme_vocab.json"
_DEFAULT_OOV_ONNX = _REPO_ROOT / "models" / "en_us" / "oov" / "model.onnx"
_DEFAULT_HOMOGRAPH_INDEX = _REPO_ROOT / "models" / "en_us" / "heteronym" / "homograph_index.json"


def load_oov_phoneme_vocab_tokens(path: str | Path | None = None) -> frozenset[str]:
    """Phoneme symbols from ``phoneme_vocab.json`` (no ``<...>`` specials)."""
    p = Path(path) if path else _DEFAULT_OOV_PHONEME_VOCAB
    stoi: dict[str, int] = json.loads(p.read_text(encoding="utf-8"))
    return frozenset(t for t in stoi if not t.startswith("<"))


def segment_ipa_with_vocab(ipa: str, vocab: frozenset[str] | set[str]) -> list[str]:
    """
    Greedy longest-prefix segmentation of *ipa* using *vocab* tokens (aligns with ONNX OOV phones).
    """
    s = unicodedata.normalize("NFC", (ipa or "").strip())
    if not s:
        return []
    toks = sorted((t for t in vocab if t), key=len, reverse=True)
    i = 0
    out: list[str] = []
    while i < len(s):
        for t in toks:
            if t and s.startswith(t, i):
                out.append(t)
                i += len(t)
                break
        else:
            out.append(s[i])
            i += 1
    return out


_LEXICON_CACHE: dict[str, str] | None = None
_LEXICON_PATH: Path | None = None

# Small set of function words where naive vowel rules misfire.
_FUNCTION_WORD_IPA: dict[str, str] = {
    "the": "ðə",
    "a": "ə",
    "an": "æn",
    "to": "tə",
    "of": "əv",
    "and": "ænd",
    "or": "ɔɹ",
    "are": "ɑɹ",
    "was": "wəz",
    "were": "wɝ",
    "from": "fɹʌm",
    "have": "hæv",
    "has": "hæz",
    "been": "bɪn",
    "do": "du",
    "does": "dʌz",
    "your": "jɔɹ",
    "you": "ju",
    "they": "ðeɪ",
    "their": "ðɛɹ",
    "there": "ðɛɹ",
}


def espeak_ng_ipa_line(text: str, *, voice: str = _DEFAULT_ESPEAK_VOICE) -> str | None:
    """Word- or phrase-level IPA from libespeak-ng (same policy as ``german_rule_g2p``)."""
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
    return raw.strip() if raw else None


def _normalize_grapheme_key(word_token: str) -> str:
    s = word_token.lower()
    if s.endswith(")") and "(" in s:
        i = s.rfind("(")
        mid = s[i + 1 : -1]
        if mid.isdigit():
            s = s[:i]
    return s


def _load_tsv_all_ipas(path: Path) -> dict[str, list[str]]:
    """``word_key`` -> sorted unique IPA strings (one row per pronunciation)."""
    raw: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                continue
            word_token, ipa = line.split("\t", 1)
            word_token = word_token.strip()
            ipa = ipa.strip()
            if not word_token or not ipa:
                continue
            key = _normalize_grapheme_key(word_token)
            raw[key].add(ipa)
    return {k: sorted(v) for k, v in raw.items()}


def _first_ipa_per_key(
    multi: dict[str, list[str]],
    homograph_order: dict[str, list[str]],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, ipas in multi.items():
        merged = merge_tsv_and_homograph_candidates(k, ipas, homograph_order)
        if merged:
            out[k] = merged[0]
    return out


def load_english_lexicon(path: Path | None = None) -> dict[str, str]:
    """Single IPA per key (heteronym prior order when ``homograph_index.json`` exists)."""
    p = path or _DEFAULT_DICT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"English lexicon not found: {p}")
    multi = _load_tsv_all_ipas(p)
    ho = load_homograph_ordered_ipa(_DEFAULT_HOMOGRAPH_INDEX)
    return _first_ipa_per_key(multi, ho)


def _get_lexicon(path: Path | None) -> dict[str, str]:
    global _LEXICON_CACHE, _LEXICON_PATH
    p = path or _DEFAULT_DICT_PATH
    if _LEXICON_CACHE is not None and _LEXICON_PATH == p:
        return _LEXICON_CACHE
    if not p.is_file():
        _LEXICON_CACHE = {}
        _LEXICON_PATH = p
        return _LEXICON_CACHE
    _LEXICON_CACHE = load_english_lexicon(p)
    _LEXICON_PATH = p
    return _LEXICON_CACHE


_VOWELS = frozenset("aeiouy")
_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")


def _is_vowel(c: str) -> bool:
    return c in _VOWELS


def _next_vowel_index(w: str, start: int) -> int | None:
    j = start
    while j < len(w):
        if _is_vowel(w[j]):
            return j
        j += 1
    return None


def _magic_e_lengthens(w: str, vowel_i: int) -> bool:
    """True if there is a single consonant between vowel at *vowel_i* and final silent e."""
    if vowel_i < 0 or vowel_i >= len(w):
        return False
    if not w.endswith("e") or len(w) < vowel_i + 3:
        return False
    # ...V C e  (at least one char between V and final e)
    j = vowel_i + 1
    if j >= len(w) - 1:
        return False
    if not (w[-1] == "e" and w[-2].isalpha() and w[-2] not in "aeiou"):
        return False
    mid = w[j : -1]
    if not mid or any(c in "aeiou" for c in mid):
        return False
    return len(mid) == 1


def _r_controlled(w: str, i: int) -> tuple[str, int] | None:
    """If vowel at *i* is followed by 'r', return (ipa, consume_len)."""
    if i + 1 >= len(w) or w[i + 1] != "r":
        return None
    v = w[i]
    if v == "a":
        return "ɑɹ", 2
    if v == "e":
        return "ɛɹ", 2
    if v == "i":
        return "ɪɹ", 2
    if v == "o":
        return "ɔɹ", 2
    if v == "u":
        return "ʊɹ", 2
    if v == "y":
        return "aɪɹ", 2
    return None


# Longest-match grapheme → IPA (OOV). Order matters only within same length bucket.
_OOV_LITERALS: list[tuple[str, str]] = sorted(
    [
        ("tch", "tʃ"),
        ("dge", "dʒ"),
        ("tion", "ʃən"),
        ("sion", "ʒən"),
        ("sure", "ʒɚ"),
        ("ture", "tʃɚ"),
        ("ough", "oʊ"),
        ("augh", "ɔː"),
        ("eigh", "eɪ"),
        ("igh", "aɪ"),
        ("oar", "ɔɹ"),
        ("our", "aʊɹ"),
        ("oor", "ɔɹ"),
        ("ear", "ɪɹ"),
        ("eer", "ɪɹ"),
        ("ier", "ɪɹ"),
        ("air", "ɛɹ"),
        ("are", "ɛɹ"),
        ("ire", "aɪɹ"),
        ("ure", "jʊɹ"),
        ("ai", "eɪ"),
        ("ay", "eɪ"),
        ("au", "ɔː"),
        ("aw", "ɔː"),
        ("ea", "iː"),
        ("ee", "iː"),
        ("ei", "eɪ"),
        ("ey", "eɪ"),
        ("eu", "juː"),
        ("ew", "juː"),
        ("ie", "iː"),
        ("oa", "oʊ"),
        ("oe", "oʊ"),
        ("oi", "ɔɪ"),
        ("oy", "ɔɪ"),
        ("oo", "uː"),
        ("ou", "aʊ"),
        ("ow", "oʊ"),
        ("ph", "f"),
        ("gh", ""),  # often silent after vowels; refined below
        ("ng", "ŋ"),
        ("ch", "tʃ"),
        ("sh", "ʃ"),
        ("th", "θ"),
        ("wh", "w"),
        ("qu", "kw"),
        ("ck", "k"),
        ("sch", "sk"),
        ("ss", "s"),
        ("ll", "l"),
        ("mm", "m"),
        ("nn", "n"),
        ("ff", "f"),
        ("pp", "p"),
        ("tt", "t"),
        ("zz", "z"),
        ("rr", "ɹ"),
        ("dd", "d"),
        ("bb", "b"),
        ("gg", "ɡ"),
    ],
    key=lambda x: len(x[0]),
    reverse=True,
)


def _oov_single_consonant(c: str, w: str, i: int) -> str:
    if c == "c":
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if nxt in "eiy":
            return "s"
        return "k"
    if c == "g":
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if nxt in "eiy":
            return "dʒ"
        return "ɡ"
    if c == "j":
        return "dʒ"
    if c == "q":
        return "k"
    if c == "x":
        return "ks"
    if c == "y":
        # y as consonant at start before vowel
        if i == 0 and _next_vowel_index(w, 1) is not None:
            return "j"
        return "aɪ"
    if c == "r":
        return "ɹ"
    if c == "h":
        return "h"
    return {
        "b": "b",
        "d": "d",
        "f": "f",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "p": "p",
        "s": "s",
        "t": "t",
        "v": "v",
        "w": "w",
        "z": "z",
    }.get(c, c)


def _oov_vowel(w: str, i: int) -> tuple[str, int]:
    v = w[i]
    rc = _r_controlled(w, i)
    if rc:
        return rc

    magic = _magic_e_lengthens(w, i)
    nxt_c = _next_vowel_index(w, i + 1)
    closed = False
    if nxt_c is not None:
        between = w[i + 1 : nxt_c]
        closed = bool(between) and all(c not in "aeiou" for c in between)
    elif i + 1 < len(w) and w[i + 1] not in "aeiou":
        closed = True

    if v == "a":
        if magic:
            return "eɪ", 1
        if closed:
            return "æ", 1
        return "ɑː", 1
    if v == "e":
        if magic:
            return "iː", 1
        if closed or (i == len(w) - 1):
            return "ɛ", 1
        return "iː", 1
    if v == "i":
        if magic:
            return "aɪ", 1
        if closed:
            return "ɪ", 1
        return "aɪ", 1
    if v == "o":
        if magic:
            return "oʊ", 1
        if closed:
            return "ɒ", 1
        return "oʊ", 1
    if v == "u":
        if magic:
            return "juː", 1
        if closed:
            return "ʌ", 1
        return "uː", 1
    if v == "y":
        if closed:
            return "ɪ", 1
        return "aɪ", 1
    return "ə", 1


def _oov_grapheme_to_ipa(word: str) -> str:
    w = normalize_word_for_lookup(word)
    if not w:
        return ""
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w:
        return ""
    if w in _FUNCTION_WORD_IPA:
        return _FUNCTION_WORD_IPA[w]

    # Word-initial th voiced for function words
    if w == "this" or w == "that" or w == "then" or w == "than" or w == "they":
        pass

    parts: list[str] = []
    i = 0
    n = len(w)
    while i < n:
        if w[i] == "e" and i == n - 1 and parts:
            i += 1
            continue
        matched = False
        for lit, ipa in _OOV_LITERALS:
            L = len(lit)
            if L and w.startswith(lit, i):
                if lit == "gh":
                    prev = parts[-1] if parts else ""
                    if prev and prev[-1] in "aæeɛiɪoɔuʊ":
                        i += 2
                        matched = True
                        break
                    parts.append("ɡ")
                    i += 2
                    matched = True
                    break
                if lit == "th":
                    if w in ("the", "this", "that", "they", "then", "than", "there", "these", "those"):
                        parts.append("ð")
                    else:
                        parts.append("θ")
                    i += 2
                    matched = True
                    break
                parts.append(ipa)
                i += L
                matched = True
                break
        if matched:
            continue
        c = w[i]
        if _is_vowel(c):
            frag, adv = _oov_vowel(w, i)
            parts.append(frag)
            i += adv
            continue
        if c in _CONSONANTS:
            parts.append(_oov_single_consonant(c, w, i))
            i += 1
            continue
        i += 1
    return "".join(parts)


_VOWEL_IPA_PREFIXES = sorted(
    [
        "aɪ",
        "aʊ",
        "eɪ",
        "oʊ",
        "ɔɪ",
        "juː",
        "iː",
        "uː",
        "ɑː",
        "ɔː",
        "ɜː",
        "ɛɹ",
        "ɑɹ",
        "ɔɹ",
        "ɪɹ",
        "ʊɹ",
        "aɪɹ",
        "ɪə",
        "eə",
        "ʊə",
        "iə",
        "ə",
        "ɪ",
        "ɛ",
        "æ",
        "ʌ",
        "ʊ",
        "ɑ",
        "ɔ",
        "i",
        "u",
        "e",
        "o",
        "ɚ",
        "ɝ",
        "ɒ",
    ],
    key=len,
    reverse=True,
)


def _add_primary_stress_if_missing(ipa: str) -> str:
    s = ipa
    if not s:
        return s
    if s[0] in "ˈˌ":
        return s
    for pref in _VOWEL_IPA_PREFIXES:
        k = s.find(pref)
        if k != -1:
            return s[:k] + "ˈ" + s[k:]
    return "ˈ" + s


def english_oov_rules_ipa(word: str) -> str:
    """OOV-only IPA (no lexicon). Adds a single ˈ before the first vowel if absent."""
    raw = _oov_grapheme_to_ipa(word)
    return _add_primary_stress_if_missing(raw)


class EnglishLexiconRuleG2p:
    """
    Dictionary lookup (normalized word key), then ONNX OOV or :func:`english_oov_rules_ipa`.

    Pass ``heteronym_index_path`` (default: repo ``homograph_index.json``) so
    single-word defaults follow heteronym candidate order; use :meth:`g2p_span`
    for contextual disambiguation when a key has multiple CMUdict IPA lines.
    """

    def __init__(
        self,
        lexicon: Mapping[str, str] | None = None,
        *,
        dict_path: Path | None = None,
        heteronym_index_path: Path | None = None,
        use_onnx_oov: bool = True,
        oov_onnx_path: Path | None = None,
        oov_use_cuda: bool = False,
    ) -> None:
        dp = dict_path or _DEFAULT_DICT_PATH
        self._homograph_order = load_homograph_ordered_ipa(
            heteronym_index_path if heteronym_index_path is not None else _DEFAULT_HOMOGRAPH_INDEX
        )
        if lexicon is not None:
            self._lex = dict(lexicon)
            self._lex_multi = {k: [v] for k, v in lexicon.items()}
        else:
            if not dp.is_file():
                self._lex = {}
                self._lex_multi = {}
            else:
                self._lex_multi = _load_tsv_all_ipas(dp)
                self._lex = _first_ipa_per_key(self._lex_multi, self._homograph_order)

        self._oov_onnx = None
        if use_onnx_oov:
            onnx_p = Path(oov_onnx_path) if oov_onnx_path is not None else _DEFAULT_OOV_ONNX
            if onnx_p.is_file():
                try:
                    from moonshine_onnx_g2p import OnnxOovG2p

                    self._oov_onnx = OnnxOovG2p(onnx_p, use_cuda=oov_use_cuda)
                except Exception:
                    self._oov_onnx = None

    def _oov_ipa(self, grapheme_key: str, rules_fallback_surface: str) -> str:
        """OOV transcription: ONNX phoneme join, else :func:`english_oov_rules_ipa`."""
        if self._oov_onnx is not None and grapheme_key:
            try:
                toks = self._oov_onnx.predict_phonemes(grapheme_key)
                if toks:
                    return "".join(toks)
            except Exception:
                pass
        base = grapheme_key if grapheme_key else rules_fallback_surface
        return english_oov_rules_ipa(base)

    @classmethod
    def from_default_paths(cls) -> EnglishLexiconRuleG2p:
        return cls()

    def lookup_only(self, word: str) -> str | None:
        key = normalize_word_for_lookup(word)
        if not key:
            return None
        gk = _normalize_grapheme_key(key)
        ipa = self._lex.get(gk)
        return ipa

    def pronunciation_candidates(self, word: str) -> list[str]:
        """All CMUdict IPA strings for *word* (heteronym merge order when applicable)."""
        key = normalize_word_for_lookup(word)
        if not key:
            return []
        gk = _normalize_grapheme_key(key)
        ipas = self._lex_multi.get(gk, [])
        if not ipas:
            return []
        return merge_tsv_and_homograph_candidates(gk, ipas, self._homograph_order)

    def g2p(self, word: str) -> str:
        hit = self.lookup_only(word)
        if hit is not None:
            return hit
        key = normalize_word_for_lookup(word)
        gk = _normalize_grapheme_key(key) if key else ""
        return self._oov_ipa(gk, word)

    def g2p_span(self, text: str, span_s: int, span_e: int) -> str:
        """
        Transcribe the surface substring ``text[span_s:span_e]`` using lexicon + OOV rules.

        When the lexicon lists multiple pronunciations for that key, uses
        :func:`english_heteronym_heuristics.disambiguate_heteronym_ipa` with
        neighboring word tokens.
        """
        if span_s < 0 or span_e <= span_s or span_e > len(text):
            return ""
        raw = text[span_s:span_e]
        key = _normalize_grapheme_key(normalize_word_for_lookup(raw))
        if not key:
            return self._oov_ipa("", raw)
        ipas = self._lex_multi.get(key)
        if not ipas:
            return self._oov_ipa(key, raw)
        cands = merge_tsv_and_homograph_candidates(key, ipas, self._homograph_order)
        if not cands:
            return self._oov_ipa(key, raw)
        if len(cands) == 1:
            return cands[0]
        left, right = context_neighbor_words(text, span_s, span_e, window=8)
        return disambiguate_heteronym_ipa(key, cands, left, right, default_primary=cands[0])

    def g2p_phoneme_tokens(self, word: str, *, vocab: frozenset[str]) -> list[str]:
        """Segment :meth:`g2p` IPA with the OOV phoneme vocabulary (same symbols as ONNX)."""
        ipa = self.g2p(word)
        return segment_ipa_with_vocab(ipa, vocab)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("words", nargs="+", help="words to transcribe")
    p.add_argument(
        "--dict-tsv",
        type=Path,
        default=_DEFAULT_DICT_PATH,
        help="CMU-style TSV",
    )
    p.add_argument("--no-espeak", action="store_true")
    p.add_argument("--espeak-voice", default=_DEFAULT_ESPEAK_VOICE)
    p.add_argument(
        "--no-oov-onnx",
        action="store_true",
        help="use hand OOV rules only (no OnnxOovG2p)",
    )
    p.add_argument(
        "--oov-onnx",
        type=Path,
        default=None,
        help=f"path to OOV model.onnx (default: {_DEFAULT_OOV_ONNX})",
    )
    args = p.parse_args(argv)

    g2p = EnglishLexiconRuleG2p(
        dict_path=args.dict_tsv,
        use_onnx_oov=not args.no_oov_onnx,
        oov_onnx_path=args.oov_onnx,
    )
    for w in args.words:
        ipa = g2p.g2p(w)
        print(f"{w}\t{ipa}")
        if not args.no_espeak:
            line = espeak_ng_ipa_line(w, voice=args.espeak_voice)
            if line:
                parts = [x for x in line.split() if x]
                ref = parts[0] if len(parts) == 1 else line
                print(f"  espeak-ng\t{ref}")


if __name__ == "__main__":
    main()
