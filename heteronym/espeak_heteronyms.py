"""
Identify CMUdict-ambiguous tokens in text and recover eSpeak NG IPA for each token.

Uses ``espeak-phonemizer`` (libespeak-ng) plus a full-sentence vs. word-removed diff
to isolate the IPA block eSpeak assigned to each heteronym. Results are aligned to
CMUdict alternative strings via ``match_dictionary_alternative``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher

try:
    from espeak_phonemizer import Phonemizer as EspeakPhonemizer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Missing package `espeak-phonemizer` (ctypes bindings to libespeak-ng). "
        "Install with: pip install espeak-phonemizer"
    ) from e

from cmudict_ipa import CmudictIpa, normalize_word_for_lookup

WORD_RE = re.compile(r"\S+")


def espeak_ipa_tokens(
    phonemizer: EspeakPhonemizer,
    text: str,
    *,
    voice: str,
) -> list[str]:
    """
    IPA phone tokens via ``espeak_phonemizer`` (same tokenization as
    ``espeak-ng --ipa`` with a space phoneme separator).
    """
    t = text.strip()
    if not t:
        return []
    try:
        raw = phonemizer.phonemize(
            t,
            voice=voice,
            phoneme_separator=" ",
            word_separator=" ",
        ).strip()
    except (AssertionError, OSError):
        return []
    if not raw:
        return []
    return [x for x in raw.split() if x]


def longest_insert_block(tokens_without: list[str], tokens_full: list[str]) -> list[str]:
    sm = SequenceMatcher(a=tokens_without, b=tokens_full, autojunk=False)
    best: list[str] = []
    best_len = 0
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "insert" and (j2 - j1) > best_len:
            best = tokens_full[j1:j2]
            best_len = j2 - j1
    return best


def normalize_ipa_compare(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def match_dictionary_alternative(extracted_joined: str, alts: list[str]) -> str | None:
    """
    Map eSpeak IPA string to one of the CMUdict IPA strings (canonical spelling
    from ``alts``). Tries exact NFC match, then match with length mark ``ː``
    stripped on both sides (eSpeak often uses long vowels where CMU does not).
    """
    e0 = normalize_ipa_compare(extracted_joined)
    for alt in alts:
        if normalize_ipa_compare(alt) == e0:
            return alt
    e1 = e0.replace("ː", "")
    for alt in alts:
        if normalize_ipa_compare(alt).replace("ː", "") == e1:
            return alt
    return None


@dataclass
class EspeakHeteronymExample:
    char_text: str
    homograph: str
    homograph_wordid: str
    homograph_char_start: int
    homograph_char_end: int


def iter_heteronym_spans_cmudict(
    text: str,
    *,
    cmudict: CmudictIpa,
    max_candidates: int | None,
    ignore_keys: frozenset[str] = frozenset(),
) -> Iterator[tuple[re.Match[str], list[str]]]:
    """Yield (regex match, ipa_alternatives) for each token with 2+ CMU readings.

    If *max_candidates* is ``None``, there is no upper bound on how many readings a
    token may have. Tokens whose normalized lookup key is in *ignore_keys* are skipped.
    """
    for m in WORD_RE.finditer(text):
        key = normalize_word_for_lookup(m.group())
        if key in ignore_keys:
            continue
        alts = cmudict.translate_to_ipa([m.group()])[0][1]
        if len(alts) < 2:
            continue
        if max_candidates is not None and len(alts) > max_candidates:
            continue
        yield m, list(alts)


def sentence_has_ambiguous_heteronym(
    text: str,
    *,
    cmudict: CmudictIpa,
    max_candidates: int | None,
    ignore_keys: frozenset[str] = frozenset(),
) -> bool:
    it = iter_heteronym_spans_cmudict(
        text,
        cmudict=cmudict,
        max_candidates=max_candidates,
        ignore_keys=ignore_keys,
    )
    return next(it, None) is not None


def extract_examples_for_sentence(
    text: str,
    *,
    cmudict: CmudictIpa,
    phonemizer: EspeakPhonemizer,
    voice: str,
    max_candidates: int | None,
    ignore_keys: frozenset[str] = frozenset(),
) -> list[EspeakHeteronymExample]:
    if not text.strip():
        return []

    ambiguous_spans = list(
        iter_heteronym_spans_cmudict(
            text,
            cmudict=cmudict,
            max_candidates=max_candidates,
            ignore_keys=ignore_keys,
        )
    )
    if not ambiguous_spans:
        return []

    tokens_full = espeak_ipa_tokens(phonemizer, text, voice=voice)
    if not tokens_full:
        return []

    out: list[EspeakHeteronymExample] = []
    for m, alts in ambiguous_spans:
        removed = text[: m.start()] + text[m.end() :]
        removed = re.sub(r"  +", " ", removed)
        tokens_without = espeak_ipa_tokens(phonemizer, removed, voice=voice)
        if not tokens_without:
            continue
        block = longest_insert_block(tokens_without, tokens_full)
        if not block:
            continue
        joined = "".join(block)
        wordid = match_dictionary_alternative(joined, alts)
        if wordid is None:
            continue
        out.append(
            EspeakHeteronymExample(
                char_text=text,
                homograph=m.group(),
                homograph_wordid=wordid,
                homograph_char_start=m.start(),
                homograph_char_end=m.end(),
            )
        )
    return out
