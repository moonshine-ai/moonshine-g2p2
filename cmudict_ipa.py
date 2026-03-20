"""
Load CMU Pronouncing Dictionary entries and expose IPA pronunciation alternatives.

Dictionary source format matches
https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict
(word token, then ARPAbet phones; alternates use a ``(n)`` suffix on the word).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO, Iterable, Iterator, List, Literal, Tuple

from arpabet_to_ipa import arpabet_words_to_ipa

# Strip CMU alternate index: hello(2) -> hello
_ALT_SUFFIX = re.compile(r"\(\d+\)$")


def _normalize_grapheme(word_token: str) -> str:
    return _ALT_SUFFIX.sub("", word_token).lower()


def split_text_to_words(text: str) -> List[str]:
    """
    Split *text* on arbitrary runs of whitespace (same as ``str.split()`` with no args).
    """
    return text.split()


def normalize_word_for_lookup(token: str) -> str:
    """
    Lowercase *token* and remove leading/trailing characters for which ``str.isalnum()``
    is false. Used to map surface tokens (e.g. ``Hello,``) to CMUdict keys. Returns
    an empty string if nothing alphanumeric remains.
    """
    s = token.lower().strip()
    if not s:
        return ""
    i, j = 0, len(s)
    while i < j and not s[i].isalnum():
        i += 1
    while i < j and not s[j - 1].isalnum():
        j -= 1
    return s[i:j]


class CmudictIpa:
    """
    ``word_lower`` -> sorted unique IPA strings for all listed pronunciations.
    """

    def __init__(
        self,
        source: str | Path | IO[str],
        *,
        format: Literal["cmudict", "tsv", "auto"] = "auto",
    ) -> None:
        self._ipa_by_word: dict[str, List[str]] = {}
        fmt: Literal["cmudict", "tsv"]
        if format == "auto":
            if isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".tsv":
                fmt = "tsv"
            else:
                fmt = "cmudict"
        else:
            fmt = format

        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8", errors="replace") as f:
                self._load_lines(f, fmt)
        else:
            self._load_lines(source, fmt)

    def _load_lines(self, f: IO[str], fmt: Literal["cmudict", "tsv"]) -> None:
        if fmt == "tsv":
            self._load_tsv_lines(f)
        else:
            self._load_cmudict_lines(f)

    def _load_cmudict_lines(self, f: IO[str]) -> None:
        raw: dict[str, set[str]] = {}
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word_token, *phones = parts
            key = _normalize_grapheme(word_token)
            ipa = arpabet_words_to_ipa(phones)
            raw.setdefault(key, set()).add(ipa)
        self._ipa_by_word = {k: sorted(v) for k, v in raw.items()}

    def _load_tsv_lines(self, f: IO[str]) -> None:
        """One pronunciation per line: ``word<TAB>ipa``. ``#`` starts a comment line."""
        raw: dict[str, set[str]] = {}
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
            key = _normalize_grapheme(word_token)
            raw.setdefault(key, set()).add(ipa)
        self._ipa_by_word = {k: sorted(v) for k, v in raw.items()}

    def iter_pronunciation_rows(self) -> Iterator[tuple[str, str]]:
        """Yield ``(word_key, ipa)`` for each stored pronunciation (sorted by word, then IPA)."""
        for w in sorted(self._ipa_by_word):
            for ipa in self._ipa_by_word[w]:
                yield w, ipa

    def translate_to_ipa(self, words: Iterable[str]) -> List[Tuple[str, List[str]]]:
        """
        For each input word (in order), return ``(original_grapheme, pronunciations)``.
        ``pronunciations`` is a sorted list of IPA strings, or empty if the word is unknown.
        Keys are :func:`normalize_word_for_lookup` of each input; the first tuple element is
        still the original string from *words*. Alternates ``word(2)`` in the file are merged
        under ``word``.
        """
        out: List[Tuple[str, List[str]]] = []
        for w in words:
            key = normalize_word_for_lookup(w)
            if not key:
                out.append((w, []))
                continue
            alts = self._ipa_by_word.get(key)
            if alts is None:
                out.append((w, []))
            else:
                out.append((w, list(alts)))
        return out
