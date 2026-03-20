"""
Load CMU Pronouncing Dictionary entries and expose IPA pronunciation alternatives.

Dictionary source format matches
https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict
(word token, then ARPAbet phones; alternates use a ``(n)`` suffix on the word).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO, Iterable, List, Tuple

from arpabet_to_ipa import arpabet_words_to_ipa

# Strip CMU alternate index: hello(2) -> hello
_ALT_SUFFIX = re.compile(r"\(\d+\)$")


def _normalize_grapheme(word_token: str) -> str:
    return _ALT_SUFFIX.sub("", word_token).lower()


class CmudictIpa:
    """
    ``word_lower`` -> sorted unique IPA strings for all listed pronunciations.
    """

    def __init__(self, source: str | Path | IO[str]) -> None:
        self._ipa_by_word: dict[str, List[str]] = {}
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8", errors="replace") as f:
                self._load_lines(f)
        else:
            self._load_lines(source)

    def _load_lines(self, f: IO[str]) -> None:
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

    def translate_to_ipa(self, words: Iterable[str]) -> List[Tuple[str, List[str]]]:
        """
        For each input word (in order), return ``(original_grapheme, pronunciations)``.
        ``pronunciations`` is a sorted list of IPA strings, or empty if the word is unknown.
        Matching is case-insensitive; alternates ``word(2)`` in the file are merged under ``word``.
        """
        out: List[Tuple[str, List[str]]] = []
        for w in words:
            key = w.lower()
            alts = self._ipa_by_word.get(key)
            if alts is None:
                out.append((w, []))
            else:
                out.append((w, list(alts)))
        return out
