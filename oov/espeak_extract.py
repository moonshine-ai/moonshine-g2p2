"""Recover eSpeak IPA for a character span via full vs removed text (word-level chunks)."""

from __future__ import annotations

import re

from heteronym.espeak_heteronyms import (
    EspeakPhonemizer,
    espeak_ipa_tokens,
    longest_insert_block,
)

_WS_COLLAPSE = re.compile(r"  +")


def extract_span_espeak_phonemes(
    phonemizer: EspeakPhonemizer,
    text: str,
    voice: str,
    char_start: int,
    char_end: int,
) -> list[str] | None:
    """
    Tokenize *text* with eSpeak (word-level IPA, no spaces within words); remove
    ``text[char_start:char_end]`` and diff token lists to find the inserted chunk.
    Returns ``None`` on failure.
    """
    if char_start < 0 or char_end <= char_start or char_end > len(text):
        return None
    tokens_full = espeak_ipa_tokens(phonemizer, text, voice=voice)
    if not tokens_full:
        return None
    removed = text[:char_start] + text[char_end:]
    removed = _WS_COLLAPSE.sub(" ", removed).strip()
    if not removed:
        tokens_without: list[str] = []
    else:
        tokens_without = espeak_ipa_tokens(phonemizer, removed, voice=voice)
        if not tokens_without:
            return None
    block = longest_insert_block(tokens_without, tokens_full)
    return block if block else None
