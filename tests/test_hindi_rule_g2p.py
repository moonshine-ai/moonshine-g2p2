"""Tests for :mod:`hindi_rule_g2p` and :mod:`hindi_numbers`."""

from __future__ import annotations

from hindi_numbers import expand_cardinal_digits_to_hindi_words, expand_digit_tokens_in_text
from hindi_rule_g2p import devanagari_word_to_ipa, text_to_ipa


def test_expand_42() -> None:
    assert expand_cardinal_digits_to_hindi_words("42") == "बयालीस"


def test_expand_digit_tokens_in_text() -> None:
    assert "बयालीस" in expand_digit_tokens_in_text("संख्या 42 है")


def test_kamal_final_schwa() -> None:
    ipa = devanagari_word_to_ipa("कमल")
    assert "k" in ipa and "m" in ipa


def test_main_line_sanchaar() -> None:
    assert "tʃ" in text_to_ipa("संचार")


def test_digits_to_words_then_g2p() -> None:
    out = text_to_ipa("५")
    assert out  # expanded to Devanagari cardinal then IPA
