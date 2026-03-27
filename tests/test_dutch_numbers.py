"""Tests for :mod:`dutch_numbers` and digit expansion in :mod:`dutch_rule_g2p`."""

from __future__ import annotations

from dutch_numbers import expand_cardinal_digits_to_dutch_words, expand_digit_tokens_in_text
from dutch_rule_g2p import text_to_ipa, word_to_ipa


def test_expand_1905_year_style() -> None:
    assert expand_cardinal_digits_to_dutch_words("1905") == "negentienhonderd vijf"


def test_expand_range_tot() -> None:
    s = expand_digit_tokens_in_text("1933-1945")
    assert "tot" in s
    assert "negentienhonderd" in s


def test_expand_leading_zeros() -> None:
    assert expand_cardinal_digits_to_dutch_words("007") == "nul nul zeven"


def test_word_to_ipa_digit_expansion() -> None:
    ipa = word_to_ipa("2000")
    assert ipa != "2000"
    assert "tʋeː" in ipa or "dœy" in ipa  # ipa-dict *tweeduizend*


def test_word_to_ipa_no_expand() -> None:
    assert word_to_ipa("42", expand_cardinal_digits=False) == "42"


def test_text_preserves_non_digits() -> None:
    out = text_to_ipa("Het jaar 2000.", expand_cardinal_digits=True)
    assert "2000" not in out.split() or "twee" in out  # expanded
