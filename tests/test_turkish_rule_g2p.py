"""Tests for :mod:`turkish_rule_g2p` and :mod:`turkish_numbers`."""

from __future__ import annotations

from turkish_numbers import expand_cardinal_digits_to_turkish_words, expand_digit_tokens_in_text
from turkish_rule_g2p import text_to_ipa, word_to_ipa


def test_turkish_lower_and_dag() -> None:
    assert word_to_ipa("Dağ") == "dˈaː"


def test_deger_intervocalic_gh() -> None:
    assert word_to_ipa("değer") == "dejˈeɾ"


def test_apostrophe_splits_suffix() -> None:
    ipa = text_to_ipa("İstanbul'da")
    assert "istanbˈul" in ipa
    assert "dˈa" in ipa


def test_digit_range_expansion() -> None:
    t = expand_digit_tokens_in_text("1206-1227 yılı")
    assert "1206" not in t
    assert "1227" not in t
    assert "bin" in t


def test_cardinal_13() -> None:
    assert expand_cardinal_digits_to_turkish_words("13") == "on üç"
