"""Tests for :mod:`ukrainian_rule_g2p` and :mod:`ukrainian_numbers`."""

from __future__ import annotations

from ukrainian_numbers import expand_cardinal_digits_to_ukrainian_words, expand_digit_tokens_in_text
from ukrainian_rule_g2p import text_to_ipa, word_to_ipa


def test_strip_stress_keeps_yi() -> None:
    assert word_to_ipa("їжак") == word_to_ipa("\u0457\u0436\u0430\u043a")


def test_apostrophe_mjaso() -> None:
    ipa = word_to_ipa("м'ясо")
    assert ipa.startswith("m")
    assert "ja" in ipa


def test_numbers_and_digits_in_text() -> None:
    assert expand_cardinal_digits_to_ukrainian_words("5") == "п'ять"
    assert "п'ять" in expand_digit_tokens_in_text("Було 5.")
    t = text_to_ipa("Було 5 котів.")
    assert "p" in t and "jat" in t.replace("ˈ", "")


def test_text_scan_preserves_punctuation() -> None:
    a = text_to_ipa("Привіт, світ.")
    assert "," in a
    assert a.endswith(".")


def test_hyphen_compound() -> None:
    ipa = text_to_ipa("рослинно-м'ясний")
    assert "-" in ipa


def test_thousand_agreement() -> None:
    w = expand_cardinal_digits_to_ukrainian_words("21000")
    assert "двадцять" in w and "одна" in w and "тисяча" in w
