"""Tests for :mod:`portuguese_numbers` and digit expansion in :mod:`portuguese_rule_g2p`."""

from __future__ import annotations

from pathlib import Path

import pytest

from portuguese_numbers import expand_cardinal_digits_to_portuguese_words, expand_digit_tokens_in_text
from portuguese_rule_g2p import text_to_ipa, word_to_ipa

ROOT = Path(__file__).resolve().parents[1]
_DICT_BR = ROOT / "data" / "pt_br" / "dict.tsv"

pytestmark = pytest.mark.skipif(
    not _DICT_BR.is_file(),
    reason="data/pt_br/dict.tsv not present",
)


def test_expand_1891_cardinal() -> None:
    w = expand_cardinal_digits_to_portuguese_words("1891")
    assert "mil" in w
    assert "oitocentos" in w


def test_expand_range_hyphen() -> None:
    s = expand_digit_tokens_in_text("1933-1945")
    assert " - " in s
    assert "trinta" in s and "quarenta" in s


def test_expand_leading_zeros() -> None:
    assert expand_cardinal_digits_to_portuguese_words("007") == "zero zero sete"


def test_br_vs_pt_teens() -> None:
    assert "dezesseis" in expand_cardinal_digits_to_portuguese_words("16", variant="pt_br")
    assert "dezasseis" in expand_cardinal_digits_to_portuguese_words("16", variant="pt_pt")


def test_word_to_ipa_digit_expansion() -> None:
    ipa = word_to_ipa("2000", variant="pt_br")
    assert ipa != "2000"
    assert "ˈ" in ipa or "i" in ipa


def test_word_to_ipa_no_expand() -> None:
    assert word_to_ipa("42", variant="pt_br", expand_cardinal_digits=False) == "42"


def test_text_expands_embedded_year() -> None:
    out = text_to_ipa("Em 1891.", variant="pt_br", expand_cardinal_digits=True)
    assert "1891" not in out
    assert "mil" in out.replace("ˈ", "")
