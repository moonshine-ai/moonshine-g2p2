"""Tests for :mod:`russian_rule_g2p` (lexicon + OOV rules)."""

from __future__ import annotations

from pathlib import Path

import pytest

from russian_rule_g2p import (
    default_dict_path,
    load_russian_lexicon,
    normalize_lookup_key,
    russian_orthographic_syllables,
    word_to_ipa,
)

ROOT = Path(__file__).resolve().parents[1]
_DICT = ROOT / "data" / "ru" / "dict.tsv"

pytestmark = pytest.mark.skipif(not _DICT.is_file(), reason="data/ru/dict.tsv not present")


def test_normalize_lookup_key_strips_stress() -> None:
    assert normalize_lookup_key("литва́") == "литва"
    assert normalize_lookup_key("Литва") == "литва"


def test_lexicon_litva() -> None:
    lex = load_russian_lexicon(_DICT)
    assert "литва" in lex
    w = word_to_ipa("Литва", lexicon=lex)
    assert w == lex["литва"]


def test_word_to_ipa_oov_emits_ipa() -> None:
    ipa = word_to_ipa("квазислово", lexicon={})
    assert ipa
    assert "ˈ" in ipa or any(x in ipa for x in "kvz")


def test_orthographic_syllables_litva() -> None:
    assert russian_orthographic_syllables("литва") == ["ли", "тва"]


def test_oov_respects_acute_stress() -> None:
    ipa = word_to_ipa("литва́", lexicon={})
    assert "ˈ" in ipa


def test_default_dict_path() -> None:
    assert default_dict_path().name == "dict.tsv"
    assert "ru" in str(default_dict_path())


def test_hyphen_compound_oov_rules() -> None:
    ipa = word_to_ipa("юго-восток", lexicon={})
    assert "-" in ipa
    assert "ˈ" in ipa or "j" in ipa
