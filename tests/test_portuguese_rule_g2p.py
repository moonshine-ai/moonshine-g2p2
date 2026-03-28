"""Tests for :mod:`portuguese_rule_g2p` (lexicon + OOV rules)."""

from __future__ import annotations

from pathlib import Path

import pytest

from portuguese_rule_g2p import (
    default_dict_path,
    load_portuguese_lexicon,
    normalize_lookup_key,
    portuguese_orthographic_syllables,
    word_to_ipa,
)

ROOT = Path(__file__).resolve().parents[1]
_DICT_BR = ROOT / "data" / "pt_br" / "dict.tsv"
_DICT_PT = ROOT / "data" / "pt_pt" / "dict.tsv"

pytestmark = pytest.mark.skipif(
    not _DICT_BR.is_file() or not _DICT_PT.is_file(),
    reason="data/pt_br/dict.tsv and/or data/pt_pt/dict.tsv not present",
)


def test_normalize_lookup_key_apostrophe() -> None:
    assert normalize_lookup_key("d'água") == "d'água"


def test_lexicon_casa_br_and_pt() -> None:
    lex_br = load_portuguese_lexicon(_DICT_BR, variant="pt_br")
    lex_pt = load_portuguese_lexicon(_DICT_PT, variant="pt_pt")
    assert "casa" in lex_br and "casa" in lex_pt
    assert lex_br["casa"].startswith("ˈ") or "a" in lex_br["casa"]
    assert "z" in lex_pt["casa"] or "s" in lex_pt["casa"]


def test_word_to_ipa_uses_lexicon_when_present() -> None:
    lex = load_portuguese_lexicon(_DICT_BR, variant="pt_br")
    w = word_to_ipa("casa", variant="pt_br", lexicon=lex)
    assert w == lex["casa"].replace(".", "")
    w_dots = word_to_ipa("casa", variant="pt_br", lexicon=lex, keep_syllable_dots=True)
    assert w_dots == lex["casa"]


def test_default_dict_path() -> None:
    assert default_dict_path("pt_br").name == "dict.tsv"
    assert "pt_br" in str(default_dict_path("pt_br"))


def test_oov_rules_emit_ipa() -> None:
    ipa = word_to_ipa("xyzabc", variant="pt_br", lexicon={})
    assert ipa
    assert any(c in ipa for c in "kszbdɡ")


def test_oov_ce_retains_vowel() -> None:
    """⟨ce⟩ was wrongly mapped to /s/ without the following vowel."""
    ipa = word_to_ipa("celestes", variant="pt_br", lexicon={})
    assert ipa.startswith("sɪ") or ipa.startswith("se")


def test_roman_xx_to_ipa() -> None:
    assert "vĩ" in word_to_ipa("XX", variant="pt_br", lexicon={})
    assert "vĩ" in word_to_ipa("xx", variant="pt_br", lexicon={})


def test_hyphen_dividiu_se() -> None:
    ipa = word_to_ipa("dividiu-se", variant="pt_br", lexicon={})
    assert "-" in ipa
    assert ipa.endswith("sˈe") or "se" in ipa.replace("-", "")


def test_pt_pt_oov_plural_final_s_to_esh() -> None:
    ipa = word_to_ipa("planetas", variant="pt_pt", lexicon={})
    assert ipa.endswith("ʃ")


def test_pt_pt_respects_final_s_exclude() -> None:
    ipa = word_to_ipa("lápis", variant="pt_pt", lexicon={})
    assert ipa.endswith("s") or "ʃ" not in ipa[-2:]


def test_orthographic_syllables_ão() -> None:
    assert portuguese_orthographic_syllables("coração") == ["co", "ra", "ção"]


def test_variant_function_word_de() -> None:
    br = word_to_ipa("de", variant="pt_br", lexicon={})
    pt = word_to_ipa("de", variant="pt_pt", lexicon={})
    assert "ʒ" in br or "d" in br
    assert pt != br or "ɨ" in pt
