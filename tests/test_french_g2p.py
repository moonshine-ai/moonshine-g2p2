"""Tests for :mod:`french_g2p` (lexicon + liaison)."""

from __future__ import annotations

import pytest

from french_g2p import (
    FrenchG2PConfig,
    _orthographic_liaison_consonant,
    ensure_french_nuclear_stress,
    load_french_lexicon,
    normalize_lookup_key,
    text_to_ipa,
    word_to_ipa,
)
from french_numbers import expand_cardinal_digits_to_french_words
from french_oov_rules import oov_word_to_ipa

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(__file__).resolve().parents[1].joinpath("data", "fr", "dict.tsv").is_file(),
    reason="data/fr/dict.tsv not present",
)


def _strip_stress(s: str) -> str:
    return s.replace("\u02c8", "").replace("\u02cc", "")


def test_normalize_key_apostrophe() -> None:
    assert normalize_lookup_key("l'homme") == "l'homme"


def test_lexicon_les_and_petit() -> None:
    lex = load_french_lexicon()
    assert lex[normalize_lookup_key("les")] == "le"
    assert lex[normalize_lookup_key("petit")] in {"pəti", "pti"}


def test_orthographic_liaison_consonant_des() -> None:
    assert _orthographic_liaison_consonant("des") == "z"


def test_liaison_les_amis() -> None:
    cfg = FrenchG2PConfig()
    out = text_to_ipa("les amis", config=cfg)
    parts = out.split()
    assert _strip_stress(parts[0]) == "lez"
    assert "ami" in _strip_stress(out)


def test_liaison_des_idees() -> None:
    cfg = FrenchG2PConfig()
    out = text_to_ipa("des idées", config=cfg)
    assert _strip_stress(out.split()[0]) == "dez"


def test_liaison_nous_avons() -> None:
    cfg = FrenchG2PConfig()
    out = text_to_ipa("nous avons", config=cfg)
    assert _strip_stress(out.split()[0]) == "nuz"


def test_liaison_mon_ami_nasal() -> None:
    cfg = FrenchG2PConfig()
    out = text_to_ipa("mon ami", config=cfg)
    assert _strip_stress(out.split()[0]) == "mɔn"


def test_no_liaison_noun_verb() -> None:
    cfg = FrenchG2PConfig()
    out = text_to_ipa("chat mange", config=cfg)
    assert _strip_stress(out.split()[0]) == "ʃa"
    assert "tʃ" not in _strip_stress(out.split()[0])


def test_no_liaison_without_flag() -> None:
    cfg = FrenchG2PConfig(liaison=False)
    out = text_to_ipa("les amis", config=cfg)
    assert _strip_stress(out.split()[0]) == "le"


def test_word_to_ipa_et() -> None:
    assert _strip_stress(word_to_ipa("et")) == "e"
    assert "\u02c8" in word_to_ipa("et")


def test_ensure_french_nuclear_stress_matches_lexicon_shape() -> None:
    # Lexicon strings often lack ˈ; align with eSpeak-style nuclear stress on the last syllable.
    assert ensure_french_nuclear_stress("bɔ̃ʒuʁ") == "bɔ̃ʒˈuʁ"
    assert ensure_french_nuclear_stress("kɔmɑ̃") == "kɔmˈɑ̃"


def test_oov_rules_produce_ipa() -> None:
    ipa = oov_word_to_ipa("foobaz", with_stress=True)
    assert ipa
    assert "f" in ipa and "u" in ipa


def test_word_to_ipa_oov_fallback() -> None:
    lex = load_french_lexicon()
    assert word_to_ipa("foobaz", lexicon=lex, use_oov_rules=True)
    assert word_to_ipa("foobaz", lexicon=lex, use_oov_rules=False) == ""


def test_text_to_ipa_oov_with_liaison() -> None:
    cfg = FrenchG2PConfig()
    # OOV vowel-initial head: liaison /z/ still applies (PRON + unknown POS).
    out = text_to_ipa("les ardvarks", config=cfg)
    assert _strip_stress(out.split()[0]) == "lez"
    assert "a" in _strip_stress(out)


def test_expand_cardinal_1891() -> None:
    assert expand_cardinal_digits_to_french_words("1891") == "mille huit cent quatre-vingt-onze"


def test_text_to_ipa_en_1891() -> None:
    cfg = FrenchG2PConfig()
    out = text_to_ipa("En 1891", config=cfg)
    assert "mˈil" in out or "mil" in _strip_stress(out)
    assert "katʁv" in _strip_stress(out) or "kˈatʁv" in out


def test_text_to_ipa_preserves_space_after_punctuation() -> None:
    cfg = FrenchG2PConfig()
    out = text_to_ipa("Bonjour! Salut", config=cfg)
    assert "! " in out
    out2 = text_to_ipa("Vu? Oui", config=cfg)
    assert "? " in out2
