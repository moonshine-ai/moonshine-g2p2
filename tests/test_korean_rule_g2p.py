"""Tests for :mod:`korean_rule_g2p` (required ``data/ko/dict.tsv`` + normalization)."""

from __future__ import annotations

from pathlib import Path

import pytest

from korean_rule_g2p import (
    apply_linking,
    apply_lateralization,
    compose_syllable,
    decompose_syllable,
    korean_g2p,
    load_korean_lexicon,
    normalize_korean_ipa,
    text_to_syllables,
    text_to_syllables_scan,
)

_REPO = Path(__file__).resolve().parents[1]
_KO_DICT = _REPO / "data" / "ko" / "dict.tsv"

pytestmark = pytest.mark.skipif(not _KO_DICT.is_file(), reason="data/ko/dict.tsv required")


def test_decompose_compose_roundtrip() -> None:
    assert decompose_syllable("각") == (0, 0, 1)
    assert compose_syllable(0, 0, 1) == "각"


def test_linking_dalgi() -> None:
    syls = text_to_syllables("닭이")
    apply_linking(syls)
    assert compose_syllable(syls[0].cho, syls[0].jung, syls[0].jong) == "달"
    assert compose_syllable(syls[1].cho, syls[1].jung, syls[1].jong) == "기"


def test_lateralization_silla() -> None:
    syls = text_to_syllables("신라")
    apply_linking(syls)
    apply_lateralization(syls)
    assert syls[0].jong == 8  # coda ㄹ


def test_normalize_strips_vowel_diacritics_keeps_tense_unreleased() -> None:
    assert normalize_korean_ipa("ha̠k̚k͈jo") == "hak̚k͈jo"
    assert normalize_korean_ipa("kuŋmuɭ") == "kuŋmul"


def test_g2p_examples_with_required_lexicon() -> None:
    lex = load_korean_lexicon(_KO_DICT)
    assert korean_g2p("닭이", dict_path=_KO_DICT) == "dal.ki"  # OOV → rules + normalize
    assert korean_g2p("국물", dict_path=_KO_DICT) == lex["국물"]
    assert korean_g2p("신라", dict_path=_KO_DICT) == lex["신라"]
    assert korean_g2p("좋다", dict_path=_KO_DICT) == lex["좋다"]
    assert korean_g2p("학교", dict_path=_KO_DICT) == lex["학교"]
    assert korean_g2p("닫는", dict_path=_KO_DICT) == "dan.nɯn"


def test_text_to_syllables_alias() -> None:
    assert text_to_syllables_scan is text_to_syllables


def test_hangul_order_three_syllables() -> None:
    t = "한국어"
    got = [compose_syllable(x.cho, x.jung, x.jong) for x in text_to_syllables(t)]
    assert got == ["한", "국", "어"]


def test_korean_g2p_ignores_legacy_kwargs() -> None:
    assert korean_g2p("학교", use_mecab=False, dict_path=_KO_DICT) == korean_g2p("학교", dict_path=_KO_DICT)
    assert korean_g2p("학교", use_lexicon=True, dict_path=_KO_DICT) == korean_g2p("학교", dict_path=_KO_DICT)


def test_lexicon_full_token_hakgyo() -> None:
    lex = load_korean_lexicon(_KO_DICT)
    assert "학교" in lex
    assert korean_g2p("학교", dict_path=_KO_DICT) == lex["학교"]


def test_lexicon_oov_dalgi_uses_rules_for_sandhi() -> None:
    assert korean_g2p("닭이", dict_path=_KO_DICT) == "dal.ki"


def test_lexicon_two_words_joined_with_space() -> None:
    lex = load_korean_lexicon(_KO_DICT)
    assert korean_g2p("학교 에", dict_path=_KO_DICT) == f'{lex["학교"]} {lex["에"]}'


def test_load_korean_lexicon_raises_without_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.tsv"
    with pytest.raises(FileNotFoundError):
        load_korean_lexicon(missing)
