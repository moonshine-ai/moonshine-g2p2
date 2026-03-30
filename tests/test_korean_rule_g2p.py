"""Tests for :mod:`korean_rule_g2p` (pure Python rule pipeline)."""

from __future__ import annotations

from korean_rule_g2p import (
    apply_linking,
    apply_lateralization,
    compose_syllable,
    decompose_syllable,
    korean_g2p,
    text_to_syllables,
    text_to_syllables_scan,
)


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


def test_g2p_examples() -> None:
    assert korean_g2p("닭이") == "dal.ki"
    assert korean_g2p("국물") == "kuŋ.mul"
    assert "ɕil.la" in korean_g2p("신라")
    assert korean_g2p("좋다") == "tɕo.tʰa"
    assert "k͈jo" in korean_g2p("학교")
    assert korean_g2p("닫는") == "dan.nɯn"


def test_text_to_syllables_alias() -> None:
    assert text_to_syllables_scan is text_to_syllables


def test_hangul_order_three_syllables() -> None:
    t = "한국어"
    got = [compose_syllable(x.cho, x.jung, x.jong) for x in text_to_syllables(t)]
    assert got == ["한", "국", "어"]


def test_korean_g2p_ignores_legacy_use_mecab_kwarg() -> None:
    assert korean_g2p("학교", use_mecab=False) == korean_g2p("학교")
    assert korean_g2p("학교", use_mecab=True) == korean_g2p("학교")
