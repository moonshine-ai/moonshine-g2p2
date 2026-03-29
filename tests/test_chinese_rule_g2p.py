"""Tests for Mandarin numerals + lexicon IPA (no ONNX required)."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def zh_lex():
    from chinese_rule_g2p import load_zh_hans_lexicon

    return load_zh_hans_lexicon(_REPO / "data" / "zh_hans" / "dict.tsv")


def test_int_to_mandarin_cardinal_han():
    from chinese_rule_g2p import int_to_mandarin_cardinal_han

    assert int_to_mandarin_cardinal_han(0) == "零"
    assert int_to_mandarin_cardinal_han(10) == "十"
    assert int_to_mandarin_cardinal_han(11) == "十一"
    assert int_to_mandarin_cardinal_han(20) == "二十"
    assert int_to_mandarin_cardinal_han(42) == "四十二"
    assert int_to_mandarin_cardinal_han(100) == "一百"
    assert int_to_mandarin_cardinal_han(101) == "一百零一"
    assert int_to_mandarin_cardinal_han(110) == "一百一十"
    assert int_to_mandarin_cardinal_han(1010) == "一千零一十"
    assert int_to_mandarin_cardinal_han(10000) == "一万"
    assert int_to_mandarin_cardinal_han(10001) == "一万零一"
    assert int_to_mandarin_cardinal_han(100200003) == "一亿零二十万零三"
    assert int_to_mandarin_cardinal_han(10**9) == "十亿"


def test_arabic_numeral_token_to_han():
    from chinese_rule_g2p import arabic_numeral_token_to_han

    assert arabic_numeral_token_to_han("42") == "四十二"
    assert arabic_numeral_token_to_han("４２") == "四十二"
    assert arabic_numeral_token_to_han("007") == "零零七"
    assert arabic_numeral_token_to_han("-42") == "负四十二"
    assert arabic_numeral_token_to_han("3.14") == "三点一四"
    assert arabic_numeral_token_to_han("1,234") == "一千二百三十四"
    assert arabic_numeral_token_to_han("hello") is None


def test_arabic_numeral_token_to_ipa(zh_lex):
    from chinese_rule_g2p import arabic_numeral_token_to_ipa

    ipa = arabic_numeral_token_to_ipa("42", zh_lex)
    assert ipa is not None
    assert len(ipa.split()) == 3  # 四 十 二

    ipa2 = arabic_numeral_token_to_ipa("3.14", zh_lex)
    assert ipa2 is not None
    assert "tjɛn" in ipa2  # 点


def test_han_reading_to_ipa_rejects_unknown(zh_lex):
    from chinese_rule_g2p import han_reading_to_ipa

    assert han_reading_to_ipa("四十二", zh_lex) is not None
    assert han_reading_to_ipa("", zh_lex) is None
    assert han_reading_to_ipa("𠀋", zh_lex) is None  # rare char unlikely in dict
