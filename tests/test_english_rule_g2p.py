"""Tests for ``english_rule_g2p``."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def test_lexicon_hello():
    from english_rule_g2p import EnglishLexiconRuleG2p, load_english_lexicon

    tsv = _REPO / "models" / "en_us" / "dict_filtered_heteronyms.tsv"
    if not tsv.is_file():
        pytest.skip("dict TSV missing")
    lex = load_english_lexicon(tsv)
    g = EnglishLexiconRuleG2p(lexicon=lex)
    ipa = g.g2p("hello")
    assert "h" in ipa and "l" in ipa
    assert ipa == lex["hello"]


def test_oov_returns_non_empty():
    from english_rule_g2p import english_oov_rules_ipa

    s = english_oov_rules_ipa("xyzqx")
    assert s
    assert s[0] in "ˈˌ" or any(c in s for c in "əæɛɪɔʊ")


def test_function_word_the():
    from english_rule_g2p import english_oov_rules_ipa

    assert english_oov_rules_ipa("the").startswith("ð")


def test_g2p_oov_matches_hand_rules_when_onnx_disabled():
    from english_rule_g2p import EnglishLexiconRuleG2p, english_oov_rules_ipa, load_english_lexicon

    tsv = _REPO / "models" / "en_us" / "dict_filtered_heteronyms.tsv"
    if not tsv.is_file():
        pytest.skip("dict TSV missing")
    lex = load_english_lexicon(tsv)
    g = EnglishLexiconRuleG2p(lexicon=lex, use_onnx_oov=False)
    w = "xyzqx"
    assert g.g2p(w) == english_oov_rules_ipa(w)


def test_english_number_token_ipa():
    from english_rule_g2p import english_number_token_ipa

    assert english_number_token_ipa("42") == "fˈɔɹtiˌtˈu"
    assert english_number_token_ipa("007") == "ˈzɪroʊˌˈzɪroʊˌˈsɛvən"
    assert english_number_token_ipa("-1") == "nˈɛɡətɪvˌwˈʌn"
    assert english_number_token_ipa("hello") is None
    ipa = english_number_token_ipa("3.14")
    assert ipa is not None and "ˈpɔɪnt" in ipa


def test_g2p_numeric_uses_number_path():
    from english_rule_g2p import EnglishLexiconRuleG2p, english_number_token_ipa, load_english_lexicon

    tsv = _REPO / "models" / "en_us" / "dict_filtered_heteronyms.tsv"
    if not tsv.is_file():
        pytest.skip("dict TSV missing")
    lex = load_english_lexicon(tsv)
    g = EnglishLexiconRuleG2p(lexicon=lex, use_onnx_oov=False)
    assert g.g2p("1234") == english_number_token_ipa("1234")


def test_g2p_oov_onnx_when_model_present():
    from english_rule_g2p import EnglishLexiconRuleG2p, load_english_lexicon
    from moonshine_onnx_g2p import OnnxOovG2p

    tsv = _REPO / "models" / "en_us" / "dict_filtered_heteronyms.tsv"
    onnx_p = _REPO / "models" / "en_us" / "oov" / "model.onnx"
    if not tsv.is_file() or not onnx_p.is_file():
        pytest.skip("dict TSV or OOV ONNX missing")
    try:
        oov = OnnxOovG2p(onnx_p)
    except Exception:
        pytest.skip("OnnxOovG2p load failed")
    lex = load_english_lexicon(tsv)
    g = EnglishLexiconRuleG2p(lexicon=lex, use_onnx_oov=True, oov_onnx_path=onnx_p)
    w = "xyzqx"
    assert g.g2p(w) == "".join(oov.predict_phonemes(w))
