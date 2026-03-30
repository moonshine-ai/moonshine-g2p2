"""Smoke tests for :mod:`arabic_rule_g2p` (skip if ONNX bundle missing)."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_ONNX = _REPO / "data" / "ar_msa" / "arabertv02_tashkeel_fadel_onnx" / "model.onnx"
_DICT = _REPO / "data" / "ar_msa" / "dict.tsv"


@pytest.mark.skipif(not _ONNX.is_file(), reason="Arabic ONNX bundle not present")
def test_arabic_g2p_smoke():
    from arabic_rule_g2p import ArabicRuleG2p

    g = ArabicRuleG2p(model_dir=_ONNX.parent, dict_path=_DICT)
    out = g.text_to_ipa("الشمس")
    assert out
    assert "ʃ" in out or "s" in out


def test_arabic_ipa_assimilation_no_onnx():
    from arabic_ipa import apply_onnx_partial_postprocess, word_to_ipa_with_assimilation

    w = apply_onnx_partial_postprocess("الشمس")
    ipa = word_to_ipa_with_assimilation(w)
    assert ipa
