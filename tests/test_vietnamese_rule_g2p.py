"""Tests for :mod:`vietnamese_rule_g2p` (lexicon + OOV rules)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vietnamese_rule_g2p import vietnamese_g2p_line, vietnamese_syllable_to_ipa

_REPO = Path(__file__).resolve().parent.parent
_VI_DICT = _REPO / "data" / "vi" / "dict.tsv"


@pytest.mark.skipif(not _VI_DICT.is_file(), reason="missing data/vi/dict.tsv")
def test_syllable_oov_matches_ipa_dict_samples() -> None:
    pairs = [
        ("tra", "ca˧˧"),
        ("chức", "cɯk˦˥"),
        ("không", "xoŋ͡m˧˧"),
        ("học", "hɔk͡p˨ˀ˩"),
        ("giáo", "zaw˨˦"),
        ("thành", "tʰɛŋ˧˨"),
        ("Việt", "viət˨ˀ˩"),
        ("gì", "ɣi˧˨"),
        ("gia", "za˧˧"),
        ("quốc", "kwok͡p˦˥"),
        ("chị", "ci˨ˀ˩ʔ"),
        ("trọng", "cɔŋ͡m˨ˀ˩ʔ"),
        ("những", "ɲɯŋ˧ˀ˥"),
        ("đến", "den˨˦"),
        ("một", "mot˨ˀ˩"),
    ]
    for w, exp in pairs:
        assert vietnamese_syllable_to_ipa(w) == exp, w


@pytest.mark.skipif(not _VI_DICT.is_file(), reason="missing data/vi/dict.tsv")
def test_longest_match_phrase() -> None:
    assert vietnamese_g2p_line("tổ chức", dict_path=_VI_DICT) == "to˧˩˨ cɯk˦˥"


@pytest.mark.skipif(not _VI_DICT.is_file(), reason="missing data/vi/dict.tsv")
def test_ascii_token_lowercase_passthrough() -> None:
    assert "internet" in vietnamese_g2p_line("Internet", dict_path=_VI_DICT)
