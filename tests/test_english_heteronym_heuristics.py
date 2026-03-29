"""Tests for heteronym context heuristics."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def g2p():
    from english_rule_g2p import EnglishLexiconRuleG2p

    tsv = _REPO / "models" / "en_us" / "dict_filtered_heteronyms.tsv"
    if not tsv.is_file():
        pytest.skip("dict TSV missing")
    return EnglishLexiconRuleG2p(dict_path=tsv)


def test_g2p_span_live_verb(g2p):
    t = "Everyone should live in an orderly way ."
    s = t.index("live")
    assert g2p.g2p_span(t, s, s + 4) == "lˈɪv"


def test_g2p_span_live_modal_contraction(g2p):
    t = "They'll live nearby ."
    s = t.index("live")
    assert g2p.g2p_span(t, s, s + 4) == "lˈɪv"


def test_g2p_span_live_adj(g2p):
    t = "They played a live album on tour ."
    s = t.index("live")
    assert g2p.g2p_span(t, s, s + 4) == "lˈaɪv"


def test_g2p_span_use_verb(g2p):
    t = "We can use it tomorrow ."
    s = t.index("use")
    assert g2p.g2p_span(t, s, s + 3) == "jˈuz"


def test_g2p_span_use_noun(g2p):
    t = "The use of force was debated ."
    s = t.index("use")
    assert g2p.g2p_span(t, s, s + 3) == "jˈus"


def test_baseline_first_candidate_is_merge_order(g2p):
    c = g2p.pronunciation_candidates("live")
    assert len(c) == 2
    assert c[0] == "lˈaɪv"


def test_g2p_span_read_infinitive_multi_token_left(g2p):
    t = "She is going to read the memo ."
    s = t.index("read")
    assert g2p.g2p_span(t, s, s + 4) == "ɹˈid"


def test_g2p_span_live_on_stage_adj(g2p):
    t = "The band played live on stage ."
    s = t.index("live")
    assert g2p.g2p_span(t, s, s + 4) == "lˈaɪv"


@pytest.mark.parametrize(
    "sentence",
    (
        "They live on the edge of town .",
        # Exception ``on the edge`` must beat broad ``wr1 == on`` → adj.
        "People who live on the edge take risks .",
    ),
)
def test_g2p_span_live_on_edge_exception_verb(g2p, sentence):
    s = sentence.index("live")
    assert g2p.g2p_span(sentence, s, s + 4) == "lˈɪv"


def test_g2p_span_close_down_phrasal(g2p):
    t = "We need to close down the shop ."
    s = t.index("close")
    assert g2p.g2p_span(t, s, s + 5) == "klˈoʊz"
