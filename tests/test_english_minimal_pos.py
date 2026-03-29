"""Tests for ``english_minimal_pos``."""

from __future__ import annotations

from english_minimal_pos import (
    coarse_pos_tag,
    immediate_left_is_det_or_poss,
    left_token_suggests_infinitive_or_finite_verb_after,
    right_token_starts_prep_phrase,
)


def test_coarse_tags():
    assert coarse_pos_tag("the") == "DET"
    assert coarse_pos_tag("their") == "POSS"
    assert coarse_pos_tag("they") == "PRON"
    assert coarse_pos_tag("will") == "MODAL"
    assert coarse_pos_tag("they'll") == "MODAL"
    assert coarse_pos_tag("don't") == "MODAL"
    assert coarse_pos_tag("was") == "AUX"
    assert coarse_pos_tag("to") == "TO"
    assert coarse_pos_tag("in") == "PREP"
    assert coarse_pos_tag("very") == "ADV"
    assert coarse_pos_tag("seems") == "COPULA"
    assert coarse_pos_tag("xyzabc") == "UNK"


def test_left_verb_context():
    assert left_token_suggests_infinitive_or_finite_verb_after(["they", "will"])
    assert left_token_suggests_infinitive_or_finite_verb_after(["to"])
    assert left_token_suggests_infinitive_or_finite_verb_after(["do", "n't"]) is False
    assert left_token_suggests_infinitive_or_finite_verb_after(["don't"])
    assert not left_token_suggests_infinitive_or_finite_verb_after(["was"])


def test_det_poss():
    assert immediate_left_is_det_or_poss(["a"])
    assert immediate_left_is_det_or_poss(["her"])
    assert not immediate_left_is_det_or_poss(["runs"])


def test_prep_right():
    assert right_token_starts_prep_phrase(["in", "London"])
    assert not right_token_starts_prep_phrase(["album"])
