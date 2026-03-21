"""Tests for heteronym disambiguation (no network; tiny synthetic JSON)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from heteronym.librig2p import (
    CharVocab,
    build_homograph_candidate_tables,
    load_homograph_json,
    save_training_artifacts,
    load_training_artifacts,
    iter_encoded_batches,
)
from heteronym.model import TinyHeteronymTransformer, masked_candidate_loss


def _tiny_json_path() -> Path:
    data = {
        "a": {
            "char": "HELLO READ WORLD",
            "homograph": "read",
            "homograph_wordid": "read_vrb",
            "homograph_char_start": 6,
            "homograph_char_end": 10,
        },
        "b": {
            "char": "SHE READ A BOOK",
            "homograph": "read",
            "homograph_wordid": "read_pst",
            "homograph_char_start": 4,
            "homograph_char_end": 8,
        },
    }
    d = tempfile.mkdtemp()
    p = Path(d) / "h.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_and_tables() -> None:
    recs = load_homograph_json(_tiny_json_path())
    assert len(recs) == 2
    ordered, maps = build_homograph_candidate_tables(recs, max_candidates=4, group_key="lower")
    assert ordered["read"] == ["read_pst", "read_vrb"]
    assert maps["read"]["read_vrb"] in (0, 1)


def test_model_and_loss() -> None:
    recs = load_homograph_json(_tiny_json_path())
    ordered, label_maps = build_homograph_candidate_tables(recs, max_candidates=4, group_key="lower")
    vocab = CharVocab.from_records(recs)
    model = TinyHeteronymTransformer(
        vocab_size=len(vocab),
        max_seq_len=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        max_candidates=4,
    )
    batches = list(
        iter_encoded_batches(
            recs,
            char_vocab=vocab,
            ordered_candidates=ordered,
            label_maps=label_maps,
            group_key="lower",
            max_seq_len=64,
            max_candidates=4,
            batch_size=8,
            shuffle=False,
            seed=0,
        )
    )
    assert batches
    b = batches[0]
    logits = model(b["input_ids"], b["attention_mask"], b["span_mask"])
    assert logits.shape == (b["labels"].shape[0], 4)
    loss = masked_candidate_loss(logits, b["labels"], b["candidate_mask"])
    assert loss.ndim == 0
    loss.backward()


def test_save_load_artifacts() -> None:
    recs = load_homograph_json(_tiny_json_path())
    ordered, label_maps = build_homograph_candidate_tables(recs, max_candidates=4, group_key="lower")
    vocab = CharVocab.from_records(recs)
    with tempfile.TemporaryDirectory() as d:
        save_training_artifacts(
            d,
            char_vocab=vocab,
            ordered_candidates=ordered,
            label_maps=label_maps,
            max_candidates=4,
            group_key="lower",
        )
        v2, o2, m2, k, gk = load_training_artifacts(d)
        assert len(v2) == len(vocab)
        assert o2 == ordered
        assert m2 == label_maps
        assert k == 4
        assert gk == "lower"
