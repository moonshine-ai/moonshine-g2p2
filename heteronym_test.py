"""Tests for heteronym disambiguation (no network; tiny synthetic JSON)."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import torch

import train_heteronym
from g2p_common import inference_context_window
from heteronym.librig2p import (
    HomographRecord,
    apply_train_augmentation,
    build_char_vocab_from_homograph_records,
    build_homograph_candidate_tables,
    iter_encoded_batches,
    load_homograph_json,
    load_training_artifacts,
    save_training_artifacts,
    _training_index_order,
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
    vocab = build_char_vocab_from_homograph_records(recs)
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
    vocab = build_char_vocab_from_homograph_records(recs)
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


def test_augment_preserves_homograph_slice() -> None:
    text = "AA BB READ CC DD EE FF"
    s, e = text.index("READ"), text.index("READ") + 4
    r = HomographRecord(text, "read", "read_x", s, e)
    vocab = build_char_vocab_from_homograph_records([r], extra_chars=".,;")
    ref = text[s:e]
    for seed in range(400):
        rng = random.Random(seed)
        out = apply_train_augmentation(
            r,
            char_vocab=vocab,
            max_seq_len=128,
            rng=rng,
            surface_noise_prob=0.85,
        )
        assert out is not None
        t2, s2, e2 = out
        assert t2[s2:e2] == ref


def test_inference_context_window_keeps_span() -> None:
    pad = "0123456789" * 30
    text = pad + "READ" + pad
    s = text.index("READ")
    e = s + 4
    out = inference_context_window(text, s, e, max_seq_len=64)
    assert out is not None
    t2, s2, e2 = out
    assert len(t2) <= 64
    assert t2[s2:e2] == "READ"


def test_random_crop_keeps_span_in_window() -> None:
    from heteronym.librig2p import _random_context_window

    pad = "0123456789" * 30
    text = pad + "READ" + pad
    s = text.index("READ")
    e = s + 4
    for seed in range(200):
        rng = random.Random(seed)
        w = _random_context_window(text, s, e, max_seq_len=64, rng=rng)
        assert w is not None
        t2, s2, e2 = w
        assert len(t2) <= 64
        assert t2[s2:e2] == "READ"


def test_balance_training_oversamples_rare_wordid() -> None:
    recs: list[HomographRecord] = []
    for _ in range(9):
        recs.append(HomographRecord("X READ Y", "read", "read_common", 2, 6))
    recs.append(HomographRecord("X READ Y", "read", "read_rare", 2, 6))
    ordered, label_maps = build_homograph_candidate_tables(recs, max_candidates=4, group_key="lower")
    vocab = build_char_vocab_from_homograph_records(recs)
    rare_label = label_maps["read"]["read_rare"]
    ys: list[int] = []
    for b in iter_encoded_batches(
        recs,
        char_vocab=vocab,
        ordered_candidates=ordered,
        label_maps=label_maps,
        group_key="lower",
        max_seq_len=64,
        max_candidates=4,
        batch_size=1,
        shuffle=True,
        seed=123,
        train_augment=False,
        balance_training=True,
    ):
        ys.extend(int(x) for x in b["labels"].tolist())
    assert len(ys) == len(recs)
    rare_count = sum(1 for y in ys if y == rare_label)
    # Without balancing E[rare]≈1; inverse-frequency sampling targets E[rare]≈5 (high variance per epoch).
    assert rare_count >= 3
    assert rare_count > sum(1 for y in ys if y != rare_label) / 9


def test_training_index_order_shuffle_uniform_without_balance() -> None:
    recs = [
        HomographRecord("A READ B", "read", "w0", 2, 6),
        HomographRecord("C READ D", "read", "w1", 2, 6),
    ]
    lm = {"read": {"w0": 0, "w1": 1}}
    rng = random.Random(0)
    order = _training_index_order(
        recs, label_maps=lm, group_key="lower", shuffle=True, balance_training=False, rng=rng
    )
    assert sorted(order) == [0, 1]


def test_train_resume_checkpoint() -> None:
    tiny = _tiny_json_path()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"
        common = [
            "--out",
            str(out),
            "--train-json",
            str(tiny),
            "--valid-json",
            str(tiny),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--max-seq-len",
            "64",
            "--d-model",
            "64",
            "--n-layers",
            "2",
            "--ffn-dim",
            "128",
        ]
        train_heteronym.main(common)
        ck = out / train_heteronym.CHECKPOINT_NAME
        assert ck.is_file()
        train_heteronym.main(common + ["--resume", "--epochs", "2"])
        assert ck.is_file()
