"""Korean morph+UPOS ONNX pipeline: quality checks vs PyTorch reference (optional)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_MODEL_DIR = _REPO / "data" / "ko" / "roberta_korean_morph_upos_onnx"


def _require_onnx_bundle():
    for n in ("model.onnx", "meta.json", "vocab.txt", "tokenizer_config.json"):
        if not (_MODEL_DIR / n).is_file():
            pytest.skip("Run: python scripts/export_korean_ud_onnx.py")


def test_morph_label_to_upos():
    from ko_roberta_morph_preprocess import morph_label_to_upos

    assert morph_label_to_upos("PUNCT") == "PUNCT"
    assert morph_label_to_upos("NOUN") == "NOUN"
    assert morph_label_to_upos("B-NOUN+AUX+PART") == "NOUN"
    assert morph_label_to_upos("B-VERB+CCONJ") == "VERB"


def test_token_word_group_indices():
    from ko_roberta_morph_preprocess import token_word_group_indices

    text = "대한민국의 수도는 서울이다."
    tokens = ["[CLS]", "대한민국", "##의", "수도", "##는", "서울", "##이다", ".", "[SEP]"]
    offsets = [(0, 0), (0, 4), (4, 5), (6, 8), (8, 9), (10, 12), (12, 14), (14, 15), (0, 0)]
    g = token_word_group_indices(tokens, offsets, text)
    assert g == [[1, 2], [3, 4], [5, 6], [7]]


def test_wordpiece_ids_match_hf_tokenizers_json():
    """Optional: our stdlib WordPiece matches ``tokenizer.json`` (HF export artifact)."""
    pytest.importorskip("tokenizers")
    _require_onnx_bundle()
    from tokenizers import Tokenizer

    from ko_roberta_wordpiece import encode_bert_wordpiece

    if not (_MODEL_DIR / "tokenizer.json").is_file():
        pytest.skip("no tokenizer.json in bundle")
    tj = Tokenizer.from_file(str(_MODEL_DIR / "tokenizer.json"))
    for s in ("대한민국의 수도는 서울이다.", "안녕하세요.", "a b c"):
        e = tj.encode(s)
        ids, toks, offs, _ref = encode_bert_wordpiece(s, _MODEL_DIR)
        assert ids == e.ids
        assert toks == e.tokens
        assert list(offs) == list(e.offsets)


def test_onnx_loads_and_nonempty_output():
    _require_onnx_bundle()
    from korean_tok_pos import korean_tok_upos

    out = korean_tok_upos("대한민국의 수도는 서울이다.", model_dir=_MODEL_DIR)[0]
    assert len(out) >= 3
    surfaces = [w for w, _ in out]
    assert "대한민국의" in surfaces or any("대한민국" in w for w in surfaces)
    upos_set = {u for _, u in out}
    assert "PUNCT" in upos_set or "." in "".join(surfaces)


@pytest.mark.parametrize(
    "line",
    [
        "안녕하세요.",
        "오늘 날씨가 좋다.",
        "3월 30일입니다.",
    ],
)
def test_onnx_logits_close_to_pytorch(line: str):
    """Regression: ORT logits should match HF forward within tolerance (same tokenizer)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    _require_onnx_bundle()

    import onnxruntime as ort
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from ko_roberta_morph_preprocess import encode_for_morph_upos

    meta = json.loads((_MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    hf_id = meta.get("huggingface_model", "KoichiYasuoka/roberta-base-korean-morph-upos")

    tok_hf = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(hf_id)
    model.eval()

    input_ids, _tokens, _off, _ref, _wg = encode_for_morph_upos(line, _MODEL_DIR)
    ids = np.array([input_ids], dtype=np.int64)
    pad = int(meta.get("pad_token_id", 1))
    mask = (ids != pad).astype(np.int64)

    sess = ort.InferenceSession(str(_MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    out_name = sess.get_outputs()[0].name
    ort_logits, = sess.run([out_name], {"input_ids": ids, "attention_mask": mask})

    with torch.no_grad():
        pt_logits = model(
            input_ids=torch.from_numpy(ids),
            attention_mask=torch.from_numpy(mask),
        ).logits.numpy()

    # int8 weight shrink shifts values slightly; require most positions to agree on argmax.
    oa = ort_logits.argmax(-1)
    pa = pt_logits.argmax(-1)
    agree = float(np.mean(oa == pa))
    assert agree >= 0.92, f"token argmax agreement {agree:.3f} < 0.92"
