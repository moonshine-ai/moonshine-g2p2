"""ONNX+NumPy pipeline must match HanLP+PyTorch reference (when both are installed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_MODEL_DIR = _REPO / "data" / "ko" / "hanlp_ud_onnx"
_TESTS_DIR = Path(__file__).resolve().parent


def _reference_module():
    path = _TESTS_DIR / "reference_korean_ud_hanlp.py"
    spec = importlib.util.spec_from_file_location("reference_korean_ud_hanlp", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _require_onnx_bundle():
    if not (_MODEL_DIR / "encoder.onnx").is_file() or not (_MODEL_DIR / "heads.npz").is_file():
        pytest.skip("Run: python scripts/export_korean_ud_onnx.py")


def test_preprocessor_ids_match_hanlp():
    pytest.importorskip("hanlp")
    _require_onnx_bundle()
    from ko_ud_mminilm_preprocess import build_tok_batch
    import hanlp
    from hanlp.pretrained import mtl

    s = "대한민국의 수도는 서울이다."
    tok_path = str(_MODEL_DIR / "tokenizer.json")
    ids, span, units = build_tok_batch(s, tok_path)

    nlp = hanlp.load(mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6, devices=-1)
    task = nlp["tok"]
    enc_xf, xf = nlp.build_transform(task)
    tlist = xf[: xf.index_by_type(type(enc_xf)) + 1]
    tlist.append(task.last_transform())
    sample: dict = {"token": s}
    for tr in tlist:
        sample = tr(sample)
    assert ids == sample["token_input_ids"]
    assert span == sample["token_token_span"]
    assert units == sample["token"]


def test_onnx_matches_hanlp_tok_upos():
    pytest.importorskip("hanlp")
    pytest.importorskip("torch")
    _require_onnx_bundle()
    import hanlp
    from hanlp.pretrained import mtl

    from korean_tok_pos import korean_tok_upos
    ref_mod = _reference_module()
    korean_tok_upos_hanlp_reference = ref_mod.korean_tok_upos_hanlp_reference

    ref_nlp = hanlp.load(
        mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6,
        devices=-1,
    )
    cases = [
        "안녕",
        "대한민국의 수도는 서울이다.",
        "감사합니다.",
    ]
    for s in cases:
        expected = korean_tok_upos_hanlp_reference(s, nlp=ref_nlp)
        hyp = korean_tok_upos(s, model_dir=_MODEL_DIR)
        assert hyp == expected, f"mismatch for {s!r}:\n  ref={expected!r}\n  hyp={hyp!r}"


def test_onnx_batch_matches_reference():
    pytest.importorskip("hanlp")
    pytest.importorskip("torch")
    _require_onnx_bundle()
    import hanlp
    from hanlp.pretrained import mtl

    from korean_tok_pos import korean_tok_upos
    ref_mod = _reference_module()
    korean_tok_upos_hanlp_reference = ref_mod.korean_tok_upos_hanlp_reference

    ref_nlp = hanlp.load(
        mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6,
        devices=-1,
    )
    lines = ["안녕하세요.", "감사합니다."]
    expected = korean_tok_upos_hanlp_reference(lines, nlp=ref_nlp)
    hyp = korean_tok_upos(lines, model_dir=_MODEL_DIR)
    assert hyp == expected
