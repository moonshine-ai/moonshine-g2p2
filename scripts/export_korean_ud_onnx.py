#!/usr/bin/env python3
"""
Export ``KoichiYasuoka/roberta-base-korean-morph-upos`` to ONNX for :mod:`korean_tok_pos`.

Requires: torch, onnx, transformers, tokenizers, onnx-shrink-ray (optional shrink).

Default output: ``data/ko/roberta_korean_morph_upos_onnx/``:
  - ``model.onnx`` — ``RobertaForTokenClassification`` logits; inputs ``input_ids``, ``attention_mask``.
    By default, large weights are packed with onnx-shrink-ray (int8 storage + float restore).
  - ``meta.json`` — ``id2label`` list, pad/cls/sep ids, Hugging Face model id
  - tokenizer files: ``vocab.txt`` + ``tokenizer_config.json`` are **required** for the pure-Python
    runtime; ``tokenizer.json`` is also saved for optional parity tests against the HF ``tokenizers`` lib.

Model card: https://huggingface.co/KoichiYasuoka/roberta-base-korean-morph-upos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_HF_MODEL = "KoichiYasuoka/roberta-base-korean-morph-upos"


def _patch_onnx_for_onnx_graphsurgeon() -> None:
    """``onnx-graphsurgeon`` (used by onnx-shrink-ray) expects these helpers on ``onnx.helper``."""
    import onnx.helper as h

    if not hasattr(h, "float32_to_bfloat16"):

        def float32_to_bfloat16(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            ui = x.view(np.uint32)
            return ((ui + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)

        h.float32_to_bfloat16 = float32_to_bfloat16  # type: ignore[method-assign]

    if not hasattr(h, "float32_to_float8e4m3"):

        def float32_to_float8e4m3(x, fn=True, uz=False):
            raise RuntimeError(
                "float32_to_float8e4m3 compat stub: not used for Korean ONNX export"
            )

        h.float32_to_float8e4m3 = float32_to_float8e4m3  # type: ignore[method-assign]


def _shrink_onnx_weights(
    onnx_path: Path,
    *,
    min_elements: int,
    verbose: bool,
) -> None:
    _patch_onnx_for_onnx_graphsurgeon()
    try:
        from onnx_shrink_ray.shrink import quantize_weights
    except ImportError as e:
        raise SystemExit(
            "ONNX shrink requires onnx-shrink-ray (pip install onnx-shrink-ray). "
            "Or pass --no-shrink-weights for a full-precision FP32 file."
        ) from e

    import onnx

    model = onnx.load(str(onnx_path))
    new_model = quantize_weights(
        model,
        min_elements=min_elements,
        float_quantization=False,
        verbose=verbose,
    )
    sidecar = onnx_path.parent / (onnx_path.name + ".data")
    if sidecar.is_file():
        sidecar.unlink()
    onnx.save(new_model, str(onnx_path))


def _clamp_onnx_ir_version(onnx_path: Path, *, max_ir: int) -> None:
    """Torch + shrink can emit IR > 11; many onnxruntime builds only load up to IR 11."""
    import onnx

    model = onnx.load(str(onnx_path))
    if model.ir_version <= max_ir:
        return
    model.ir_version = max_ir
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


class _TokenClassificationOnnx(nn.Module):
    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Korean RoBERTa morph+UPOS to ONNX.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "data" / "ko" / "roberta_korean_morph_upos_onnx",
        help="Output directory",
    )
    ap.add_argument(
        "--hf-model",
        type=str,
        default=DEFAULT_HF_MODEL,
        help="Hugging Face model id or local path",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--no-shrink-weights",
        action="store_true",
        help="Skip onnx-shrink-ray (keep raw FP32 initializers).",
    )
    ap.add_argument(
        "--no-shrink-encoder-weights",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--shrink-min-elements",
        type=int,
        default=16 * 1024,
        help="Minimum tensor size (elements) to pack as int8.",
    )
    ap.add_argument("--shrink-verbose", action="store_true")
    ap.add_argument(
        "--max-onnx-ir",
        type=int,
        default=11,
        help="Clamp ONNX IR version for onnxruntime (default 11).",
    )
    args = ap.parse_args()
    no_shrink = args.no_shrink_weights or args.no_shrink_encoder_weights

    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    print("Loading", args.hf_model, file=sys.stderr)
    model = AutoModelForTokenClassification.from_pretrained(args.hf_model)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model.eval()
    wrapped = _TokenClassificationOnnx(model).eval()

    seq = 16
    dummy_ids = torch.ones(1, seq, dtype=torch.long)
    dummy_mask = torch.ones(1, seq, dtype=torch.long)
    onnx_path = out / "model.onnx"
    torch.onnx.export(
        wrapped,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset,
    )

    if not no_shrink:
        _shrink_onnx_weights(
            onnx_path,
            min_elements=args.shrink_min_elements,
            verbose=args.shrink_verbose,
        )

    _clamp_onnx_ir_version(onnx_path, max_ir=args.max_onnx_ir)

    cfg = model.config
    id2label = [cfg.id2label[i] for i in range(cfg.num_labels)]
    meta = {
        "huggingface_model": args.hf_model,
        "num_labels": int(cfg.num_labels),
        "pad_token_id": int(cfg.pad_token_id),
        "cls_token_id": int(cfg.bos_token_id),
        "sep_token_id": int(cfg.eos_token_id),
        "id2label": id2label,
        "onnx_model_file": "model.onnx",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer.save_pretrained(out)

    print("Wrote", onnx_path, "and", out / "meta.json")
    if no_shrink:
        print("(weights: FP32 initializers)")
    else:
        print("(weights: int8 storage + Cast/Mul/Add → float, onnx-shrink-ray)")


if __name__ == "__main__":
    main()
