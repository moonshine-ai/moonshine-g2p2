#!/usr/bin/env python3
"""
Export ``KoichiYasuoka/chinese-roberta-base-upos`` to ONNX for :mod:`chinese_tok_pos_onnx`.

Requires: torch, onnx, transformers. Default output:
``data/zh_hans/roberta_chinese_base_upos_onnx/``.

Model card: https://huggingface.co/KoichiYasuoka/chinese-roberta-base-upos
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


DEFAULT_HF_MODEL = "KoichiYasuoka/chinese-roberta-base-upos"


def _patch_onnx_for_onnx_graphsurgeon() -> None:
    import onnx.helper as h

    if not hasattr(h, "float32_to_bfloat16"):

        def float32_to_bfloat16(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            ui = x.view(np.uint32)
            return ((ui + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)

        h.float32_to_bfloat16 = float32_to_bfloat16  # type: ignore[method-assign]

    if not hasattr(h, "float32_to_float8e4m3"):

        def float32_to_float8e4m3(x, fn=True, uz=False):
            raise RuntimeError("float32_to_float8e4m3 compat stub: not used")

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
            "ONNX shrink requires onnx-shrink-ray. Or pass --no-shrink-weights."
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
    ap = argparse.ArgumentParser(description="Export Chinese RoBERTa UPOS (BIO) to ONNX.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "data" / "zh_hans" / "roberta_chinese_base_upos_onnx",
        help="Output directory",
    )
    ap.add_argument("--hf-model", type=str, default=DEFAULT_HF_MODEL)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-shrink-weights", action="store_true")
    ap.add_argument("--shrink-min-elements", type=int, default=16 * 1024)
    ap.add_argument("--shrink-verbose", action="store_true")
    ap.add_argument("--max-onnx-ir", type=int, default=11)
    args = ap.parse_args()

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

    if not args.no_shrink_weights:
        _shrink_onnx_weights(
            onnx_path,
            min_elements=args.shrink_min_elements,
            verbose=args.shrink_verbose,
        )

    _clamp_onnx_ir_version(onnx_path, max_ir=args.max_onnx_ir)

    cfg = model.config
    raw_lab = dict(cfg.id2label)
    id2label: list[str] = []
    for i in range(cfg.num_labels):
        id2label.append(raw_lab.get(i, raw_lab.get(str(i), f"LABEL_{i}")))
    max_len = int(getattr(cfg, "max_position_embeddings", 512))
    pad_id = int(cfg.pad_token_id) if cfg.pad_token_id is not None else 0
    meta = {
        "huggingface_model": args.hf_model,
        "num_labels": int(cfg.num_labels),
        "pad_token_id": pad_id,
        "id2label": id2label,
        "onnx_model_file": "model.onnx",
        "max_sequence_length": max_len,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer.save_pretrained(out)

    print("Wrote", onnx_path, "and", out / "meta.json")
    if args.no_shrink_weights:
        print("(weights: FP32 initializers)")
    else:
        print("(weights: int8 storage via onnx-shrink-ray)")


if __name__ == "__main__":
    main()
