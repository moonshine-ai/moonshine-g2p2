#!/usr/bin/env python3
"""
Export ``AbderrahmanSkiredj1/arabertv02_tashkeel_fadel`` (BERT token classification) to ONNX for
:mod:`arabic_diac_onnx_infer` / C++ ``ArabicDiacOnnx``.

**Build-time deps (not used at G2P runtime):** ``torch``, ``transformers``, ``onnx``. Optional: ``onnx-shrink-ray``
(plus ``onnx-graphsurgeon``) to pack FP32 weights into int8 initializers with dequant at runtime (smaller
``model.onnx``; same float compute graph).

Recent ``transformers`` + legacy ``torch.onnx.export(dynamo=False)`` can break inside ``masking_utils`` when
sequence length is traced as a tensor; this script expands ``attention_mask`` to 4D first to avoid that path.

Output layout (default ``data/ar_msa/arabertv02_tashkeel_fadel_onnx/``)::

    model.onnx
    meta.json
    vocab.txt
    tokenizer_config.json
    (other tokenizer files from ``save_pretrained``)

Example::

    pip install torch transformers onnx onnxruntime onnx-shrink-ray onnx-graphsurgeon
    python scripts/export_arabic_msa_diacritizer_onnx.py --out-dir data/ar_msa/arabertv02_tashkeel_fadel_onnx

Shrink an existing FP32 export in place (no PyTorch)::

    python scripts/export_arabic_msa_diacritizer_onnx.py --only-shrink \\
      --out-dir data/ar_msa/arabertv02_tashkeel_fadel_onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO / "data" / "ar_msa" / "arabertv02_tashkeel_fadel_onnx"
_MODEL_ID = "AbderrahmanSkiredj1/arabertv02_tashkeel_fadel"


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
            "ONNX shrink requires onnx-shrink-ray. Install with "
            "`pip install onnx-shrink-ray onnx-graphsurgeon` or pass --no-shrink-weights."
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


def write_vocab_txt(tok, path: Path) -> None:
    """C++ ``ArabicDiacOnnx`` loads WordPiece ids from ``vocab.txt`` (one token per line, id = line index)."""
    pairs = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    path.write_text("\n".join(t for t, _ in pairs) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--model-id", type=str, default=_MODEL_ID)
    ap.add_argument(
        "--only-shrink",
        action="store_true",
        help="Skip PyTorch export; quantize weights in existing out-dir/model.onnx only.",
    )
    ap.add_argument("--no-shrink-weights", action="store_true")
    ap.add_argument("--shrink-min-elements", type=int, default=16 * 1024)
    ap.add_argument("--shrink-verbose", action="store_true")
    ap.add_argument("--max-onnx-ir", type=int, default=11)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    onnx_path = out / "model.onnx"
    if args.only_shrink:
        if not onnx_path.is_file():
            raise SystemExit(f"--only-shrink: missing {onnx_path}")
        if args.no_shrink_weights:
            raise SystemExit("--only-shrink conflicts with --no-shrink-weights")
        print("Shrinking weights (onnx-shrink-ray, int8 storage + dequant)…", file=sys.stderr)
        _shrink_onnx_weights(
            onnx_path,
            min_elements=args.shrink_min_elements,
            verbose=args.shrink_verbose,
        )
        _clamp_onnx_ir_version(onnx_path, max_ir=args.max_onnx_ir)
        print("Updated", onnx_path)
        return

    tok = AutoTokenizer.from_pretrained(args.model_id)
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_id,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForTokenClassification.from_pretrained(args.model_id)
    model.eval()
    tok.save_pretrained(out)
    write_vocab_txt(tok, out / "vocab.txt")

    class Wrap(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            # Under torch.jit trace, inputs_embeds.shape[1] becomes a Tensor; Transformers then mis-handles it
            # inside sdpa_mask (BC branch for deprecated cache_position).  A 4D additive mask skips mask creation.
            am = attention_mask.to(dtype=torch.float32)
            am = am[:, None, None, :]
            am = (1.0 - am) * torch.finfo(torch.float32).min
            return self.m(input_ids=input_ids, attention_mask=am).logits

    w = Wrap(model)
    w.eval()
    dummy = torch.ones(1, 8, dtype=torch.long)
    mask = torch.ones_like(dummy)
    export_kw = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    try:
        torch.onnx.export(
            w,
            (dummy, mask),
            str(onnx_path),
            dynamo=False,
            **export_kw,
        )
    except TypeError:
        torch.onnx.export(w, (dummy, mask), str(onnx_path), **export_kw)

    raw_labels = model.config.id2label
    keys = list(raw_labels.keys())
    if keys and isinstance(keys[0], int):
        id2label = [raw_labels[i] for i in range(len(raw_labels))]
    else:
        id2label = [raw_labels[str(i)] for i in range(len(raw_labels))]
    meta = {
        "pad_token_id": int(model.config.pad_token_id),
        "max_sequence_length": int(getattr(model.config, "max_position_embeddings", 512)),
        "onnx_model_file": "model.onnx",
        "id2label": id2label,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_shrink_weights:
        _shrink_onnx_weights(
            onnx_path,
            min_elements=args.shrink_min_elements,
            verbose=args.shrink_verbose,
        )
    _clamp_onnx_ir_version(onnx_path, max_ir=args.max_onnx_ir)

    print("Wrote", onnx_path)
    if args.no_shrink_weights:
        print("(weights: FP32 initializers)")
    else:
        print("(weights: int8 storage via onnx-shrink-ray)")


if __name__ == "__main__":
    main()
