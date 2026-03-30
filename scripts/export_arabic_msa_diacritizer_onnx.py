#!/usr/bin/env python3
"""
Export ``AbderrahmanSkiredj1/arabertv02_tashkeel_fadel`` (BERT token classification) to ONNX for
:mod:`arabic_diac_onnx_infer` / C++ ``ArabicDiacOnnx``.

**Build-time deps (not used at G2P runtime):** ``torch``, ``transformers``, ``onnx``.

Recent ``transformers`` + legacy ``torch.onnx.export(dynamo=False)`` can break inside ``masking_utils`` when
sequence length is traced as a tensor; this script expands ``attention_mask`` to 4D first to avoid that path.

Output layout (default ``data/ar_msa/arabertv02_tashkeel_fadel_onnx/``)::

    model.onnx
    meta.json
    vocab.txt
    tokenizer_config.json
    (other tokenizer files from ``save_pretrained``)

Example::

    pip install torch transformers onnx onnxruntime
    python scripts/export_arabic_msa_diacritizer_onnx.py --out-dir data/ar_msa/arabertv02_tashkeel_fadel_onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO / "data" / "ar_msa" / "arabertv02_tashkeel_fadel_onnx"
_MODEL_ID = "AbderrahmanSkiredj1/arabertv02_tashkeel_fadel"


def write_vocab_txt(tok, path: Path) -> None:
    """C++ ``ArabicDiacOnnx`` loads WordPiece ids from ``vocab.txt`` (one token per line, id = line index)."""
    pairs = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    path.write_text("\n".join(t for t, _ in pairs) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--model-id", type=str, default=_MODEL_ID)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

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
    onnx_path = out / "model.onnx"
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
    print("Wrote", onnx_path)


if __name__ == "__main__":
    main()
