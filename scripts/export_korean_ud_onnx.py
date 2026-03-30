#!/usr/bin/env python3
"""
Export MiniLM subword encoder to ONNX and tok / UPOS decoder weights to NumPy.

Requires: hanlp, torch, onnx, tokenizers (export-time only).

Writes into ``--out-dir`` (default: ``data/ko/hanlp_ud_onnx/``):
  - ``encoder.onnx`` — raw transformer; inputs ``input_ids``, ``attention_mask`` (int64)
  - ``heads.npz`` — tok linear, adaptive UPOS matrices, JSON meta (vocabs, cutoffs)
  - copies tokenizer JSON assets next to the model
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "data" / "ko" / "hanlp_ud_onnx",
        help="Output directory",
    )
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    import hanlp
    from hanlp.pretrained import mtl

    nlp = hanlp.load(mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6, devices=-1)

    enc_mod = nlp.model.encoder

    class RawTransformer(nn.Module):
        def __init__(self, encoder_module):
            super().__init__()
            self.transformer = encoder_module.transformer

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            out = self.transformer(input_ids, attention_mask)
            return out[0]

    raw = RawTransformer(enc_mod).eval()
    dummy_ids = torch.ones(1, 8, dtype=torch.long)
    dummy_mask = torch.ones(1, 8, dtype=torch.long)
    torch.onnx.export(
        raw,
        (dummy_ids, dummy_mask),
        str(out / "encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset,
    )

    tok_dec = nlp.model.decoders["tok"]
    tok_w = tok_dec.classifier.weight.detach().cpu().numpy().astype(np.float32)
    tok_b = tok_dec.classifier.bias.detach().cpu().numpy().astype(np.float32)

    ud = nlp.model.decoders["ud"]
    asm = ud.decoders["upos"].task_output
    weights: dict[str, np.ndarray] = {
        "tok_w": tok_w,
        "tok_b": tok_b,
        "head_w": asm.head.weight.detach().cpu().numpy().astype(np.float32),
    }
    if asm.head.bias is not None:
        weights["head_b"] = asm.head.bias.detach().cpu().numpy().astype(np.float32)
    else:
        weights["head_b"] = np.zeros(asm.head.out_features, dtype=np.float32)

    for i, tail in enumerate(asm.tail):
        i2h, h2o = tail
        weights[f"tail_{i}_i2h_w"] = i2h.weight.detach().cpu().numpy().astype(np.float32)
        weights[f"tail_{i}_h2o_w"] = h2o.weight.detach().cpu().numpy().astype(np.float32)
        weights[f"tail_{i}_i2h_b"] = (
            i2h.bias.detach().cpu().numpy().astype(np.float32)
            if i2h.bias is not None
            else np.zeros(i2h.out_features, dtype=np.float32)
        )
        weights[f"tail_{i}_h2o_b"] = (
            h2o.bias.detach().cpu().numpy().astype(np.float32)
            if h2o.bias is not None
            else np.zeros(h2o.out_features, dtype=np.float32)
        )

    tag_vocab = nlp["tok"].vocabs["tag"]
    pos_vocab = nlp["ud"].vocabs.pos
    meta = {
        "cutoffs": list(asm.cutoffs),
        "n_classes": asm.n_classes,
        "shortlist_size": asm.shortlist_size,
        "n_clusters": asm.n_clusters,
        "hidden_size": int(tok_w.shape[1]),
        "pad_token_id": 0,
        "tag_id_to_label": [tag_vocab.idx_to_token[i] for i in range(len(tag_vocab))],
        "upos_id_to_label": [pos_vocab.idx_to_token[i] for i in range(len(pos_vocab))],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    np.savez_compressed(out / "heads.npz", **weights)

    tfm_dir = Path(enc_mod.transformer.name_or_path)
    if not tfm_dir.is_dir():
        print("Could not locate tokenizer directory; copy tokenizer.json manually", file=sys.stderr)
    else:
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ):
            src = tfm_dir / name
            if src.is_file():
                shutil.copy2(src, out / name)

    print("Wrote", out / "encoder.onnx", "and", out / "heads.npz")


if __name__ == "__main__":
    main()
