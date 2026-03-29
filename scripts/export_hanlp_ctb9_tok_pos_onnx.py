#!/usr/bin/env python3
"""
Export HanLP **CTB9 ELECTRA-small** CWS + POS PyTorch checkpoints to ONNX.

Writes under ``models/zh_hans/hanlp_ctb9_electra_small/`` by default::

    tok.onnx   — char-level segmentation logits (BMES), CLS/SEP stripped in-graph
    pos.onnx   — word-level POS logits (needs ``token_span`` from the same tokenizer recipe)
    tokenizer/ — HuggingFace ``BertTokenizer`` files (shared by both tasks)
    metadata.json — tag vocabularies and export notes

Requires::

    pip install hanlp torch onnx onnxruntime 'transformers>=4.40,<5' tokenizers

The POS graph is exported with the **legacy** TorchScript ONNX exporter
(``dynamo=False``) because HanLP’s ``transformer_encode`` contains Python
control flow on sequence length that the default exporter mishandles.

**Limitation:** Traced graphs bake in the branch ``len(input_ids) <= 512`` (typical
Electra max length). Sentences whose subword sequence exceeds that need to be
split upstream (same practical limit as the PyTorch path for these models).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO_ROOT / "models" / "zh_hans" / "hanlp_ctb9_electra_small"


def _tag_vocab(component) -> list[str]:
    v = component.vocabs.tag
    # idx_to_token may be list aligned by index
    if isinstance(v.idx_to_token, list):
        return list(v.idx_to_token)
    return [v.idx_to_token[i] for i in range(len(v))]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output directory (default: {_DEFAULT_OUT})",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14).",
    )
    args = p.parse_args(argv)

    import torch
    import torch.nn as nn
    import hanlp
    import hanlp.pretrained.tok as ptok
    import hanlp.pretrained.pos as ppos

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HanLP CTB9 ELECTRA-small tok + pos (CPU) …")
    tok = hanlp.load(ptok.CTB9_TOK_ELECTRA_SMALL, devices=-1)
    pos = hanlp.load(ppos.CTB9_POS_ELECTRA_SMALL, devices=-1)
    tok.model.eval()
    pos.model.eval()

    tok.transformer_tokenizer.save_pretrained(str(tok_dir))

    span_w = 16
    meta = {
        "source_tok": ptok.CTB9_TOK_ELECTRA_SMALL,
        "source_pos": ppos.CTB9_POS_ELECTRA_SMALL,
        "tag_vocab_tok": _tag_vocab(tok),
        "tag_vocab_pos": _tag_vocab(pos),
        "pos_token_span_inner": span_w,
        "note": "POS token_span is [B, W, S_max] with 0 padding. Offline inference: char_normalize.json + scripts/ctb9_electra_onnx_pure.py.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    tr = pos.config.get("transform") or tok.config.get("transform")
    cmap: dict = {}
    if tr is not None and hasattr(tr, "_table"):
        cmap = dict(tr._table)
    (out_dir / "char_normalize.json").write_text(
        json.dumps(cmap, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'char_normalize.json'} ({len(cmap)} entries).")

    class TokWrap(nn.Module):
        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.encoder = m.encoder
            self.classifier = m.classifier

        def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.Tensor:
            h = self.encoder(
                input_ids, attention_mask=attention_mask, token_type_ids=None, token_span=None
            )
            return self.classifier(h)[:, 1:-1, :]

    class PosWrap(nn.Module):
        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.encoder = m.encoder
            self.classifier = m.classifier

        def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            token_span: torch.LongTensor,
        ) -> torch.Tensor:
            h = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                token_span=token_span,
            )
            return self.classifier(h)

    # Example tensors for POS trace (token_span inner width must be >= HanLP's padded width)
    ex_ids = torch.tensor([[101, 3144, 2110, 3221, 4906, 2110, 511, 102]], dtype=torch.long)
    ex_attn = (ex_ids != 0).long()
    ex_span = torch.zeros((1, 4, span_w), dtype=torch.long)
    ex_span[0, 0, :2] = torch.tensor([1, 2])
    ex_span[0, 1, :1] = 3
    ex_span[0, 2, :3] = torch.tensor([4, 5, 6])
    ex_span[0, 3, :2] = torch.tensor([7, 0])

    tok_path = out_dir / "tok.onnx"
    pos_path = out_dir / "pos.onnx"

    print(f"Exporting tok → {tok_path} …")
    torch.onnx.export(
        TokWrap(tok.model),
        (ex_ids, ex_attn),
        str(tok_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "char"},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    print(f"Exporting pos → {pos_path} …")
    torch.onnx.export(
        PosWrap(pos.model),
        (ex_ids, ex_attn, ex_span),
        str(pos_path),
        input_names=["input_ids", "attention_mask", "token_span"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_span": {0: "batch", 1: "words", 2: "span_width"},
            "logits": {0: "batch", 1: "words"},
        },
        opset_version=args.opset,
        dynamo=False,
    )

    print("Done.")


if __name__ == "__main__":
    main()
