#!/usr/bin/env python3
"""
Simplified-Chinese **word segmentation** + **UD UPOS** via ONNX Runtime, using
``KoichiYasuoka/chinese-roberta-base-upos`` (BIO tagging over WordPiece).

Uses the same ``vocab.txt`` + ``tokenizer_config.json`` WordPiece path as
:mod:`ko_roberta_wordpiece` (matches C++ ``ChineseTokPosOnnx``).

**Runtime:** ``onnxruntime``, ``numpy``, stdlib only.

Assets: ``data/zh_hans/roberta_chinese_base_upos_onnx/`` (``scripts/export_chinese_roberta_upos_onnx.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from ko_roberta_wordpiece import encode_bert_wordpiece

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "zh_hans" / "roberta_chinese_base_upos_onnx"


def _bio_decode_words(
    ref_text: str,
    tokens: Sequence[str],
    offsets: Sequence[tuple[int, int]],
    pred_ids: Sequence[int],
    id2label: Sequence[str],
    *,
    cls_token: str,
    sep_token: str,
) -> List[Tuple[str, str]]:
    """Merge BIO labels into (surface, UPOS) using character offsets into *ref_text*."""
    pairs: List[Tuple[str, str]] = []
    cur_spans: list[tuple[int, int]] = []
    cur_tag: str | None = None

    def flush() -> None:
        nonlocal cur_spans, cur_tag
        if not cur_spans:
            cur_tag = None
            return
        # ``ref_text`` has spaces around CJK chars; spans are per piece — join slices, not [first:last].
        surf = "".join(ref_text[s:e] for s, e in cur_spans)
        pairs.append((surf, cur_tag or "X"))
        cur_spans = []
        cur_tag = None

    for ti, tok in enumerate(tokens):
        s, e = offsets[ti]
        if tok in (cls_token, sep_token):
            flush()
            continue
        lab = id2label[int(pred_ids[ti])]
        if lab.startswith("B-"):
            flush()
            cur_tag = lab[2:]
            cur_spans = [(s, e)]
        elif lab.startswith("I-"):
            suf = lab[2:]
            if not cur_spans:
                cur_tag = suf
                cur_spans = [(s, e)]
            else:
                cur_spans.append((s, e))
                if cur_tag is None:
                    cur_tag = suf
        else:
            flush()
            pairs.append((ref_text[s:e], lab))
    flush()
    return pairs


class ChineseTokPosOnnx:
    def __init__(self, model_dir: Path | None = None, *, providers: Sequence[str] | None = None):
        self.model_dir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        for name in ("vocab.txt", "tokenizer_config.json"):
            p = self.model_dir / name
            if not p.is_file():
                raise FileNotFoundError(
                    f"Missing {p}; run scripts/export_chinese_roberta_upos_onnx.py (exports tokenizer)."
                )
        meta_path = self.model_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing {meta_path}")
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._id2label: List[str] = list(self._meta["id2label"])
        self._pad_id = int(self._meta.get("pad_token_id", 0))
        self._max_seq = int(self._meta.get("max_sequence_length", 512))

        onnx_name = self._meta.get("onnx_model_file", "model.onnx")
        onnx_path = self.model_dir / onnx_name
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Missing {onnx_path}")

        cfg = json.loads((self.model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        self._cls = str(cfg["cls_token"])
        self._sep = str(cfg["sep_token"])

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = list(providers) if providers else ort.get_available_providers()
        self._sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=prov)
        outs = self._sess.get_outputs()
        self._logits_output = outs[0].name

    def annotate(self, text: str) -> List[Tuple[str, str]]:
        if not text.strip():
            return []
        input_ids, tokens, offsets, ref_text = encode_bert_wordpiece(text, self.model_dir)
        if len(input_ids) > self._max_seq:
            raise ValueError(
                f"ChineseTokPosOnnx: sequence length {len(input_ids)} > max {self._max_seq}"
            )
        ids = np.array([input_ids], dtype=np.int64)
        mask = (ids != self._pad_id).astype(np.int64)
        logits, = self._sess.run(
            [self._logits_output],
            {"input_ids": ids, "attention_mask": mask},
        )
        logits = np.asarray(logits, dtype=np.float32)[0]
        pred = logits.argmax(axis=-1).astype(int).tolist()
        return _bio_decode_words(
            ref_text,
            tokens,
            offsets,
            pred,
            self._id2label,
            cls_token=self._cls,
            sep_token=self._sep,
        )


def chinese_tok_upos(
    text: Union[str, Sequence[str]],
    *,
    model_dir: Path | None = None,
    onnx_session: ChineseTokPosOnnx | None = None,
) -> List[List[Tuple[str, str]]]:
    pipe = onnx_session or ChineseTokPosOnnx(model_dir=model_dir)
    if isinstance(text, str):
        return [pipe.annotate(text)]
    return [pipe.annotate(t) for t in text]


def main() -> None:
    p = argparse.ArgumentParser(description="Chinese WordPiece + UPOS (ONNX RoBERTa BIO).")
    p.add_argument("text", nargs="?", default="上海是一座城市。")
    p.add_argument("--model-dir", type=Path, default=None)
    args = p.parse_args()
    for w, tag in chinese_tok_upos(args.text, model_dir=args.model_dir)[0]:
        print(f"{w}/{tag}", end=" ")
    print()


if __name__ == "__main__":
    main()
