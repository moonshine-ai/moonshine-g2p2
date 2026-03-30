#!/usr/bin/env python3
"""
Japanese whitespace-level **long-unit words** + UD UPOS via ONNX Runtime.

Uses ``KoichiYasuoka/roberta-small-japanese-char-luw-upos`` (RoBERTa token classification),
exported to ONNX. Tokenization matches HuggingFace ``BertTokenizer`` / this repo’s
``ko_roberta_wordpiece.encode_bert_wordpiece`` (same JSON + ``vocab.txt`` layout).

**Runtime:** ``onnxruntime``, ``numpy``, stdlib only.

Assets: ``data/ja/roberta_japanese_char_luw_upos_onnx/`` (``python scripts/export_japanese_ud_onnx.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from ko_roberta_morph_preprocess import encode_for_morph_upos, morph_label_to_upos

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "ja" / "roberta_japanese_char_luw_upos_onnx"


class JapaneseTokPosOnnx:
    def __init__(self, model_dir: Path | None = None, *, providers: Sequence[str] | None = None):
        self.model_dir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        for name in ("vocab.txt", "tokenizer_config.json"):
            p = self.model_dir / name
            if not p.is_file():
                raise FileNotFoundError(
                    f"Missing {p}; run scripts/export_japanese_ud_onnx.py (exports vocab + config)."
                )
        meta_path = self.model_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing {meta_path}")
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._id2label: List[str] = self._meta["id2label"]
        self._pad_id = int(self._meta.get("pad_token_id", 0))
        self._max_seq = int(self._meta.get("max_sequence_length", 512))

        onnx_name = self._meta.get("onnx_model_file", "model.onnx")
        onnx_path = self.model_dir / onnx_name
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Missing {onnx_path}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = list(providers) if providers else ort.get_available_providers()
        self._sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=prov)
        outs = self._sess.get_outputs()
        self._logits_output = outs[0].name

    def annotate(self, text: str) -> List[Tuple[str, str]]:
        if not text.strip():
            return []
        input_ids, _tokens, offsets, ref_text, word_groups = encode_for_morph_upos(text, self.model_dir)
        if len(input_ids) > self._max_seq:
            raise ValueError(
                f"JapaneseTokPosOnnx: sequence length {len(input_ids)} > max {self._max_seq}"
            )
        ids = np.array([input_ids], dtype=np.int64)
        mask = (ids != self._pad_id).astype(np.int64)
        logits, = self._sess.run(
            [self._logits_output],
            {"input_ids": ids, "attention_mask": mask},
        )
        logits = np.asarray(logits, dtype=np.float32)[0]
        pairs: List[Tuple[str, str]] = []
        for g in word_groups:
            if not g:
                continue
            pooled = logits[g].mean(axis=0)
            lid = int(np.argmax(pooled))
            raw_label = self._id2label[lid]
            upos = morph_label_to_upos(raw_label)
            st = offsets[g[0]][0]
            en = offsets[g[-1]][1]
            pairs.append((ref_text[st:en], upos))
        return pairs


def japanese_tok_upos(
    text: Union[str, Sequence[str]],
    *,
    model_dir: Path | None = None,
    onnx_session: JapaneseTokPosOnnx | None = None,
) -> List[List[Tuple[str, str]]]:
    pipe = onnx_session or JapaneseTokPosOnnx(model_dir=model_dir)
    if isinstance(text, str):
        return [pipe.annotate(text)]
    return [pipe.annotate(t) for t in text]


def main() -> None:
    p = argparse.ArgumentParser(description="Japanese LUW + UPOS (ONNX RoBERTa char-luw-upos).")
    p.add_argument("text", nargs="?", default="国境の長いトンネルを抜けると雪国であった。")
    p.add_argument("--model-dir", type=Path, default=None)
    args = p.parse_args()
    for pair in japanese_tok_upos(args.text, model_dir=args.model_dir)[0]:
        print(f"{pair[0]}/{pair[1]}", end=" ")
    print()


if __name__ == "__main__":
    main()
