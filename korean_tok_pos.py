#!/usr/bin/env python3
"""
Korean tokenization + Universal Dependencies UPOS using ONNX Runtime and NumPy.

Requires exported assets under ``data/ko/hanlp_ud_onnx/`` (run
``python scripts/export_korean_ud_onnx.py``). Dependencies: ``onnxruntime``,
``numpy``, ``tokenizers``.

There is **no** PyTorch or HanLP on this code path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort

from ko_ud_mminilm_preprocess import build_tok_batch
from ko_ud_numpy_heads import adaptive_log_prob, bmes_to_words, load_heads, pool_word_hidden, tok_logits

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "ko" / "hanlp_ud_onnx"


def _pick_token_hidden(subword_h: np.ndarray, token_span: Sequence[Sequence[int]]) -> np.ndarray:
    """Average subword vectors for each span row (CLS / chars / SEP)."""
    rows: List[np.ndarray] = []
    for g in token_span:
        rows.append(subword_h[g].mean(axis=0))
    return np.stack(rows, axis=0)


def _word_row_spans_from_units(units: Sequence[str], words: Sequence[str]) -> List[Tuple[int, int]]:
    """Map each segmented word to half-open row ranges in ``char_h`` (row 0 = CLS)."""
    idx = 0
    spans: List[Tuple[int, int]] = []
    for w in words:
        buf = ""
        start = idx
        while buf != w:
            if idx >= len(units):
                raise ValueError(f"BMES word {w!r} does not align with units {units!r}")
            buf += units[idx]
            idx += 1
        spans.append((1 + start, 1 + idx))
    if idx != len(units):
        raise ValueError("BMES segmentation does not consume all units")
    return spans


class KoreanTokPosOnnx:
    def __init__(self, model_dir: Path | None = None, *, providers: Sequence[str] | None = None):
        self.model_dir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        tok_path = self.model_dir / "tokenizer.json"
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Missing {tok_path}; run scripts/export_korean_ud_onnx.py or copy tokenizer assets."
            )
        self._tokenizer_json = str(tok_path)
        self.meta, self.weights = load_heads(self.model_dir)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = list(providers) if providers else ort.get_available_providers()
        self._sess = ort.InferenceSession(
            str(self.model_dir / "encoder.onnx"),
            sess_options=so,
            providers=prov,
        )

    def annotate(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []
        input_ids, token_span, units = build_tok_batch(text, self._tokenizer_json)
        ids = np.array([input_ids], dtype=np.int64)
        mask = (ids != self.meta["pad_token_id"]).astype(np.int64)
        h_sub, = self._sess.run(
            ["last_hidden_state"],
            {"input_ids": ids, "attention_mask": mask},
        )
        h_sub = np.asarray(h_sub, dtype=np.float32)[0]
        char_h = _pick_token_hidden(h_sub, token_span)
        mid = char_h[1:-1]
        tl = tok_logits(mid, self.weights["tok_w"], self.weights["tok_b"])
        tag_ids = np.argmax(tl, axis=-1)
        tags_bm = [self.meta["tag_id_to_label"][int(i)] for i in tag_ids]
        words = bmes_to_words(units, tags_bm)
        spans = _word_row_spans_from_units(units, words)
        ud_h = pool_word_hidden(char_h, spans)
        lp = adaptive_log_prob(ud_h.astype(np.float32), self.meta, self.weights)
        upos_labels = self.meta["upos_id_to_label"]
        pairs: List[Tuple[str, str]] = []
        for i in range(1, lp.shape[0]):
            pid = int(np.argmax(lp[i]))
            pairs.append((words[i - 1], upos_labels[pid]))
        return pairs


def korean_tok_upos(
    text: Union[str, Sequence[str]],
    *,
    model_dir: Path | None = None,
    onnx_session: KoreanTokPosOnnx | None = None,
) -> List[List[Tuple[str, str]]]:
    """
    Tokenize and tag UPOS for one string or a list of strings.

    Returns one list of ``(token, upos)`` pairs per input sentence.
    """
    pipe = onnx_session or KoreanTokPosOnnx(model_dir=model_dir)
    if isinstance(text, str):
        return [pipe.annotate(text)]
    return [pipe.annotate(t) for t in text]


def main() -> None:
    p = argparse.ArgumentParser(description="Korean tok + UPOS (ONNX + NumPy).")
    p.add_argument("text", nargs="?", default="대한민국의 수도는 서울이다.")
    p.add_argument("--model-dir", type=Path, default=None)
    args = p.parse_args()
    for pair in korean_tok_upos(args.text, model_dir=args.model_dir)[0]:
        print(f"{pair[0]}/{pair[1]}", end=" ")
    print()


if __name__ == "__main__":
    main()
