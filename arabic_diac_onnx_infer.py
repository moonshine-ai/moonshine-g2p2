#!/usr/bin/env python3
"""
Arabic **diacritization** (tashkīl) via ONNX token classification (Arabert-style).

**Runtime:** ``onnxruntime``, ``numpy``, and :mod:`ko_roberta_wordpiece` (no ``transformers``).

Assets: ``data/ar_msa/arabertv02_tashkeel_fadel_onnx/`` from
``scripts/export_arabic_msa_diacritizer_onnx.py`` (``model.onnx``, ``vocab.txt``,
``tokenizer_config.json``, ``meta.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import onnxruntime as ort

from arabic_ipa import DIAC_LABEL_TO_UTF8, strip_arabic_diacritics
from ko_roberta_wordpiece import encode_bert_wordpiece

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _REPO_ROOT / "data" / "ar_msa" / "arabertv02_tashkeel_fadel_onnx"


def _anchor_index(ref: str, s: int, e: int) -> int | None:
    from arabic_ipa import is_arabic_base_letter

    j: int | None = None
    for k in range(s, min(e, len(ref))):
        if is_arabic_base_letter(ref[k]):
            j = k
    return j


class ArabicDiacOnnx:
    def __init__(self, model_dir: Path | None = None, *, providers: Sequence[str] | None = None):
        self.model_dir = Path(model_dir) if model_dir is not None else _DEFAULT_MODEL_DIR
        for name in ("vocab.txt", "tokenizer_config.json", "meta.json"):
            p = self.model_dir / name
            if not p.is_file():
                raise FileNotFoundError(
                    f"Missing {p}; run scripts/export_arabic_msa_diacritizer_onnx.py"
                )
        self._meta = json.loads((self.model_dir / "meta.json").read_text(encoding="utf-8"))
        self._id2label: List[str] = list(self._meta["id2label"])
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
        self._logits_output = self._sess.get_outputs()[0].name

    def diacritize(self, text: str) -> str:
        """Return NFC Arabic with predicted harakāt (best-effort)."""
        raw = text.strip()
        if not raw:
            return ""
        und = strip_arabic_diacritics(raw)
        if not und:
            return raw

        input_ids, tokens, offsets, ref = encode_bert_wordpiece(und, self.model_dir)
        if len(input_ids) > self._max_seq:
            keep = self._max_seq - 2
            inner_i = input_ids[1:-1][:keep]
            inner_o = offsets[1:-1][:keep]
            input_ids = [input_ids[0]] + inner_i + [input_ids[-1]]
            offsets = [offsets[0]] + inner_o + [offsets[-1]]

        ids = np.array([input_ids], dtype=np.int64)
        mask = (ids != self._pad_id).astype(np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        ins = self._sess.get_inputs()
        if len(ins) > 2 and ins[2].name == "token_type_ids":
            feed["token_type_ids"] = np.zeros_like(ids)

        logits, = self._sess.run([self._logits_output], feed)
        logits = np.asarray(logits, dtype=np.float32)[0]
        pred = logits.argmax(axis=-1).astype(int).tolist()

        diac_after: dict[int, str] = {}
        T = len(tokens)
        for ti in range(T):
            if ti == 0 or ti == T - 1:
                continue
            lab = self._id2label[int(pred[ti])]
            if lab == "X":
                continue
            diac = DIAC_LABEL_TO_UTF8.get(lab)
            if not diac:
                continue
            s, e = offsets[ti]
            j = _anchor_index(ref, s, e)
            if j is None:
                continue
            diac_after[j] = diac_after.get(j, "") + diac

        out: list[str] = []
        for i, ch in enumerate(ref):
            out.append(ch)
            if i in diac_after:
                out.append(diac_after[i])
        from arabic_ipa import nfc

        return nfc("".join(out))
