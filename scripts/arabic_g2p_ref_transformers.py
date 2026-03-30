#!/usr/bin/env python3
"""
**Reference / testing** MSA G2P using HuggingFace ``transformers`` + the same Arabert tashkīl
checkpoint as the ONNX export, then :mod:`arabic_ipa` for IPA.

Use this to validate ONNX parity (``onnxruntime``) against PyTorch logits on held-out sentences.

Install::

    pip install torch transformers

Example::

    python scripts/arabic_g2p_ref_transformers.py "القاهرة عاصمة مصر."
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arabic_ipa import (  # noqa: E402
    DIAC_LABEL_TO_UTF8,
    apply_onnx_partial_postprocess,
    strip_arabic_diacritics,
    word_to_ipa_with_assimilation,
)
from ko_roberta_wordpiece import encode_bert_wordpiece  # noqa: E402


def _anchor_index(ref: str, s: int, e: int) -> int | None:
    from arabic_ipa import is_arabic_base_letter

    j: int | None = None
    for k in range(s, min(e, len(ref))):
        if is_arabic_base_letter(ref[k]):
            j = k
    return j


def diacritize_torch(model_id: str, text: str, model_dir: Path | None) -> str:
    import torch
    from transformers import AutoModelForTokenClassification

    und = strip_arabic_diacritics(text.strip())
    if not und:
        return ""
    md = model_dir or (_REPO / "data" / "ar_msa" / "arabertv02_tashkeel_fadel_onnx")
    input_ids, _tokens, offsets, ref = encode_bert_wordpiece(und, md)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.eval()
    ids = torch.tensor([input_ids], dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits[0].numpy()
    pred = logits.argmax(axis=-1)
    raw = model.config.id2label
    keys = list(raw.keys())
    if keys and isinstance(keys[0], int):
        id2label = [raw[i] for i in range(len(raw))]
    else:
        id2label = [raw[str(i)] for i in range(len(raw))]
    diac_after: dict[int, str] = {}
    for ti in range(len(input_ids)):
        if ti == 0 or ti == len(input_ids) - 1:
            continue
        lab = id2label[int(pred[ti])]
        if lab == "X":
            continue
        di = DIAC_LABEL_TO_UTF8.get(lab)
        if not di:
            continue
        s, e = offsets[ti]
        j = _anchor_index(ref, s, e)
        if j is None:
            continue
        diac_after[j] = diac_after.get(j, "") + di
    out: list[str] = []
    for i, ch in enumerate(ref):
        out.append(ch)
        if i in diac_after:
            out.append(diac_after[i])
    return unicodedata.normalize("NFC", "".join(out))


def line_to_ipa(model_id: str, line: str, model_dir: Path | None) -> str:
    line = unicodedata.normalize("NFC", line.strip())
    if not line:
        return ""
    parts: list[str] = []
    for raw in line.split():
        w = raw.strip()
        if not w or not any(0x0600 <= ord(c) <= 0x06FF for c in w):
            continue
        diac = apply_onnx_partial_postprocess(diacritize_torch(model_id, w, model_dir))
        parts.append(word_to_ipa_with_assimilation(diac))
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--model-id", default="AbderrahmanSkiredj1/arabertv02_tashkeel_fadel")
    ap.add_argument("--tokenizer-dir", type=Path, default=None, help="Exported bundle dir (vocab + config)")
    args = ap.parse_args()
    if args.stdin or not args.text:
        text = sys.stdin.read()
    else:
        text = args.text
    print(line_to_ipa(args.model_id, text, args.tokenizer_dir))


if __name__ == "__main__":
    main()
