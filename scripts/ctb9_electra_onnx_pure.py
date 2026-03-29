"""
Port of HanLP CTB9 ELECTRA-small **preprocessing + decode** paths used by
``chinese_hanlp_ws_pos_onnx.py``, without ``hanlp`` or ``torch``.

Default: Hugging Face ``tokenizers`` loads ``tokenizer.json`` (no ``transformers``).

Optional: ``load_electra_tokenizer_vocab_txt()`` uses a from-scratch BERT
BasicTokenizer + WordPiece on ``vocab.txt`` (see ``bert_vocab_tokenizer.py``),
matching the same encoding for this export.

Also requires ``numpy``.

Character normalization table is loaded from ``char_normalize.json`` in the
model directory (written by ``export_hanlp_ctb9_tok_pos_onnx.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from tokenizers import Tokenizer


class ElectraTokenizerLike(Protocol):
    """Either ``tokenizers.Tokenizer`` or ``bert_vocab_tokenizer.BertWordPieceTokenizer``."""

    def encode(self, text: str, add_special_tokens: bool = True) -> Any: ...

    def token_to_id(self, token: str) -> int | None: ...

# Defaults aligned with HanLP CTB9 ELECTRA-small configs
_MAX_SEQ_LEN = 512
_POS_SPAN_PAD = 16


def load_electra_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    """Load the exported fast tokenizer (same file ``transformers`` would use)."""
    p = tokenizer_dir / "tokenizer.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return Tokenizer.from_file(str(p))


def load_electra_tokenizer_vocab_txt(tokenizer_dir: Path) -> Any:
    """BERT BasicTokenizer + WordPiece from ``vocab.txt`` + ``tokenizer_config.json`` (no ``tokenizers``)."""
    from bert_vocab_tokenizer import load_bert_wordpiece_tokenizer

    d = Path(tokenizer_dir)
    if not (d / "vocab.txt").is_file():
        raise FileNotFoundError(f"Missing {d / 'vocab.txt'}")
    return load_bert_wordpiece_tokenizer(d)


def load_char_normalize_json(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Re-run scripts/export_hanlp_ctb9_tok_pos_onnx.py to generate it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_chars(text: str, table: dict[str, str]) -> str:
    if not table:
        return text
    return "".join(table.get(c, c) for c in text)


def _truncate_cws_ids(input_ids: list[int], max_len: int) -> list[int]:
    if len(input_ids) <= max_len:
        return input_ids
    return input_ids[:max_len]


def cws_features(
    raw_text: str,
    tokenizer: ElectraTokenizerLike,
    char_map: dict[str, str],
    *,
    max_seq_length: int = _MAX_SEQ_LEN,
) -> dict[str, Any]:
    """Build CWS ``input_ids`` + fields needed for BMES decode (HanLP string path).

    Atomic strings follow HanLP ``generate_tags_for_subtokens``: slices use the **original**
    characters at the same indices as offsets on the normalized string (1:1 length).
    """
    text = normalize_chars(raw_text, char_map)
    cls_id = tokenizer.token_to_id("[CLS]")
    unk_id = tokenizer.token_to_id("[UNK]")
    if cls_id is None or unk_id is None:
        raise RuntimeError("tokenizer.json must define [CLS] and [UNK]")
    has_cls = True

    enc = tokenizer.encode(text, add_special_tokens=True)
    subtoken_offsets = list(enc.offsets)
    input_tokens = list(enc.tokens)
    input_ids = list(enc.ids)

    offset = 0
    fixed_offsets: list[tuple[int, int]] = []
    fixed_tokens: list[str] = []
    fixed_ids: list[int] = []
    for token, tid, (b, e) in zip(input_tokens, input_ids, subtoken_offsets):
        if b > offset:
            missing = text[offset:b]
            if not missing.isspace():
                fixed_tokens.append(missing)
                fixed_ids.append(unk_id)
                fixed_offsets.append((offset, b))
        if e == offset:
            if fixed_offsets and fixed_offsets[-1][0] < b:
                fixed_offsets[-1] = (fixed_offsets[-1][0], b)
        fixed_tokens.append(token)
        fixed_ids.append(tid)
        fixed_offsets.append((b, e))
        offset = e
    subtoken_offsets = fixed_offsets
    input_tokens = fixed_tokens
    input_ids = fixed_ids

    # Drop CLS/SEP offsets from HF ``add_special_tokens=True`` encoding.
    subtoken_offsets = subtoken_offsets[1 if has_cls else 0 : -1]

    if text and not subtoken_offsets and not text.isspace():
        idx_ins = 1 if has_cls else 0
        input_tokens.insert(idx_ins, text)
        input_ids.insert(idx_ins, unk_id)
        subtoken_offsets.append((0, len(text)))

    if not has_cls:
        input_tokens = ["[CLS]"] + input_tokens
        input_ids = [cls_id] + input_ids

    # ``encode_plus(..., add_special_tokens=True)`` already ends with SEP; do not append twice.

    atomic = [raw_text[b:e] for b, e in subtoken_offsets]
    input_ids = _truncate_cws_ids(input_ids, max_seq_length)

    if len(input_ids) < len(subtoken_offsets) + 2:
        n_keep = max(0, len(input_ids) - 2)
        subtoken_offsets = subtoken_offsets[:n_keep]
        atomic = [raw_text[b:e] for b, e in subtoken_offsets]

    return {
        "raw_text": raw_text,
        "norm_text": text,
        "atomic": atomic,
        "token_subtoken_offsets": subtoken_offsets,
        "input_ids": np.asarray([input_ids], dtype=np.int64),
    }


def _bmes_to_spans(tags: list[str]) -> list[tuple[int, int]]:
    """``hanlp.utils.span_util.bmes_to_spans``."""
    result: list[tuple[int, int]] = []
    offset = 0
    pre_offset = 0
    for t in tags[1:]:
        offset += 1
        if t in ("B", "S"):
            result.append((pre_offset, offset))
            pre_offset = offset
    if offset != len(tags):
        result.append((pre_offset, len(tags)))
    return result


def _tag_to_span_adjust(tags: list[str], subtoken_offsets: list[tuple[int, int]]) -> None:
    """HanLP ``TransformerTaggingTokenizer.tag_to_span`` subtoken merge (no group / no custom_words)."""
    offset = -1
    prev_tag: str | None = None
    for i, (tag, (b, e)) in enumerate(zip(tags, subtoken_offsets)):
        if b < offset:
            if prev_tag == "S":
                tags[i - 1] = "B"
            elif prev_tag == "E":
                tags[i - 1] = "M"
            tags[i] = tag = "M"
        offset = e
        prev_tag = tag


def cws_logits_to_words(
    logits: np.ndarray,
    tag_vocab: list[str],
    feat: dict[str, Any],
) -> list[str]:
    """``logits``: [1, n_atom, n_tag] from ``tok.onnx`` (CLS/SEP already stripped)."""
    pred = logits[0].argmax(axis=-1).astype(int).tolist()
    tags = [tag_vocab[i] for i in pred]
    atomic: list[str] = feat["atomic"]
    offs: list[tuple[int, int]] = feat["token_subtoken_offsets"]
    if len(tags) != len(atomic):
        raise RuntimeError(f"CWS decode length mismatch: {len(tags)} tags vs {len(atomic)} atomics.")
    _tag_to_span_adjust(tags, offs)
    spans = _bmes_to_spans(tags)
    return ["".join(atomic[b:e]) for b, e in spans]


def _pos_prefix_token_spans(prefix_mask: list[bool]) -> list[list[int]]:
    """HanLP ``TransformerSequenceTokenizer`` token_span from prefix_mask (word-list path)."""
    cls_is_bos = False
    sep_is_eos = False
    if prefix_mask:
        if cls_is_bos:
            prefix_mask = list(prefix_mask)
            prefix_mask[0] = True
        if sep_is_eos:
            prefix_mask = list(prefix_mask)
            prefix_mask[-1] = True
    inner = prefix_mask[1:-1]
    token_span: list[list[int]] = []
    offset = 1
    span: list[int] = []
    for mask in inner:
        if mask and span:
            token_span.append(span)
            span = []
        span.append(offset)
        offset += 1
    if span:
        token_span.append(span)
    return token_span


def pos_features(
    words: list[str],
    tokenizer: ElectraTokenizerLike,
    char_map: dict[str, str],
    *,
    max_seq_length: int = _MAX_SEQ_LEN,
    span_inner_pad: int = _POS_SPAN_PAD,
) -> dict[str, Any]:
    """Word-list path with ``ret_token_span=True`` (HanLP POS tagger)."""
    words_n = [normalize_chars(w, char_map) for w in words]
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    unk_id = tokenizer.token_to_id("[UNK]")
    if cls_id is None or sep_id is None or unk_id is None:
        raise RuntimeError("tokenizer.json must define [CLS], [SEP], and [UNK]")

    subtoken_ids_per_token: list[list[int]] = []
    for w in words_n:
        e = tokenizer.encode(w, add_special_tokens=False)
        subtoken_ids_per_token.append(list(e.ids))

    subtoken_ids_per_token = [ids if ids else [unk_id] for ids in subtoken_ids_per_token]
    input_ids: list[int] = sum(subtoken_ids_per_token, [cls_id])
    if sep_id not in input_ids:
        input_ids.append(sep_id)
    else:
        input_ids.append(sep_id)

    prefix_mask = [False] * len(input_ids)
    pos = 1
    for ids in subtoken_ids_per_token:
        prefix_mask[pos] = True
        pos += len(ids)

    input_ids = _truncate_cws_ids(input_ids, max_seq_length)
    if len(prefix_mask) > len(input_ids):
        prefix_mask = prefix_mask[: len(input_ids)]

    span_lists = _pos_prefix_token_spans(prefix_mask)
    max_inner = max((len(s) for s in span_lists), default=0)
    if max_inner > span_inner_pad:
        raise ValueError(
            f"POS subword span width {max_inner} exceeds span_inner_pad={span_inner_pad}; re-export ONNX."
        )
    w_ct = len(words_n)
    span_arr = np.zeros((1, w_ct, span_inner_pad), dtype=np.int64)
    for i, sp in enumerate(span_lists):
        if i >= w_ct:
            break
        for j, v in enumerate(sp[:span_inner_pad]):
            span_arr[0, i, j] = v

    return {
        "words_orig": words,
        "words_norm": words_n,
        "input_ids": np.asarray([input_ids], dtype=np.int64),
        "token_span": span_arr,
        "n_words": w_ct,
    }


def pos_logits_to_tags(logits: np.ndarray, tag_vocab: list[str], n_words: int) -> list[str]:
    pred = logits[0].argmax(axis=-1).astype(int).tolist()[:n_words]
    return [tag_vocab[i] for i in pred]
