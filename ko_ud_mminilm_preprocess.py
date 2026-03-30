"""
Build HanLP-compatible ``token_input_ids`` and ``token_token_span`` for the
mMiniLMv2 ``no-space`` tokenizer (same settings as HanLP UD MTL ``tok`` task).

Uses the ``tokenizers`` library only (no PyTorch / HanLP).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from tokenizers import Tokenizer

CLS_ID = 0
SEP_ID = 2
SPACE_PIECE = "▁"


def load_mminilm_tokenizer(tokenizer_json_path: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_json_path)


def _hanlp_style_ids_and_offsets(text: str, tok: Tokenizer) -> Tuple[List[str], List[int], List[Tuple[int, int]]]:
    """Mirror HanLP ``TransformerSequenceTokenizer.tokenize_str`` (``check_space_before``)."""
    enc = tok.encode(text)
    input_tokens = list(enc.tokens)
    input_ids = list(enc.ids)
    subtoken_offsets = list(enc.offsets)
    unk_id = tok.token_to_id("[UNK]") or 3

    offset = 0
    fixed_offsets: List[Tuple[int, int]] = []
    fixed_tokens: List[str] = []
    fixed_ids: List[int] = []
    for token, tid, (b, e) in zip(input_tokens, input_ids, subtoken_offsets):
        if b > offset:
            missing_token = text[offset:b]
            if not missing_token.isspace():
                fixed_tokens.append(missing_token)
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

    if text and not enc.offsets and not text.isspace():
        __index = 1
        input_tokens.insert(__index, text)
        input_ids.insert(__index, unk_id)
        subtoken_offsets.append((0, len(text)))

    if input_tokens and input_tokens[0] == "<s>":
        subtoken_offsets = subtoken_offsets[1:-1]

    add_special_tokens = True
    non_blank_offsets = [i for i, t in enumerate(input_tokens) if t != SPACE_PIECE]
    input_tokens = [input_tokens[i] for i in non_blank_offsets]
    input_ids = [input_ids[i] for i in non_blank_offsets]
    if add_special_tokens:
        non_blank_offsets = non_blank_offsets[1:-1]
        subtoken_offsets = [subtoken_offsets[i - 1] for i in non_blank_offsets]
    else:
        subtoken_offsets = [subtoken_offsets[i] for i in non_blank_offsets]

    inner = input_tokens[1:-1] if add_special_tokens else input_tokens
    for i in range(len(inner)):
        if text[subtoken_offsets[i][0]] == " ":
            b, e = subtoken_offsets[i]
            subtoken_offsets[i] = (b + 1, e)

    return input_tokens, input_ids, subtoken_offsets


def _tok_unit_spans(num_middle_subwords: int, len_input_ids: int) -> List[List[int]]:
    """One subword index per BMES tagging unit, plus CLS (0) and SEP (last)."""
    span: List[List[int]] = [[0]]
    for j in range(num_middle_subwords):
        span.append([j + 1])
    span.append([len_input_ids - 1])
    return span


def build_tok_batch(
    sentence: str,
    tokenizer_json_path: str,
) -> Tuple[List[int], List[List[int]], List[str]]:
    """
    Returns:
        input_ids: [CLS] + one id per middle subword + [SEP]
        token_span: HanLP ``token_token_span`` (one row per tagging unit + specials)
        units: BMES tagging units (from ``subtoken_offsets_to_subtokens``)
    """
    tok = load_mminilm_tokenizer(tokenizer_json_path)
    _, input_ids, subtoken_offsets = _hanlp_style_ids_and_offsets(sentence, tok)
    if len(subtoken_offsets) + 2 != len(input_ids):
        raise ValueError(
            f"unexpected tokenization layout: {len(input_ids)=} {len(subtoken_offsets)=}"
        )
    units = [sentence[b:e] for b, e in subtoken_offsets]
    token_span = _tok_unit_spans(len(subtoken_offsets), len(input_ids))
    return input_ids, token_span, units
