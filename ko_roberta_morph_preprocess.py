"""
Word grouping and UPOS label parsing for ``KoichiYasuoka/roberta-base-korean-morph-upos``.

**Runtime uses only the stdlib** plus :mod:`ko_roberta_wordpiece` (pure Python WordPiece from
``vocab.txt`` / ``tokenizer_config.json`` in the model directory). No ``tokenizers`` or
``transformers`` packages.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import List, Sequence, Tuple

from ko_roberta_wordpiece import encode_bert_wordpiece

# Universal Dependencies coarse UPOS (17 tags).
_UD_UPOS = frozenset(
    {
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    }
)


def morph_label_to_upos(label: str) -> str:
    """
    Map an esupar-style morph label (e.g. ``B-NOUN+AUX+PART``, ``PUNCT``) to a single UD UPOS.
    Uses the first ``+``-separated segment that is a valid UPOS tag.
    """
    s = label.strip()
    if s.startswith(("B-", "I-")):
        s = s[2:]
    for part in s.split("+"):
        if part in _UD_UPOS:
            return part
    return "X"


def _is_punct_char(c: str) -> bool:
    return bool(c) and unicodedata.category(c).startswith("P")


def token_word_group_indices(
    tokens: Sequence[str],
    offsets: Sequence[Tuple[int, int]],
    ref_text: str,
    *,
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
) -> List[List[int]]:
    """
    Group **non-special** token indices into whitespace- and punctuation-separated surface words.

    ``tokens`` / ``offsets`` are parallel (length ``T``), including CLS/SEP.
    ``ref_text`` is the same string offsets index into (see :func:`encode_bert_wordpiece`).
    """
    idxs: List[Tuple[int, int, int]] = []
    for i, (tok, (s, e)) in enumerate(zip(tokens, offsets)):
        if tok in (cls_token, sep_token) or (s == 0 and e == 0):
            continue
        idxs.append((i, s, e))

    groups: List[List[int]] = []
    cur: List[Tuple[int, int, int]] = []
    for i, s, e in idxs:
        if not cur:
            cur.append((i, s, e))
            continue
        _pi, ps, pe = cur[-1]
        gap = ref_text[pe:s]
        new_word = gap != "" and gap.strip() == ""
        last_ch = ref_text[pe - 1] if pe > 0 else ""
        first_ch = ref_text[s] if s < len(ref_text) else ""
        punct_break = (not _is_punct_char(last_ch)) and _is_punct_char(first_ch)
        if new_word or punct_break:
            groups.append([t[0] for t in cur])
            cur = [(i, s, e)]
        else:
            cur.append((i, s, e))
    if cur:
        groups.append([t[0] for t in cur])
    return groups


def encode_for_morph_upos(
    text: str,
    model_dir: str | Path,
) -> Tuple[List[int], List[str], List[Tuple[int, int]], str, List[List[int]]]:
    """
    Returns:
        input_ids, tokens, offsets (into ``ref_text``), ``ref_text``, word_groups (token indices).
    """
    model_dir = Path(model_dir)
    input_ids, tokens, offsets, ref_text = encode_bert_wordpiece(text, model_dir)
    groups = token_word_group_indices(tokens, offsets, ref_text, cls_token="[CLS]", sep_token="[SEP]")
    return input_ids, tokens, offsets, ref_text, groups


def load_meta(model_dir: Path) -> dict:
    return json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
