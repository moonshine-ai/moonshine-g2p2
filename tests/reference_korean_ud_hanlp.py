"""
Reference implementation using HanLP + PyTorch (tests only).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

_TOK_UD_TASKS = ("tok", "ud")


def load_korean_ud_mtl(
    *,
    model_id: Optional[str] = None,
    devices: Union[int, Sequence[int]] = -1,
) -> Any:
    import hanlp
    from hanlp.pretrained import mtl

    resolved = model_id or mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L6
    return hanlp.load(resolved, devices=devices)


def _is_flat_tok_pos(tok: Any, pos: Any) -> bool:
    return bool(tok) and isinstance(tok[0], str) and bool(pos) and isinstance(pos[0], str)


def _document_to_sentences(doc: Any) -> List[Tuple[List[str], List[str]]]:
    tok = doc["tok"]
    pos = doc["pos"]
    if _is_flat_tok_pos(tok, pos):
        return [(list(tok), list(pos))]
    out: List[Tuple[List[str], List[str]]] = []
    for t, p in zip(tok, pos):
        out.append((list(t), list(p)))
    return out


def korean_tok_upos_hanlp_reference(
    text: Union[str, Sequence[str]],
    *,
    nlp: Any = None,
    devices: Union[int, Sequence[int]] = -1,
) -> List[List[Tuple[str, str]]]:
    if nlp is None:
        nlp = load_korean_ud_mtl(devices=devices)
    if isinstance(text, str):
        doc = nlp(text, tasks=list(_TOK_UD_TASKS))
    else:
        if not text:
            return []
        doc = nlp(list(text), tasks=list(_TOK_UD_TASKS))
    out: List[List[Tuple[str, str]]] = []
    for tokens, tags in _document_to_sentences(doc):
        out.append(list(zip(tokens, tags)))
    return out
