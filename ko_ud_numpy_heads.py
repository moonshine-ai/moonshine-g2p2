"""NumPy inference for tok (BMES) and UPOS (adaptive softmax), matching HanLP decoders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def bmes_to_words(chars: Sequence[str], tags: Sequence[str]) -> List[str]:
    if not chars:
        return []
    word = chars[0]
    out: List[str] = []
    for c, t in zip(chars[1:], tags[1:]):
        if t in ("B", "S"):
            out.append(word)
            word = ""
        word += c
    if word:
        out.append(word)
    return out


def pool_word_hidden(
    char_hidden: np.ndarray,
    word_char_spans: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """
    ``char_hidden`` shape ``[n_char + 2, H]`` with row 0 = CLS, -1 = SEP, middle = chars.
    ``word_char_spans`` are inclusive-exclusive indices into **character** rows (1 .. n_char).
    """
    rows: List[np.ndarray] = [char_hidden[0]]
    for a, b in word_char_spans:
        sl = char_hidden[a:b]
        rows.append(sl.mean(axis=0))
    return np.stack(rows, axis=0)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return x - np.log(np.sum(ex, axis=axis, keepdims=True))


def adaptive_log_prob(
    x: np.ndarray,
    meta: Dict[str, Any],
    weights: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Full-vocabulary log-probabilities (matches ``AdaptiveLogSoftmaxWithLoss.log_prob``).

    ``x`` shape ``[N, in_features]``.
    Returns ``[N, n_classes]``.
    """
    cutoffs: List[int] = meta["cutoffs"]
    n_classes: int = meta["n_classes"]
    shortlist: int = meta["shortlist_size"]
    n_clusters: int = meta["n_clusters"]
    head_w = weights["head_w"]
    head_b = weights["head_b"]
    ho = x @ head_w.T + head_b
    head_lp = _log_softmax(ho, axis=-1)
    out = np.empty((x.shape[0], n_classes), dtype=np.float32)
    out[:, :shortlist] = head_lp[:, :shortlist]
    for i in range(n_clusters):
        start_idx, stop_idx = cutoffs[i], cutoffs[i + 1]
        i2h_w = weights[f"tail_{i}_i2h_w"]
        i2h_b = weights[f"tail_{i}_i2h_b"]
        h2o_w = weights[f"tail_{i}_h2o_w"]
        h2o_b = weights[f"tail_{i}_h2o_b"]
        hid = x @ i2h_w.T + i2h_b
        cluster = hid @ h2o_w.T + h2o_b
        clp = _log_softmax(cluster, axis=-1)
        out[:, start_idx:stop_idx] = clp + head_lp[:, shortlist + i][:, None]
    return out


def load_heads(model_dir: Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    z = np.load(model_dir / "heads.npz", allow_pickle=False)
    weights = {k: z[k] for k in z.files}
    return meta, weights


def tok_logits(char_hidden_middle: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``char_hidden_middle`` is ``[n_char, H]`` (CLS/SEP stripped)."""
    return char_hidden_middle @ w.T + b
