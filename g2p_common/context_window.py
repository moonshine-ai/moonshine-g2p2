"""Deterministic context cropping / padding for encoder text windows."""

from __future__ import annotations


def inference_context_window(
    text: str,
    span_s: int,
    span_e: int,
    max_seq_len: int,
    *,
    max_left_pad: int = 48,
) -> tuple[str, int, int] | None:
    """
    Crop or left-pad *text* so length is at most *max_seq_len* while keeping
    ``[span_s, span_e)`` inside the window.

    If *text* is longer than ``max_seq_len``, crops a window that still contains
    the span, choosing the midpoint of the valid ``w0`` range so the span
    stays centered when possible. If shorter, left-pads with spaces (same cap
    as training augmentation for heteronym).
    """
    L = len(text)
    s, e = span_s, span_e
    if e > L or s < 0 or s >= e:
        return None
    if L > max_seq_len:
        lo = max(0, e - max_seq_len)
        hi = min(s, L - max_seq_len)
        if lo > hi:
            return None
        w0 = (lo + hi) // 2
        text = text[w0 : w0 + max_seq_len]
        s -= w0
        e -= w0
        L = len(text)
    if L < max_seq_len:
        budget = max_seq_len - L
        left = min(budget, max_left_pad) if budget > 0 else 0
        if left:
            text = " " * left + text
            s += left
            e += left
    return text, s, e
