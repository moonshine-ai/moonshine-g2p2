"""Deterministic context cropping / padding for encoder text windows."""

from __future__ import annotations

# Heteronym encoder sees at most this many characters of surface text (and
# should use the same value for model ``max_seq_len`` / positional embeddings).
# Span is centered when possible.
HETERONYM_CONTEXT_MAX_CHARS = 32


def heteronym_centered_context_window(
    text: str,
    span_s: int,
    span_e: int,
    *,
    max_chars: int = HETERONYM_CONTEXT_MAX_CHARS,
) -> tuple[str, int, int] | None:
    """
    Build a window of at most *max_chars* characters with ``[span_s, span_e)``
    inside it, placing the homograph span as close as possible to the center.

    - If *text* is longer than *max_chars*, crops a contiguous slice of length
      *max_chars*; *w0* is chosen in the feasible range and biased toward
      centering the span midpoint in the window.
    - If shorter, pads with spaces on the left and/or right so the final string
      has length *max_chars* and the span midpoint is centered when padding
      allows it.
    """
    L = len(text)
    s, e = span_s, span_e
    if e > L or s < 0 or s >= e or max_chars < 1:
        return None
    span_w = e - s
    if span_w > max_chars:
        return None

    if L > max_chars:
        w0_lo = max(0, e - max_chars)
        w0_hi = min(s, L - max_chars)
        if w0_lo > w0_hi:
            return None
        ideal = (s + e) / 2.0 - max_chars / 2.0
        w0 = int(round(ideal))
        w0 = max(w0_lo, min(w0, w0_hi))
        text = text[w0 : w0 + max_chars]
        s -= w0
        e -= w0
        L = len(text)

    if L < max_chars:
        total_pad = max_chars - L
        center = (s + e) / 2.0
        left_pad = int(round(max_chars / 2.0 - center))
        left_pad = max(0, min(left_pad, total_pad))
        right_pad = total_pad - left_pad
        text = " " * left_pad + text + " " * right_pad
        s += left_pad
        e += left_pad

    return text, s, e


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
