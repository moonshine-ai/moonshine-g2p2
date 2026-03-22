"""Shared building blocks for heteronym and OOV grapheme-to-phoneme code."""

from g2p_common.char_vocab import SPECIAL_PAD, SPECIAL_UNK, CharVocab
from g2p_common.context_window import inference_context_window

__all__ = [
    "SPECIAL_PAD",
    "SPECIAL_UNK",
    "CharVocab",
    "inference_context_window",
]
