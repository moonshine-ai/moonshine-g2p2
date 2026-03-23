"""Teacher-forced forward helpers (e.g. optional NLL scoring over fixed IPA strings)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from heteronym.ipa_postprocess import ipa_string_to_phoneme_tokens
from oov.data import SPECIAL_PHON_BOS, SPECIAL_PHON_EOS, SPECIAL_PHON_PAD

if TYPE_CHECKING:
    from heteronym.model import TinyHeteronymTransformer


def mean_teacher_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Scalar mean cross-entropy over positions where ``labels != -100``."""
    b, t, v = logits.shape
    return F.cross_entropy(
        logits.reshape(b * t, v),
        labels.reshape(b * t),
        ignore_index=-100,
        reduction="mean",
    )


@torch.no_grad()
def teacher_forward_logits_labels_for_ipa(
    model: TinyHeteronymTransformer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    span_mask: torch.Tensor,
    ipa: str,
    phoneme_vocab,
    max_phoneme_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    One teacher-forced forward for a single IPA string; returns logits and label
    tensors shaped ``[1, T, V]`` and ``[1, T]``, or ``None`` if *ipa* is empty or
    too long for ``max_phoneme_len``.
    """
    bos = phoneme_vocab.stoi[SPECIAL_PHON_BOS]
    eos = phoneme_vocab.stoi[SPECIAL_PHON_EOS]
    pad_p = phoneme_vocab.stoi[SPECIAL_PHON_PAD]
    enc_ph = phoneme_vocab.encode_sequence(tuple(ipa_string_to_phoneme_tokens(ipa)))
    if not enc_ph or len(enc_ph) + 2 > max_phoneme_len:
        return None
    dec_in = [bos] + enc_ph
    dec_tgt = enc_ph + [eos]
    tlen = len(dec_in)
    dec_row = dec_in + [pad_p] * (max_phoneme_len - tlen)
    labels = dec_tgt + [-100] * (max_phoneme_len - tlen)
    dec_mask = [True] * tlen + [False] * (max_phoneme_len - tlen)

    dec = torch.tensor([dec_row], dtype=torch.long, device=device)
    dec_m = torch.tensor([dec_mask], dtype=torch.bool, device=device)
    lab = torch.tensor([labels], dtype=torch.long, device=device)

    logits = model(input_ids, attention_mask, span_mask, dec, dec_m)
    return logits, lab


@torch.no_grad()
def pick_lowest_nll_cmudict_index(
    model: TinyHeteronymTransformer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    span_mask: torch.Tensor,
    cmudict_alternatives: list[str],
    phoneme_vocab,
    max_phoneme_len: int,
    device: torch.device,
) -> int:
    """
    Score each CMUdict IPA string with mean teacher-forced NLL (one full forward
    per alternative), return the index of the lowest loss.
    """
    if not cmudict_alternatives:
        return 0
    if len(cmudict_alternatives) == 1:
        return 0

    losses: list[float] = []
    for ipa in cmudict_alternatives:
        out = teacher_forward_logits_labels_for_ipa(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_mask=span_mask,
            ipa=ipa,
            phoneme_vocab=phoneme_vocab,
            max_phoneme_len=max_phoneme_len,
            device=device,
        )
        if out is None:
            losses.append(math.inf)
            continue
        logits, lab = out
        losses.append(float(mean_teacher_nll(logits, lab).item()))

    best_i = min(range(len(losses)), key=lambda i: losses[i])
    return best_i
