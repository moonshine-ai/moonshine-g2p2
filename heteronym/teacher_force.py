"""Teacher-forced decoding helpers (no autoregressive greedy loops)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from heteronym.ipa_postprocess import ipa_string_to_phoneme_tokens
from oov.data import SPECIAL_PHON_BOS, SPECIAL_PHON_EOS, SPECIAL_PHON_PAD

if TYPE_CHECKING:
    from heteronym.model import TinyHeteronymTransformer


def teacher_forced_argmax_phoneme_tokens(
    logits_row: torch.Tensor,
    labels_row: torch.Tensor,
    phoneme_vocab,
) -> list[str]:
    """
    One-pass argmax phoneme strings given teacher-forced *logits_row* [T, V] and
    gold *labels_row* [T] (-100 = pad). Stops at first label pad or predicted EOS.
    """
    eos_id = phoneme_vocab.stoi[SPECIAL_PHON_EOS]
    itos = phoneme_vocab.itos
    out: list[str] = []
    T = logits_row.shape[0]
    for t in range(T):
        if int(labels_row[t].item()) == -100:
            break
        pid = int(logits_row[t].argmax(dim=-1).item())
        if pid == eos_id:
            break
        tok = itos[pid] if 0 <= pid < len(itos) else ""
        if tok in (SPECIAL_PHON_PAD, SPECIAL_PHON_BOS, SPECIAL_PHON_EOS):
            continue
        out.append(tok)
    return out


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

    bos = phoneme_vocab.stoi[SPECIAL_PHON_BOS]
    eos = phoneme_vocab.stoi[SPECIAL_PHON_EOS]
    pad_p = phoneme_vocab.stoi[SPECIAL_PHON_PAD]

    losses: list[float] = []
    for ipa in cmudict_alternatives:
        enc_ph = phoneme_vocab.encode_sequence(tuple(ipa_string_to_phoneme_tokens(ipa)))
        if not enc_ph or len(enc_ph) + 2 > max_phoneme_len:
            losses.append(math.inf)
            continue
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
        losses.append(float(mean_teacher_nll(logits, lab).item()))

    best_i = min(range(len(losses)), key=lambda i: losses[i])
    return best_i
