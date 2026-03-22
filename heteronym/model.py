"""
Compact Transformer encoder for heteronym disambiguation.

Pools hidden states over the marked grapheme span (similar in spirit to a
single-word focus with sentence context), then scores each pronunciation
alternative using pooled context and a mean-pooled embedding of that
alternative's IPA character sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyHeteronymTransformer(nn.Module):
    """
    Character-level Transformer over sentence context with additive span marking.

    Parameters
    ----------
    vocab_size
        Including special tokens (pad, unk, etc.).
    max_seq_len
        Maximum sequence length after tokenization (no CLS; span is in-char).
    ipa_vocab_size
        Separate char vocabulary for per-candidate IPA strings.
    max_ipa_len
        Max IPA characters per candidate slot (padded).
    max_candidates
        Upper bound on pronunciation alternatives per homograph (K). Model
        always outputs this many logits; use a validity mask in the loss.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        *,
        ipa_vocab_size: int,
        max_ipa_len: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        max_candidates: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.ipa_vocab_size = ipa_vocab_size
        self.max_ipa_len = max_ipa_len
        self.d_model = d_model
        self.max_candidates = max_candidates

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # Learned marking for grapheme positions inside the homograph span
        self.span_bias = nn.Parameter(torch.zeros(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.ipa_token_emb = nn.Embedding(ipa_vocab_size, d_model, padding_idx=0)
        self.ipa_pos_emb = nn.Embedding(max_ipa_len, d_model)
        self.ipa_ln = nn.LayerNorm(d_model)
        self.ctx_ln = nn.LayerNorm(d_model)
        self.ipa_ctx_score = nn.Linear(d_model, d_model, bias=False)

        self.drop = nn.Dropout(dropout)
        self.ipa_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        if self.token_emb.padding_idx is not None:
            with torch.no_grad():
                self.token_emb.weight[self.token_emb.padding_idx].zero_()
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.ipa_token_emb.weight, mean=0.0, std=0.02)
        if self.ipa_token_emb.padding_idx is not None:
            with torch.no_grad():
                self.ipa_token_emb.weight[self.ipa_token_emb.padding_idx].zero_()
        nn.init.normal_(self.ipa_pos_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_mask: torch.Tensor,
        ipa_input_ids: torch.Tensor,
        ipa_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids
            Long tensor [batch, seq_len] token ids.
        attention_mask
            Bool or 0/1 [batch, seq_len]; 1 = real token, 0 = pad.
        span_mask
            Float or bool [batch, seq_len]; non-zero = inside homograph span.
        ipa_input_ids
            Long [batch, K, ipa_len] IPA char ids per candidate.
        ipa_attention_mask
            Bool [batch, K, ipa_len]; True = real IPA char.

        Returns
        -------
        logits
            [batch, max_candidates]
        """
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"seq_len {t} > max_seq_len {self.max_seq_len}")

        positions = torch.arange(t, device=input_ids.device).clamp(max=self.max_seq_len - 1)
        pos = self.pos_emb(positions).unsqueeze(0).expand(b, -1, -1)
        x = self.token_emb(input_ids) + pos

        sm = span_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        x = x + sm * self.span_bias
        x = self.drop(x)

        pad = attention_mask == 0
        h = self.encoder(x, src_key_padding_mask=pad)

        sm_f = span_mask.to(dtype=h.dtype, device=h.device).unsqueeze(-1)
        denom = sm_f.sum(dim=1).clamp(min=1e-6)
        pooled = (h * sm_f).sum(dim=1) / denom
        pooled = self.ctx_ln(pooled)

        bk, k_slot, ipa_t = ipa_input_ids.shape
        if bk != b:
            raise ValueError(f"ipa batch {bk} != context batch {b}")
        if k_slot != self.max_candidates:
            raise ValueError(f"ipa K {k_slot} != max_candidates {self.max_candidates}")
        if ipa_t > self.max_ipa_len:
            raise ValueError(f"ipa_len {ipa_t} > max_ipa_len {self.max_ipa_len}")

        ipa_flat = ipa_input_ids.reshape(b * self.max_candidates, ipa_t)
        ipa_m = ipa_attention_mask.reshape(b * self.max_candidates, ipa_t)
        ipa_pos = torch.arange(ipa_t, device=ipa_input_ids.device).clamp(max=self.max_ipa_len - 1)
        ipa_pos_e = self.ipa_pos_emb(ipa_pos).unsqueeze(0).expand(b * self.max_candidates, -1, -1)
        ipa_h = self.ipa_token_emb(ipa_flat) + ipa_pos_e
        ipa_h = self.ipa_drop(ipa_h)
        m_f = ipa_m.to(dtype=ipa_h.dtype).unsqueeze(-1)
        ipa_denom = m_f.sum(dim=1).clamp(min=1e-6)
        ipa_pooled = (ipa_h * m_f).sum(dim=1) / ipa_denom
        ipa_pooled = ipa_pooled.view(b, self.max_candidates, self.d_model)
        ipa_pooled = self.ipa_ln(ipa_pooled)

        ctx_proj = self.ipa_ctx_score(pooled)
        logits = (ipa_pooled * ctx_proj.unsqueeze(1)).sum(dim=-1)
        return logits


def masked_candidate_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy where invalid candidate slots are masked out (NeMo-style).

    logits: [B, K]
    target: [B] in 0..K-1
    candidate_mask: [B, K] bool, True where that candidate is valid for the row.
    """
    neg_inf = torch.finfo(logits.dtype).min / 4
    masked = logits.masked_fill(~candidate_mask, neg_inf)
    return nn.functional.cross_entropy(masked, target)
