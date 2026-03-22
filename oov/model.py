"""
Standard small Transformer encoder–decoder for grapheme → phoneme (word-level).

Encoder: self-attention over padded character ids. Decoder: causal self-attention
plus cross-attention to encoder memory; output projection to phoneme vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyOovG2pTransformer(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        phoneme_vocab_size: int,
        max_seq_len: int,
        max_phoneme_len: int,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.char_vocab_size = char_vocab_size
        self.phoneme_vocab_size = phoneme_vocab_size
        self.max_seq_len = max_seq_len
        self.max_phoneme_len = max_phoneme_len
        self.d_model = d_model

        self.char_emb = nn.Embedding(char_vocab_size, d_model, padding_idx=0)
        self.char_pos = nn.Embedding(max_seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)

        self.phon_emb = nn.Embedding(phoneme_vocab_size, d_model, padding_idx=0)
        self.phon_pos = nn.Embedding(max_phoneme_len, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_decoder_layers)
        self.lm_head = nn.Linear(d_model, phoneme_vocab_size)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.char_emb.weight, mean=0.0, std=0.02)
        if self.char_emb.padding_idx is not None:
            with torch.no_grad():
                self.char_emb.weight[self.char_emb.padding_idx].zero_()
        nn.init.normal_(self.char_pos.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.phon_emb.weight, mean=0.0, std=0.02)
        if self.phon_emb.padding_idx is not None:
            with torch.no_grad():
                self.phon_emb.weight[self.phon_emb.padding_idx].zero_()
        nn.init.normal_(self.phon_pos.weight, mean=0.0, std=0.02)

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns logits [batch, decoder_len, phoneme_vocab_size].
        """
        b, t_enc = encoder_input_ids.shape
        b2, t_dec = decoder_input_ids.shape
        if t_enc > self.max_seq_len:
            raise ValueError(f"encoder seq_len {t_enc} > max_seq_len {self.max_seq_len}")
        if t_dec > self.max_phoneme_len:
            raise ValueError(f"decoder seq_len {t_dec} > max_phoneme_len {self.max_phoneme_len}")

        cpos = torch.arange(t_enc, device=encoder_input_ids.device).clamp(max=self.max_seq_len - 1)
        x = self.drop(self.char_emb(encoder_input_ids) + self.char_pos(cpos).unsqueeze(0))
        enc_pad = encoder_attention_mask == 0
        memory = self.encoder(x, src_key_padding_mask=enc_pad)

        ppos = torch.arange(t_dec, device=decoder_input_ids.device).clamp(max=self.max_phoneme_len - 1)
        tgt = self.drop(self.phon_emb(decoder_input_ids) + self.phon_pos(ppos).unsqueeze(0))

        causal = nn.Transformer.generate_square_subsequent_mask(t_dec, device=decoder_input_ids.device)
        dec_pad = decoder_attention_mask == 0
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=dec_pad,
            memory_key_padding_mask=enc_pad,
        )
        return self.lm_head(out)


def decoder_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy; *labels* uses -100 on padded / ignored positions."""
    b, t, v = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(b * t, v),
        labels.reshape(b * t),
        ignore_index=-100,
    )
