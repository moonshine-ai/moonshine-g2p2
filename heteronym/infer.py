"""Load a trained heteronym checkpoint and run single-span disambiguation."""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from g2p_common import SPECIAL_PAD, heteronym_centered_context_window
from heteronym.ipa_postprocess import pick_closest_cmudict_ipa
from heteronym.librig2p import load_training_artifacts
from heteronym.model import TinyHeteronymTransformer
from oov.data import SPECIAL_PHON_BOS, SPECIAL_PHON_EOS, SPECIAL_PHON_PAD


def _normalize_ipa_compare(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def _snap_from_train_config(cfg: dict[str, Any]) -> dict[str, object]:
    return {
        "max_seq_len": int(cfg["max_seq_len"]),
        "max_candidates": int(cfg["max_candidates"]),
        "max_phoneme_len": int(cfg.get("max_phoneme_len", cfg.get("max_ipa_len", 64))),
        "group_key": str(cfg["group_key"]),
        "d_model": int(cfg["d_model"]),
        "n_heads": int(cfg["n_heads"]),
        "n_layers": int(cfg["n_layers"]),
        "n_decoder_layers": int(cfg.get("n_decoder_layers", cfg["n_layers"])),
        "ffn_dim": int(cfg["ffn_dim"]),
        "dropout": float(cfg["dropout"]),
        "levenshtein_extra_phonemes": int(cfg.get("levenshtein_extra_phonemes", 4)),
    }


def _resolve_snap_and_state_dict(path: Path, ckpt: dict[str, Any]) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """
    Training writes ``checkpoint.pt`` (full dict with ``args_snapshot``) and
    ``model.pt`` (weights only). Accept either, plus optional ``train_config.json``.
    """
    artifacts_dir = path.parent

    def load_snap_from_sidecar_files() -> dict[str, object] | None:
        tc = artifacts_dir / "train_config.json"
        if tc.is_file():
            with tc.open(encoding="utf-8") as f:
                return _snap_from_train_config(json.load(f))
        sib = artifacts_dir / "checkpoint.pt"
        if sib.is_file() and sib.resolve() != path.resolve():
            other = torch.load(sib, map_location="cpu", weights_only=False)
            snap = other.get("args_snapshot")
            if isinstance(snap, dict):
                return snap
        return None

    snap = ckpt.get("args_snapshot")
    if isinstance(snap, dict):
        if "model" not in ckpt:
            raise ValueError(f"checkpoint missing model weights: {path}")
        return snap, ckpt["model"]

    state: dict[str, torch.Tensor]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif "token_emb.weight" in ckpt:
        state = ckpt  # type: ignore[assignment]
    else:
        raise ValueError(
            f"unrecognized heteronym checkpoint layout (expected full checkpoint or "
            f"model state_dict): {path}"
        )

    snap = load_snap_from_sidecar_files()
    if snap is None:
        raise ValueError(
            f"checkpoint missing args_snapshot: {path}. Use checkpoint.pt, or keep "
            f"train_config.json (or checkpoint.pt) in the same directory as model.pt."
        )
    if isinstance(snap, dict) and "max_phoneme_len" not in snap and "max_ipa_len" in snap:
        snap = {**snap, "max_phoneme_len": int(snap["max_ipa_len"])}
    return snap, state


def match_prediction_to_cmudict_ipa(predicted: str, alts: list[str]) -> str | None:
    """
    Map model output (IPA string from training) to the canonical string in *alts*
    (same rules as ``match_dictionary_alternative`` in the eSpeak dataset script).
    """
    e0 = _normalize_ipa_compare(predicted)
    for alt in alts:
        if _normalize_ipa_compare(alt) == e0:
            return alt
    e1 = e0.replace("ː", "")
    for alt in alts:
        if _normalize_ipa_compare(alt).replace("ː", "") == e1:
            return alt
    return None


def greedy_decode_phoneme_strings(
    model: TinyHeteronymTransformer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    span_mask: torch.Tensor,
    phoneme_vocab,
    max_phoneme_len: int,
    device: torch.device,
) -> list[str]:
    """Grapheme context → greedy-decoded phoneme token strings (no BOS/EOS/PAD in output)."""
    bos = phoneme_vocab.stoi[SPECIAL_PHON_BOS]
    eos = phoneme_vocab.stoi[SPECIAL_PHON_EOS]
    pad_p = phoneme_vocab.stoi[SPECIAL_PHON_PAD]
    itos = phoneme_vocab.itos
    cur: list[int] = [bos]

    for _ in range(max_phoneme_len):
        dec = torch.tensor([cur], dtype=torch.long, device=device)
        dec_mask = torch.ones(1, len(cur), dtype=torch.bool, device=device)
        logits = model(input_ids, attention_mask, span_mask, dec, dec_mask)
        nxt = int(logits[0, -1].argmax(dim=-1).item())
        if nxt == eos or nxt == pad_p:
            break
        cur.append(nxt)
        if len(cur) >= max_phoneme_len:
            break

    out: list[str] = []
    for tid in cur[1:]:
        if tid == eos:
            break
        if 0 <= tid < len(itos):
            tok = itos[tid]
            if tok in (SPECIAL_PHON_PAD, SPECIAL_PHON_BOS, SPECIAL_PHON_EOS):
                continue
            out.append(tok)
    return out


class HeteronymDisambiguator:
    """
    Loads ``checkpoint.pt`` (full training checkpoint) and sibling
    ``char_vocab.json``, ``phoneme_vocab.json``, and ``homograph_index.json``
    from the same directory.
    """

    def __init__(self, checkpoint_path: Path | str, *, device: str | None = None) -> None:
        self._path = Path(checkpoint_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"heteronym checkpoint not found: {self._path}")
        artifacts_dir = self._path.parent
        (
            self._char_vocab,
            self._phoneme_vocab,
            self._ordered,
            self._ordered_ipa,
            self._label_maps,
            self._max_candidates,
            self._group_key,
        ) = load_training_artifacts(artifacts_dir)
        ckpt = torch.load(self._path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"heteronym checkpoint must be a dict or state_dict: {self._path}")
        snap, state_dict = _resolve_snap_and_state_dict(self._path, ckpt)

        self._max_seq_len = int(snap["max_seq_len"])
        self._max_phoneme_len = int(snap.get("max_phoneme_len", snap.get("max_ipa_len", 64)))
        self._lev_extra = int(snap.get("levenshtein_extra_phonemes", 4))
        if int(snap["max_candidates"]) != self._max_candidates:
            raise ValueError(
                "checkpoint max_candidates does not match homograph_index.json "
                f"({snap['max_candidates']} vs {self._max_candidates})"
            )
        if str(snap.get("group_key")) != self._group_key:
            raise ValueError(
                f"checkpoint group_key {snap.get('group_key')!r} != "
                f"homograph_index.json {self._group_key!r}"
            )

        n_enc = int(snap["n_layers"])
        n_dec = int(snap.get("n_decoder_layers", n_enc))

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(dev)

        self._model = TinyHeteronymTransformer(
            char_vocab_size=len(self._char_vocab),
            phoneme_vocab_size=len(self._phoneme_vocab),
            max_seq_len=self._max_seq_len,
            max_phoneme_len=self._max_phoneme_len,
            d_model=int(snap["d_model"]),
            n_heads=int(snap["n_heads"]),
            n_encoder_layers=n_enc,
            n_decoder_layers=n_dec,
            dim_feedforward=int(snap["ffn_dim"]),
            dropout=float(snap["dropout"]),
        )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    def disambiguate_ipa(
        self,
        full_text: str,
        span_s: int,
        span_e: int,
        *,
        lookup_key: str,
        cmudict_alternatives: list[str],
    ) -> str:
        """
        Greedy phoneme decode from sentence context, then Levenshtein-closest string
        among *cmudict_alternatives* (prefix length candidate + *N* from training config).
        """
        if len(cmudict_alternatives) <= 1:
            return cmudict_alternatives[0] if cmudict_alternatives else ""

        if self._group_key == "lower":
            gkey = lookup_key
        else:
            gkey = full_text[span_s:span_e]

        if gkey not in self._ordered:
            return cmudict_alternatives[0]

        win = heteronym_centered_context_window(full_text, span_s, span_e)
        if win is None:
            return cmudict_alternatives[0]
        window_text, ws, we = win

        ids = self._char_vocab.encode(window_text)
        span = [0.0] * len(ids)
        for j in range(ws, min(we, len(span))):
            span[j] = 1.0

        pad_id = self._char_vocab.stoi[SPECIAL_PAD]
        while len(ids) < self._max_seq_len:
            ids.append(pad_id)
            span.append(0.0)
        ids = ids[: self._max_seq_len]
        span = span[: self._max_seq_len]
        attn = [i != pad_id for i in ids]

        if sum(span) < 1.0:
            return cmudict_alternatives[0]

        with torch.no_grad():
            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
            attention_mask = torch.tensor([attn], dtype=torch.bool, device=self._device)
            span_mask = torch.tensor([span], dtype=torch.float32, device=self._device)
            pred_tokens = greedy_decode_phoneme_strings(
                self._model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_mask=span_mask,
                phoneme_vocab=self._phoneme_vocab,
                max_phoneme_len=self._max_phoneme_len,
                device=self._device,
            )

        raw = pick_closest_cmudict_ipa(
            pred_tokens,
            cmudict_alternatives,
            extra_phonemes=self._lev_extra,
        )
        matched = match_prediction_to_cmudict_ipa(raw, cmudict_alternatives)
        return matched if matched is not None else cmudict_alternatives[0]
