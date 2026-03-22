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

from g2p_common import SPECIAL_PAD, inference_context_window
from heteronym.librig2p import load_training_artifacts
from heteronym.model import TinyHeteronymTransformer


def _normalize_ipa_compare(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def _snap_from_train_config(cfg: dict[str, Any]) -> dict[str, object]:
    return {
        "max_seq_len": int(cfg["max_seq_len"]),
        "max_candidates": int(cfg["max_candidates"]),
        "group_key": str(cfg["group_key"]),
        "d_model": int(cfg["d_model"]),
        "n_heads": int(cfg["n_heads"]),
        "n_layers": int(cfg["n_layers"]),
        "ffn_dim": int(cfg["ffn_dim"]),
        "dropout": float(cfg["dropout"]),
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


class HeteronymDisambiguator:
    """
    Loads ``checkpoint.pt`` (full training checkpoint) and sibling
    ``char_vocab.json`` / ``homograph_index.json`` from the same directory.
    """

    def __init__(self, checkpoint_path: Path | str, *, device: str | None = None) -> None:
        self._path = Path(checkpoint_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"heteronym checkpoint not found: {self._path}")
        artifacts_dir = self._path.parent
        self._char_vocab, self._ordered, self._label_maps, self._max_candidates, self._group_key = (
            load_training_artifacts(artifacts_dir)
        )
        ckpt = torch.load(self._path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(f"heteronym checkpoint must be a dict or state_dict: {self._path}")
        snap, state_dict = _resolve_snap_and_state_dict(self._path, ckpt)

        self._max_seq_len = int(snap["max_seq_len"])
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

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(dev)

        self._model = TinyHeteronymTransformer(
            vocab_size=len(self._char_vocab),
            max_seq_len=self._max_seq_len,
            d_model=int(snap["d_model"]),
            n_heads=int(snap["n_heads"]),
            n_layers=int(snap["n_layers"]),
            dim_feedforward=int(snap["ffn_dim"]),
            max_candidates=self._max_candidates,
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
        Return one IPA string from *cmudict_alternatives* using the model when the
        homograph is in the training index; otherwise the first alternative.
        """
        if len(cmudict_alternatives) <= 1:
            return cmudict_alternatives[0] if cmudict_alternatives else ""

        if self._group_key == "lower":
            gkey = lookup_key
        else:
            gkey = full_text[span_s:span_e]

        if gkey not in self._ordered:
            return cmudict_alternatives[0]

        win = inference_context_window(
            full_text, span_s, span_e, self._max_seq_len, max_left_pad=48
        )
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

        cands = self._ordered[gkey]
        cm = [True] * len(cands) + [False] * (self._max_candidates - len(cands))

        with torch.no_grad():
            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
            attention_mask = torch.tensor([attn], dtype=torch.bool, device=self._device)
            span_mask = torch.tensor([span], dtype=torch.float32, device=self._device)
            logits = self._model(input_ids, attention_mask, span_mask)
            neg_inf = torch.finfo(logits.dtype).min / 4
            cm_t = torch.tensor([cm], dtype=torch.bool, device=self._device)
            masked = logits.masked_fill(~cm_t, neg_inf)
            pred = int(masked.argmax(dim=-1).item())

        predicted = cands[pred] if 0 <= pred < len(cands) else cands[0]
        matched = match_prediction_to_cmudict_ipa(predicted, cmudict_alternatives)
        return matched if matched is not None else cmudict_alternatives[0]
