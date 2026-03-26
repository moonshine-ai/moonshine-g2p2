"""Load a trained OOV G2P checkpoint and greedy-decode phonemes for a word string.

By default also prints eSpeak NG IPA on the following line (``--no-espeak`` to disable).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Allow `python oov/infer.py` (sys.path[0] is `oov/`, not the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch

from g2p_common import SPECIAL_PAD

from oov.data import SPECIAL_PHON_BOS, SPECIAL_PHON_EOS, SPECIAL_PHON_PAD, PhonemeVocab, load_training_artifacts
from oov.model import TinyOovG2pTransformer

_DEFAULT_ESPEAK_VOICE = "en-us"


def _snap_from_train_config(cfg: dict[str, Any]) -> dict[str, object]:
    """Hyperparameters as saved by ``train_oov.py`` (``train_config.json``)."""
    return {
        "max_seq_len": int(cfg["max_seq_len"]),
        "d_model": int(cfg["d_model"]),
        "n_heads": int(cfg["n_heads"]),
        "n_encoder_layers": int(cfg["n_encoder_layers"]),
        "n_decoder_layers": int(cfg["n_decoder_layers"]),
        "ffn_dim": int(cfg["ffn_dim"]),
        "dropout": float(cfg["dropout"]),
    }


def _resolve_oov_snap_and_state(
    path: Path, ckpt: Any
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """
    Training writes ``checkpoint.pt`` (dict with ``args_snapshot`` + ``model``) and
    ``model.pt`` (state_dict only). Accept either, using ``train_config.json`` or
    sibling ``checkpoint.pt`` for hyperparameters when needed.
    """
    artifacts_dir = path.parent

    def load_snap_from_sidecars() -> dict[str, object] | None:
        tc = artifacts_dir / "train_config.json"
        if tc.is_file():
            with tc.open(encoding="utf-8") as f:
                return _snap_from_train_config(json.load(f))
        sib = artifacts_dir / "checkpoint.pt"
        if sib.is_file() and sib.resolve() != path.resolve():
            other = torch.load(sib, map_location="cpu", weights_only=False)
            snap = other.get("args_snapshot") if isinstance(other, dict) else None
            if isinstance(snap, dict):
                return snap
        return None

    if not isinstance(ckpt, dict):
        raise ValueError(f"OOV checkpoint must be a dict or state_dict: {path}")

    snap = ckpt.get("args_snapshot")
    if isinstance(snap, dict):
        if "model" not in ckpt:
            raise ValueError(f"checkpoint missing model weights: {path}")
        return snap, ckpt["model"]

    state: dict[str, torch.Tensor]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif "char_emb.weight" in ckpt:
        state = ckpt  # type: ignore[assignment]
    else:
        raise ValueError(
            f"unrecognized OOV checkpoint layout (expected full checkpoint or "
            f"model state_dict): {path}"
        )

    snap = load_snap_from_sidecars()
    if snap is None:
        raise ValueError(
            f"checkpoint missing args_snapshot: {path}. Use checkpoint.pt, or keep "
            f"train_config.json (or checkpoint.pt) in the same directory as model.pt."
        )
    return snap, state


def _espeak_ng_ipa_for_surface(text: str, *, voice: str) -> str | None:
    """IPA from libespeak-ng; same separator policy as ``heteronym.espeak_heteronyms``."""
    try:
        from heteronym.espeak_heteronyms import EspeakPhonemizer, espeak_phonemize_ipa_raw
    except ImportError:
        return None
    t = text.strip()
    if not t:
        return None
    try:
        phon = EspeakPhonemizer(default_voice=voice)
        raw = espeak_phonemize_ipa_raw(phon, t, voice=voice)
    except (AssertionError, OSError, RuntimeError):
        return None
    return raw or None


class OovG2pPredictor:
    """
    Loads ``checkpoint.pt`` or ``model.pt`` and sibling ``char_vocab.json`` /
    ``phoneme_vocab.json`` / ``oov_index.json`` from the same directory (as written
    by ``train_oov.py``). For ``model.pt``, hyperparameters come from
    ``train_config.json`` or sibling ``checkpoint.pt``.
    """

    def __init__(self, checkpoint_path: Path | str, *, device: str | None = None) -> None:
        self._path = Path(checkpoint_path)
        if not self._path.is_file():
            raise FileNotFoundError(f"OOV checkpoint not found: {self._path}")
        artifacts_dir = self._path.parent
        char_vocab, phon_stoi, max_phoneme_len = load_training_artifacts(artifacts_dir)
        self._char_vocab = char_vocab
        self._phoneme_vocab = PhonemeVocab.from_stoi(phon_stoi)
        self._max_phoneme_len = max_phoneme_len

        ckpt = torch.load(self._path, map_location="cpu", weights_only=False)
        snap, state = _resolve_oov_snap_and_state(self._path, ckpt)

        self._max_seq_len = int(snap["max_seq_len"])
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(dev)

        self._model = TinyOovG2pTransformer(
            char_vocab_size=len(self._char_vocab),
            phoneme_vocab_size=len(self._phoneme_vocab),
            max_seq_len=self._max_seq_len,
            max_phoneme_len=max_phoneme_len,
            d_model=int(snap["d_model"]),
            n_heads=int(snap["n_heads"]),
            n_encoder_layers=int(snap["n_encoder_layers"]),
            n_decoder_layers=int(snap["n_decoder_layers"]),
            dim_feedforward=int(snap["ffn_dim"]),
            dropout=float(snap["dropout"]),
        )
        self._model.load_state_dict(state)
        self._model.to(self._device)
        self._model.eval()

    @torch.no_grad()
    def predict_phonemes(self, word: str) -> list[str]:
        """Grapheme string → greedy-decoded phone tokens."""
        if not word:
            return []
        ids = self._char_vocab.encode(word)
        if len(ids) > self._max_seq_len:
            ids = ids[: self._max_seq_len]

        pad_id = self._char_vocab.stoi[SPECIAL_PAD]
        t_max = self._max_seq_len
        pad = t_max - len(ids)
        enc_ids = torch.tensor([ids + [pad_id] * pad], dtype=torch.long, device=self._device)
        enc_mask = torch.tensor(
            [[1] * len(ids) + [0] * pad],
            dtype=torch.bool,
            device=self._device,
        )

        bos = self._phoneme_vocab.stoi[SPECIAL_PHON_BOS]
        eos = self._phoneme_vocab.stoi[SPECIAL_PHON_EOS]
        pad_p = self._phoneme_vocab.stoi[SPECIAL_PHON_PAD]
        cur: list[int] = [bos]
        itos = self._phoneme_vocab.itos

        for _ in range(self._max_phoneme_len):
            dec = torch.tensor([cur], dtype=torch.long, device=self._device)
            dec_mask = torch.ones(1, len(cur), dtype=torch.bool, device=self._device)
            logits = self._model(enc_ids, enc_mask, dec, dec_mask)
            nxt = int(logits[0, -1].argmax(dim=-1).item())
            if nxt == eos or nxt == pad_p:
                break
            cur.append(nxt)
            if len(cur) >= self._max_phoneme_len:
                break

        out: list[str] = []
        for tid in cur[1:]:
            if tid == eos:
                break
            if 0 <= tid < len(itos):
                tok = itos[tid]
                if tok in (SPECIAL_PHON_PAD, SPECIAL_PHON_BOS):
                    continue
                out.append(tok)
        return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "checkpoint",
        type=Path,
        help="checkpoint.pt or model.pt (vocab JSON + train_config.json or checkpoint.pt in same dir)",
    )
    p.add_argument("word", help="Single word to transcribe")
    p.add_argument("--device", default=None, help="Device (cuda/cpu); default: auto")
    p.add_argument(
        "--no-espeak",
        action="store_true",
        help="do not print a second IPA line from eSpeak NG",
    )
    p.add_argument(
        "--espeak-voice",
        type=str,
        default=_DEFAULT_ESPEAK_VOICE,
        metavar="VOICE",
        help=f"eSpeak voice for the reference line (default: {_DEFAULT_ESPEAK_VOICE})",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pred = OovG2pPredictor(args.checkpoint, device=args.device)
    phones = pred.predict_phonemes(args.word)
    print("".join(phones))
    if not args.no_espeak:
        espeak_line = _espeak_ng_ipa_for_surface(args.word, voice=args.espeak_voice)
        if espeak_line is not None:
            print(f"{espeak_line} (espeak-ng)")


if __name__ == "__main__":
    main()
