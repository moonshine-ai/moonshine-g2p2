#!/usr/bin/env python3
"""
Export published en_us PyTorch checkpoints under ``models/en_us/{heteronym,oov}``
to ONNX for ONNX Runtime.

Dependencies (install in the same environment as PyTorch)::

    pip install onnx

Optional verification (compares one forward pass against PyTorch)::

    pip install onnxruntime

PyTorch 2.9+ defaults to a dynamo-based ONNX exporter that needs extra packages;
this script uses the legacy TorchScript exporter (``dynamo=False``), which only
requires ``onnx``.

Input convention (matches :meth:`TinyHeteronymTransformer.forward` /
:meth:`TinyOovG2pTransformer.forward`):

- ``encoder_input_ids``, ``decoder_input_ids``: int64, token ids
- ``encoder_attention_mask``, ``decoder_attention_mask``: int64 (or bool), **1**
  for real positions and **0** for padding (padding mask is ``== 0`` inside the
  model)
- ``span_mask`` (heteronym only): float32, same length as the encoder; additive
  span mark (0 or 1 per position)

Output:

- ``logits``: float32 ``[batch, decoder_seq_len, phoneme_vocab_size]``

At inference, **decoder_seq_len must equal** ``max_phoneme_len`` from training (pad
shorter prefixes and mask padding); only the batch axis is dynamic on the decoder
inputs. Encoder sequence length may still be dynamic up to ``max_seq_len``.

After export, a single ``onnx-config.json`` is written next to ``model.onnx``,
embedding ``char_vocab.json``, ``phoneme_vocab.json``, ``train_config.json``, the
index file (``homograph_index.json`` or ``oov_index.json``), and ONNX I/O metadata
so runtimes (e.g. ``moonshine_onnx_g2p.py``) need only that file plus the ``.onnx``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from g2p_common.char_vocab import CharVocab
from heteronym.infer import _resolve_snap_and_state_dict
from heteronym.model import TinyHeteronymTransformer
from oov.data import PhonemeVocab, load_training_artifacts
from oov.model import TinyOovG2pTransformer

CONFIG_ONNX_SCHEMA_VERSION = 1


def _require_onnx() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "The `onnx` package is required for export. Install with: pip install onnx"
        ) from e


def _load_heteronym_bundle(
    artifacts_dir: Path,
) -> tuple[TinyHeteronymTransformer, dict[str, Any]]:
    ckpt_path = artifacts_dir / "checkpoint.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing heteronym checkpoint: {ckpt_path}")
    char_vocab = CharVocab.from_stoi(
        json.loads((artifacts_dir / "char_vocab.json").read_text(encoding="utf-8"))
    )
    phon_vocab = PhonemeVocab.from_stoi(
        json.loads((artifacts_dir / "phoneme_vocab.json").read_text(encoding="utf-8"))
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    snap, state = _resolve_snap_and_state_dict(ckpt_path, ckpt)
    n_enc = int(snap["n_layers"])
    n_dec = int(snap.get("n_decoder_layers", n_enc))
    model = TinyHeteronymTransformer(
        char_vocab_size=len(char_vocab),
        phoneme_vocab_size=len(phon_vocab),
        max_seq_len=int(snap["max_seq_len"]),
        max_phoneme_len=int(snap.get("max_phoneme_len", snap.get("max_ipa_len", 64))),
        d_model=int(snap["d_model"]),
        n_heads=int(snap["n_heads"]),
        n_encoder_layers=n_enc,
        n_decoder_layers=n_dec,
        dim_feedforward=int(snap["ffn_dim"]),
        dropout=float(snap["dropout"]),
    )
    model.load_state_dict(state)
    model.eval()
    meta: dict[str, Any] = {
        "model_kind": "heteronym",
        "char_vocab_size": len(char_vocab),
        "phoneme_vocab_size": len(phon_vocab),
        "max_encoder_seq_len": int(snap["max_seq_len"]),
        "max_decoder_seq_len": int(
            snap.get("max_phoneme_len", snap.get("max_ipa_len", 64))
        ),
        "d_model": int(snap["d_model"]),
        "n_encoder_layers": n_enc,
        "n_decoder_layers": n_dec,
    }
    return model, meta


def _load_oov_bundle(
    artifacts_dir: Path,
) -> tuple[TinyOovG2pTransformer, dict[str, Any]]:
    ckpt_path = artifacts_dir / "checkpoint.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing OOV checkpoint: {ckpt_path}")
    char_vocab, phon_stoi, max_phoneme_len = load_training_artifacts(artifacts_dir)
    phon_vocab = PhonemeVocab.from_stoi(phon_stoi)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    snap = ckpt.get("args_snapshot")
    if not isinstance(snap, dict):
        raise ValueError(f"OOV checkpoint missing args_snapshot: {ckpt_path}")
    model = TinyOovG2pTransformer(
        char_vocab_size=len(char_vocab),
        phoneme_vocab_size=len(phon_vocab),
        max_seq_len=int(snap["max_seq_len"]),
        max_phoneme_len=int(max_phoneme_len),
        d_model=int(snap["d_model"]),
        n_heads=int(snap["n_heads"]),
        n_encoder_layers=int(snap["n_encoder_layers"]),
        n_decoder_layers=int(snap["n_decoder_layers"]),
        dim_feedforward=int(snap["ffn_dim"]),
        dropout=float(snap["dropout"]),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta: dict[str, Any] = {
        "model_kind": "oov",
        "char_vocab_size": len(char_vocab),
        "phoneme_vocab_size": len(phon_vocab),
        "max_encoder_seq_len": int(snap["max_seq_len"]),
        "max_decoder_seq_len": int(max_phoneme_len),
        "d_model": int(snap["d_model"]),
        "n_encoder_layers": int(snap["n_encoder_layers"]),
        "n_decoder_layers": int(snap["n_decoder_layers"]),
    }
    return model, meta


def _heteronym_example_inputs(meta: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    t_enc = int(meta["max_encoder_seq_len"])
    # Decoder must be traced at full ``max_phoneme_len``: legacy ONNX export bakes
    # MHA reshape sizes from the example length; greedy decode then pads to this.
    t_dec = int(meta["max_decoder_seq_len"])
    enc_ids = torch.zeros(1, t_enc, dtype=torch.long)
    enc_ids[0, :3] = torch.tensor([2, 3, 4], dtype=torch.long)
    enc_mask = torch.zeros(1, t_enc, dtype=torch.long)
    enc_mask[0, :3] = 1
    span = torch.zeros(1, t_enc, dtype=torch.float32)
    span[0, 1] = 1.0
    dec_ids = torch.zeros(1, t_dec, dtype=torch.long)
    dec_ids[0, 0] = 2  # BOS
    if t_dec > 1:
        dec_ids[0, 1] = 4
    if t_dec > 2:
        dec_ids[0, 2] = 5
    dec_mask = torch.zeros(1, t_dec, dtype=torch.long)
    prefix = min(3, t_dec)
    dec_mask[0, :prefix] = 1
    return (enc_ids, enc_mask, span, dec_ids, dec_mask)


def _oov_example_inputs(meta: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    t_enc = int(meta["max_encoder_seq_len"])
    t_dec = int(meta["max_decoder_seq_len"])
    enc_ids = torch.zeros(1, t_enc, dtype=torch.long)
    enc_ids[0, :3] = 1
    enc_mask = torch.zeros(1, t_enc, dtype=torch.long)
    enc_mask[0, :3] = 1
    dec_ids = torch.zeros(1, t_dec, dtype=torch.long)
    dec_ids[0, 0] = 2
    if t_dec > 1:
        dec_ids[0, 1] = 4
    if t_dec > 2:
        dec_ids[0, 2] = 5
    dec_mask = torch.zeros(1, t_dec, dtype=torch.long)
    prefix = min(3, t_dec)
    dec_mask[0, :prefix] = 1
    return (enc_ids, enc_mask, dec_ids, dec_mask)


def _export_heteronym(
    model: TinyHeteronymTransformer, onnx_path: Path, *, opset: int
) -> None:
    args = _heteronym_example_inputs(
        {
            "max_encoder_seq_len": model.max_seq_len,
            "max_decoder_seq_len": model.max_phoneme_len,
        }
    )
    torch.onnx.export(
        model,
        args,
        str(onnx_path),
        input_names=[
            "encoder_input_ids",
            "encoder_attention_mask",
            "span_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
        ],
        output_names=["logits"],
        dynamic_axes={
            "encoder_input_ids": {0: "batch", 1: "encoder_seq_len"},
            "encoder_attention_mask": {0: "batch", 1: "encoder_seq_len"},
            "span_mask": {0: "batch", 1: "encoder_seq_len"},
            # Decoder length is fixed at ``max_phoneme_len`` (see module docstring).
            "decoder_input_ids": {0: "batch"},
            "decoder_attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
        external_data=False,
    )


def _export_oov(model: TinyOovG2pTransformer, onnx_path: Path, *, opset: int) -> None:
    args = _oov_example_inputs(
        {
            "max_encoder_seq_len": model.max_seq_len,
            "max_decoder_seq_len": model.max_phoneme_len,
        }
    )
    torch.onnx.export(
        model,
        args,
        str(onnx_path),
        input_names=[
            "encoder_input_ids",
            "encoder_attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
        ],
        output_names=["logits"],
        dynamic_axes={
            "encoder_input_ids": {0: "batch", 1: "encoder_seq_len"},
            "encoder_attention_mask": {0: "batch", 1: "encoder_seq_len"},
            "decoder_input_ids": {0: "batch"},
            "decoder_attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
        external_data=False,
    )


def _metadata_payload(
    *,
    onnx_path: Path,
    model_kind: str,
    meta: dict[str, Any],
    opset: int,
) -> dict[str, Any]:
    base_inputs: list[dict[str, Any]] = [
        {
            "name": "encoder_input_ids",
            "dtype": "int64",
            "shape": ["batch", "encoder_seq_len"],
        },
        {
            "name": "encoder_attention_mask",
            "dtype": "int64",
            "shape": ["batch", "encoder_seq_len"],
            "note": "1 = token, 0 = pad (same as training)",
        },
    ]
    if model_kind == "heteronym":
        base_inputs.append(
            {
                "name": "span_mask",
                "dtype": "float32",
                "shape": ["batch", "encoder_seq_len"],
            }
        )
    dmax = int(meta["max_decoder_seq_len"])
    base_inputs.extend(
        [
            {
                "name": "decoder_input_ids",
                "dtype": "int64",
                "shape": ["batch", dmax],
                "note": f"fixed length {dmax}; pad with <pad> and mask tail",
            },
            {
                "name": "decoder_attention_mask",
                "dtype": "int64",
                "shape": ["batch", dmax],
                "note": "1 = token, 0 = pad",
            },
        ]
    )
    return {
        "onnx_path": onnx_path.name,
        "onnx_opset": opset,
        "exporter": "torch.onnx.export (TorchScript, dynamo=False)",
        "model_kind": model_kind,
        "inputs": base_inputs,
        "outputs": [
            {
                "name": "logits",
                "dtype": "float32",
                "shape": ["batch", dmax, meta["phoneme_vocab_size"]],
            }
        ],
        "limits": {
            "encoder_seq_len_max": meta["max_encoder_seq_len"],
            "decoder_seq_len_max": meta["max_decoder_seq_len"],
            "char_vocab_size": meta["char_vocab_size"],
            "phoneme_vocab_size": meta["phoneme_vocab_size"],
        },
        "architecture": {
            "d_model": meta["d_model"],
            "n_encoder_layers": meta["n_encoder_layers"],
            "n_decoder_layers": meta["n_decoder_layers"],
        },
    }


def _build_config_onnx(
    kind: str,
    directory: Path,
    meta: dict[str, Any],
    onnx_path: Path,
    opset: int,
) -> dict[str, Any]:
    """Single JSON bundle for ONNX inference (vocabs, training/index JSON, I/O metadata)."""
    char_vocab = json.loads((directory / "char_vocab.json").read_text(encoding="utf-8"))
    phoneme_vocab = json.loads(
        (directory / "phoneme_vocab.json").read_text(encoding="utf-8")
    )
    train_config = json.loads(
        (directory / "train_config.json").read_text(encoding="utf-8")
    )
    onnx_export = _metadata_payload(
        onnx_path=onnx_path,
        model_kind=kind,
        meta=meta,
        opset=opset,
    )
    out: dict[str, Any] = {
        "config_schema_version": CONFIG_ONNX_SCHEMA_VERSION,
        "model_kind": kind,
        "char_vocab": char_vocab,
        "phoneme_vocab": phoneme_vocab,
        "train_config": train_config,
        "onnx_export": onnx_export,
    }
    if kind == "heteronym":
        out["homograph_index"] = json.loads(
            (directory / "homograph_index.json").read_text(encoding="utf-8")
        )
    elif kind == "oov":
        out["oov_index"] = json.loads(
            (directory / "oov_index.json").read_text(encoding="utf-8")
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown model kind: {kind}")
    return out


def _verify_onnxruntime(
    onnx_path: Path,
    model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    input_names: list[str],
    *,
    atol: float,
) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skip --verify", file=sys.stderr)
        return

    if len(input_names) != len(args):
        raise ValueError("input_names length must match args")
    with torch.no_grad():
        expected = model(*args).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {n: t.numpy() for n, t in zip(input_names, args)}
    got = sess.run(None, feeds)[0]
    if not np.allclose(got, expected, atol=atol, rtol=0.0):
        diff = float(np.max(np.abs(got - expected)))
        raise RuntimeError(
            f"ONNX Runtime output mismatch vs PyTorch (max abs diff {diff})"
        )
    print(f"verified {onnx_path.name} against PyTorch (atol={atol})", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--language",
        type=str,
        default="en_us",
        help="Language to export models for",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for data",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models"),
        help="Root directory for models",
    )
    parser.add_argument(
        "--heteronym-dir",
        type=Path,
        default=Path("heteronym"),
        help="Directory with checkpoint.pt and vocab JSON files",
    )
    parser.add_argument(
        "--oov-dir",
        type=Path,
        default=Path("oov"),
        help="Directory with checkpoint.pt and vocab JSON files",
    )
    parser.add_argument(
        "--heteronym-output-name",
        type=Path,
        default=Path("model.onnx"),
        help="ONNX filename written inside the heteronym model directory",
    )
    parser.add_argument(
        "--oov-output-name",
        type=Path,
        default=Path("model.onnx"),
        help="ONNX filename written inside the oov model directory",
    )
    parser.add_argument(
        "--onnx-config-name",
        type=Path,
        default=Path("onnx-config.json"),
        help="Merged vocab + index + train_config + onnx_export metadata (next to model.onnx)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--only",
        choices=("config", "heteronym", "oov", "both"),
        default="config",
        help="Which bundle to export",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="If onnxruntime is installed, compare one forward pass to PyTorch",
    )
    parser.add_argument(
        "--verify-atol", type=float, default=1e-4, help="Tolerance for --verify"
    )
    parser.add_argument(
        "--heteronym-checkpoint",
        type=Path,
        default=None,
        help="Heteronym checkpoint path",
    )
    parser.add_argument(
        "--oov-checkpoint", type=Path, default=None, help="OOV checkpoint path"
    )
    args_ns = parser.parse_args(argv)

    _require_onnx()

    g2p_config_path = args_ns.model_root / args_ns.language / "g2p-config.json"
    if g2p_config_path.is_file() and args_ns.only == "config":
        g2p_config = json.loads(g2p_config_path.read_text(encoding="utf-8"))
    else:
        print(f"no g2p-config.json found at {g2p_config_path}, using cli config")
        g2p_config = {
            "uses_heteronym_model": args_ns.only in ("heteronym", "both"),
            "uses_oov_model": args_ns.only in ("oov", "both"),
        }
    if not g2p_config["uses_heteronym_model"] and not g2p_config["uses_oov_model"]:
        print("no models to export, exiting")
        return

    tasks: list[tuple[str, Path]] = []
    if g2p_config["uses_heteronym_model"]:
        if not args_ns.heteronym_checkpoint:
            print(
                "no --heteronym-checkpoint provided, but heteronym exporting is requested"
            )
            return
        tasks.append(("heteronym", args_ns.heteronym_checkpoint.resolve()))
    if g2p_config["uses_oov_model"]:
        if not args_ns.oov_checkpoint:
            print("no --oov-checkpoint provided, but oov exporting is requested")
            return
        tasks.append(("oov", args_ns.oov_checkpoint.resolve()))

    language_root = args_ns.model_root / args_ns.language
    heteronym_dir = language_root / args_ns.heteronym_dir
    oov_dir = language_root / args_ns.oov_dir
    for kind, checkpoint_path in tasks:
        checkpoint_dir = checkpoint_path.parent
        if kind == "heteronym":
            model, meta = _load_heteronym_bundle(checkpoint_dir)
            onnx_path = heteronym_dir / args_ns.heteronym_output_name
            _export_heteronym(model, onnx_path, opset=args_ns.opset)
            ex_args = _heteronym_example_inputs(meta)
        else:
            model, meta = _load_oov_bundle(checkpoint_dir)
            onnx_path = oov_dir / args_ns.oov_output_name
            _export_oov(model, onnx_path, opset=args_ns.opset)
            ex_args = _oov_example_inputs(meta)

        onnx_config_path = onnx_path.parent / args_ns.onnx_config_name
        merged = _build_config_onnx(
            kind, checkpoint_dir, meta, onnx_path, args_ns.opset
        )
        onnx_config_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {onnx_path}", file=sys.stderr)
        print(f"wrote {onnx_config_path}", file=sys.stderr)

        if args_ns.verify:
            if kind == "heteronym":
                in_names = [
                    "encoder_input_ids",
                    "encoder_attention_mask",
                    "span_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
            else:
                in_names = [
                    "encoder_input_ids",
                    "encoder_attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                ]
            _verify_onnxruntime(
                onnx_path, model, ex_args, in_names, atol=args_ns.verify_atol
            )


if __name__ == "__main__":
    main()
