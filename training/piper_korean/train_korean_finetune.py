#!/usr/bin/env python3
"""
Piper training entry with:
  - optional --resume_from_checkpoint (fine-tune from e.g. Chinese medium 22050 Hz)
  - KoreanPhraseEpochCallback: fixed phrase WAV + TensorBoard audio each epoch

Delegates model construction to the same logic as piper_train.__main__.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from piper_train.vits.lightning import VitsModel
from piper_train.__main__ import load_state_dict

from korean_phrase_epoch_callback import KoreanPhraseEpochCallback, phoneme_ids_for_text_ko

_LOGGER = logging.getLogger("train_korean_finetune")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Preprocessed Piper dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save a .ckpt every N epochs (default: 25)",
        default=25,
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium", "high"),
        help="VITS size preset (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="Multi-speaker only (see piper_train docs)",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load weights from this Piper .ckpt (e.g. Chinese medium 22050) but start "
            "training from epoch 0 with fresh optimizers. Use this for cross-language "
            "fine-tuning; do not use Trainer --resume_from_checkpoint for that case."
        ),
    )
    # --resume_from_checkpoint: registered by Trainer.add_argparse_args (full resume)
    parser.add_argument(
        "--epoch-audio-dir",
        type=Path,
        default=None,
        help="Directory for per-epoch WAVs (default: <dataset-dir>/epoch_audio)",
    )
    parser.add_argument(
        "--fixed-korean-text",
        default="안녕하세요",
        help="Short phrase for epoch snapshots (phonemized with eSpeak ko).",
    )
    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = str(args.dataset_dir)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    num_symbols = int(config["num_symbols"])
    num_speakers = int(config["num_speakers"])
    sample_rate = int(config["audio"]["sample_rate"])

    audio_dir = args.epoch_audio_dir or (args.dataset_dir / "epoch_audio")
    phrase_ids = phoneme_ids_for_text_ko(args.fixed_korean_text)
    _LOGGER.info(
        "Fixed phrase %r -> %s phoneme ids", args.fixed_korean_text, len(phrase_ids)
    )

    phrase_cb = KoreanPhraseEpochCallback(
        phoneme_ids=phrase_ids,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        phrase_tag="korean_fixed_phrase",
    )
    ckpt_cb = ModelCheckpoint(
        every_n_epochs=args.checkpoint_epochs,
        save_last=True,
    )

    if args.base_checkpoint:
        # Avoid Lightning interpreting this as "resume training at epoch N+1"
        setattr(args, "resume_from_checkpoint", None)

    trainer = Trainer.from_argparse_args(args, callbacks=[ckpt_cb, phrase_cb])

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        assert num_speakers > 1
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info("Loaded multi-speaker partial state from single-speaker ckpt")

    ckpt_path = getattr(args, "resume_from_checkpoint", None)
    if args.base_checkpoint:
        base = Path(args.base_checkpoint).resolve()
        _LOGGER.info("Loading weights only from %s (epoch counter starts at 0)", base)
        raw = torch.load(str(base), map_location="cpu")
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            _LOGGER.warning("Missing keys when loading base ckpt: %s", incompatible.missing_keys[:8])
        if incompatible.unexpected_keys:
            _LOGGER.warning(
                "Unexpected keys when loading base ckpt: %s",
                incompatible.unexpected_keys[:8],
            )
        trainer.fit(model)
    elif ckpt_path:
        ckpt_path = str(Path(ckpt_path).resolve())
        _LOGGER.info("Resuming full training state from %s", ckpt_path)
        trainer.fit(model, ckpt_path=ckpt_path)
    else:
        trainer.fit(model)


if __name__ == "__main__":
    main()
