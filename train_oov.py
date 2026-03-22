#!/usr/bin/env python3
"""
Train the small OOV G2P Transformer (encoder–decoder: grapheme string → phoneme string).

Expects JSON from ``scripts/build_oov_espeak_dataset.py`` under
``data/en_us/oov-training`` by default.

Example::

    python scripts/build_oov_espeak_dataset.py --out-dir data/en_us/oov-training
    python train_oov.py --out ./oov_runs/run1 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from oov.data import (
    PhonemeVocab,
    build_char_vocab_from_records,
    iter_encoded_batches,
    load_oov_json,
    load_training_artifacts,
    save_training_artifacts,
)
from oov.model import TinyOovG2pTransformer, decoder_ce_loss

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint.pt"

_CKPT_ARG_KEYS = (
    "max_seq_len",
    "max_phoneme_len",
    "d_model",
    "n_heads",
    "n_encoder_layers",
    "n_decoder_layers",
    "ffn_dim",
    "dropout",
    "seed",
    "device",
    "train_json",
    "valid_json",
)


def _args_snapshot(args: argparse.Namespace) -> dict[str, object]:
    d = vars(args).copy()
    out: dict[str, object] = {}
    for k in _CKPT_ARG_KEYS:
        v = d.get(k)
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def _check_resume_args(saved: dict[str, object] | None, args: argparse.Namespace) -> None:
    if not isinstance(saved, dict):
        raise SystemExit("Checkpoint is missing args_snapshot (cannot --resume).")
    cur = _args_snapshot(args)
    mismatches = [k for k in _CKPT_ARG_KEYS if saved.get(k) != cur.get(k)]
    if mismatches:
        raise SystemExit(
            "Refusing to resume: the following flags differ from the checkpoint "
            f"({', '.join(mismatches)}). Use the same hyperparameters and data paths "
            "as the original run, or train in a fresh --out directory."
        )


def _save_checkpoint(
    out: Path,
    *,
    model: TinyOovG2pTransformer,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    best_acc: float,
    args: argparse.Namespace,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "completed_epochs": completed_epochs,
        "best_acc": best_acc,
        "args_snapshot": _args_snapshot(args),
    }
    torch.save(payload, out / CHECKPOINT_NAME)


def _try_load_resume(
    out: Path,
    *,
    model: TinyOovG2pTransformer,
    optimizer: AdamW,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[int, float]:
    ckpt_path = out / CHECKPOINT_NAME
    if not ckpt_path.is_file():
        raise SystemExit(f"Cannot --resume: missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    _check_resume_args(ckpt.get("args_snapshot"), args)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["completed_epochs"]), float(ckpt["best_acc"])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("oov_out"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=64,
        help="max grapheme characters per word (encoder)",
    )
    p.add_argument(
        "--max-phoneme-len",
        type=int,
        default=64,
        help="max decoder steps (BOS + phones + EOS; longer sequences skipped)",
    )
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-encoder-layers", type=int, default=4)
    p.add_argument("--n-decoder-layers", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--train-json",
        type=Path,
        default=Path("data/en_us/oov-training/oov_train.json"),
    )
    p.add_argument(
        "--valid-json",
        type=Path,
        default=Path("data/en_us/oov-training/oov_valid.json"),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="oov-g2p")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    return p.parse_args(argv)


@torch.no_grad()
def _teacher_forcing_token_accuracy(
    model: TinyOovG2pTransformer,
    records,
    *,
    char_vocab,
    phoneme_vocab: PhonemeVocab,
    max_seq_len: int,
    max_phoneme_len: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    n_match = 0
    n_tot = 0
    for batch in iter_encoded_batches(
        records,
        char_vocab=char_vocab,
        phoneme_vocab=phoneme_vocab,
        max_seq_len=max_seq_len,
        max_phoneme_len=max_phoneme_len,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    ):
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["decoder_input_ids"].to(device),
            batch["decoder_attention_mask"].to(device),
        )
        pred = logits.argmax(dim=-1)
        labels = batch["decoder_labels"].to(device)
        valid = labels != -100
        n_match += int(((pred == labels) & valid).sum().item())
        n_tot += int(valid.sum().item())
    model.train()
    return n_match / max(n_tot, 1)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_path = args.train_json.resolve()
    valid_path = args.valid_json.resolve()
    if not train_path.is_file():
        raise SystemExit(f"Missing training JSON: {train_path}")
    train_recs = load_oov_json(train_path)
    valid_recs = load_oov_json(valid_path) if valid_path.is_file() else []

    if args.resume:
        char_vocab, phon_stoi, mpl_saved = load_training_artifacts(out)
        if mpl_saved != args.max_phoneme_len:
            raise SystemExit(
                "Refusing to resume: saved max_phoneme_len="
                f"{mpl_saved} vs --max-phoneme-len {args.max_phoneme_len}"
            )
        phoneme_vocab = PhonemeVocab.from_stoi(phon_stoi)
    else:
        char_vocab = build_char_vocab_from_records(train_recs)
        phoneme_vocab = PhonemeVocab.from_records(train_recs)
        with (out / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, default=str, indent=2)

    model = TinyOovG2pTransformer(
        char_vocab_size=len(char_vocab),
        phoneme_vocab_size=len(phoneme_vocab),
        max_seq_len=args.max_seq_len,
        max_phoneme_len=args.max_phoneme_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        start_epoch, best_acc = _try_load_resume(
            out, model=model, optimizer=opt, device=device, args=args
        )
    else:
        start_epoch, best_acc = 0, -1.0

    if start_epoch >= args.epochs:
        logger.info(
            "Nothing to do: checkpoint already finished %d epochs (--epochs %d).",
            start_epoch,
            args.epochs,
        )
        return

    torch.manual_seed(args.seed)

    wandb_run = None
    if args.wandb:
        import wandb

        wb_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=wb_config,
            dir=str(out),
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            losses: list[float] = []
            epoch_bar = tqdm(
                total=len(train_recs),
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                unit="ex",
                dynamic_ncols=True,
                leave=False,
            )
            try:
                for batch in iter_encoded_batches(
                    train_recs,
                    char_vocab=char_vocab,
                    phoneme_vocab=phoneme_vocab,
                    max_seq_len=args.max_seq_len,
                    max_phoneme_len=args.max_phoneme_len,
                    batch_size=args.batch_size,
                    shuffle=True,
                    seed=args.seed + epoch,
                    on_record=epoch_bar.update,
                ):
                    opt.zero_grad(set_to_none=True)
                    logits = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["decoder_input_ids"].to(device),
                        batch["decoder_attention_mask"].to(device),
                    )
                    loss = decoder_ce_loss(
                        logits,
                        batch["decoder_labels"].to(device),
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    lv = float(loss.item())
                    losses.append(lv)
                    epoch_bar.set_postfix(loss=f"{lv:.4f}", refresh=False)
            finally:
                epoch_bar.close()

            mean_loss = sum(losses) / max(len(losses), 1)
            acc = 0.0
            if valid_recs:
                acc = _teacher_forcing_token_accuracy(
                    model,
                    valid_recs,
                    char_vocab=char_vocab,
                    phoneme_vocab=phoneme_vocab,
                    max_seq_len=args.max_seq_len,
                    max_phoneme_len=args.max_phoneme_len,
                    batch_size=args.batch_size,
                    device=device,
                )
            logger.info(
                "epoch %d | train loss %.4f | valid phoneme tok acc %.4f",
                epoch + 1,
                mean_loss,
                acc,
            )
            best_acc = max(best_acc, acc)
            torch.save(model.state_dict(), out / "model.pt")
            save_training_artifacts(
                out,
                char_vocab=char_vocab,
                phoneme_vocab_stoi=phoneme_vocab.stoi,
                max_phoneme_len=args.max_phoneme_len,
            )
            with (out / "train_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args), f, default=str, indent=2)
            _save_checkpoint(
                out,
                model=model,
                optimizer=opt,
                completed_epochs=epoch + 1,
                best_acc=float(best_acc),
                args=args,
            )
            if args.wandb:
                import wandb

                wandb.log(
                    {
                        "train/loss": mean_loss,
                        "valid/phoneme_token_acc": acc,
                    },
                    step=epoch + 1,
                )
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()

    logger.info("done. best valid phoneme tok acc %.4f -> %s", best_acc, out / "model.pt")


if __name__ == "__main__":
    main(sys.argv[1:])
