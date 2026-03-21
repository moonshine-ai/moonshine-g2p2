#!/usr/bin/env python3
"""
Train the small heteronym Transformer on LibriG2P homograph JSON.

Example::

    python train_heteronym.py --out ./heteronym_runs/run1 --epochs 3

Data files are fetched from the Hugging Face dataset repo via huggingface_hub
(see flexthink/librig2p-nostress-space ``dataset/homograph_*.json``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

from heteronym.librig2p import (
    CharVocab,
    build_homograph_candidate_tables,
    download_librig2p_homograph_split,
    iter_encoded_batches,
    load_homograph_json,
    save_training_artifacts,
)
from heteronym.model import TinyHeteronymTransformer, masked_candidate_loss

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("heteronym_out"), help="output directory")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-seq-len", type=int, default=384)
    p.add_argument("--max-candidates", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="override path to homograph_train.json",
    )
    p.add_argument("--valid-json", type=Path, default=None)
    p.add_argument("--group-key", choices=("lower", "exact"), default="lower")
    return p.parse_args(argv)


def _evaluate(
    model: TinyHeteronymTransformer,
    records,
    *,
    char_vocab,
    ordered,
    label_maps,
    group_key: str,
    max_seq_len: int,
    max_candidates: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    n_ok = 0
    n_tot = 0
    with torch.no_grad():
        for batch in iter_encoded_batches(
            records,
            char_vocab=char_vocab,
            ordered_candidates=ordered,
            label_maps=label_maps,
            group_key=group_key,
            max_seq_len=max_seq_len,
            max_candidates=max_candidates,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        ):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["span_mask"].to(device),
            )
            neg_inf = torch.finfo(logits.dtype).min / 4
            masked = logits.masked_fill(~batch["candidate_mask"].to(device), neg_inf)
            pred = masked.argmax(dim=-1)
            y = batch["labels"].to(device)
            cm = batch["candidate_mask"].to(device)
            valid = cm.gather(1, y.unsqueeze(1)).squeeze(1).bool()
            n_ok += int((pred == y)[valid].sum().item())
            n_tot += int(valid.sum().item())
    return n_ok / max(n_tot, 1)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_path = args.train_json
    if train_path is None:
        train_path = download_librig2p_homograph_split("train")
    valid_path = args.valid_json
    if valid_path is None:
        valid_path = download_librig2p_homograph_split("valid")

    train_recs = load_homograph_json(train_path)
    valid_recs = load_homograph_json(valid_path)
    ordered, label_maps = build_homograph_candidate_tables(
        train_recs,
        max_candidates=args.max_candidates,
        group_key=args.group_key,
    )
    char_vocab = CharVocab.from_records(train_recs)

    # Dev set: only rows whose homograph appears in train index
    def _filter_dev(recs):
        kept = []
        skipped = 0
        for r in recs:
            g = r.homograph.lower() if args.group_key == "lower" else r.homograph
            if g not in label_maps:
                skipped += 1
                continue
            if r.homograph_wordid not in label_maps[g]:
                skipped += 1
                continue
            kept.append(r)
        if skipped:
            logger.info("validation: skipped %d rows (unknown homograph/wordid)", skipped)
        return kept

    valid_recs = _filter_dev(valid_recs)

    model = TinyHeteronymTransformer(
        vocab_size=len(char_vocab),
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.ffn_dim,
        max_candidates=args.max_candidates,
        dropout=args.dropout,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in iter_encoded_batches(
            train_recs,
            char_vocab=char_vocab,
            ordered_candidates=ordered,
            label_maps=label_maps,
            group_key=args.group_key,
            max_seq_len=args.max_seq_len,
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["span_mask"].to(device),
            )
            loss = masked_candidate_loss(
                logits,
                batch["labels"].to(device),
                batch["candidate_mask"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        acc = _evaluate(
            model,
            valid_recs,
            char_vocab=char_vocab,
            ordered=ordered,
            label_maps=label_maps,
            group_key=args.group_key,
            max_seq_len=args.max_seq_len,
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
            device=device,
        )
        logger.info(
            "epoch %d | train loss %.4f | valid acc %.4f",
            epoch + 1,
            sum(losses) / max(len(losses), 1),
            acc,
        )
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out / "model.pt")
            save_training_artifacts(
                out,
                char_vocab=char_vocab,
                ordered_candidates=ordered,
                label_maps=label_maps,
                max_candidates=args.max_candidates,
                group_key=args.group_key,
            )
            with (out / "train_config.json").open("w", encoding="utf-8") as f:
                json.dump(vars(args), f, default=str, indent=2)

    logger.info("done. best valid acc %.4f -> %s", best_acc, out / "model.pt")


if __name__ == "__main__":
    main(sys.argv[1:])
