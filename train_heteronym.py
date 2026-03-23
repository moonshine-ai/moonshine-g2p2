#!/usr/bin/env python3
"""
Train the small heteronym encoder–decoder on LibriG2P homograph JSON (phoneme
targets, OOV-style decoder). Validation uses greedy phoneme decoding plus Levenshtein
matching among candidates (same post-processing as inference).

Example::

    python train_heteronym.py --out ./heteronym_runs/run1 --epochs 3
    python train_heteronym.py --out ./heteronym_runs/run1 --epochs 10 --resume
    python train_heteronym.py --out ./heteronym_runs/run1 --wandb --wandb-project myproj

Each epoch writes ``checkpoint.pt`` under ``--out`` (model, optimizer, progress).
With ``--resume``, training continues from ``checkpoint.pt`` under ``--out``;
``--epochs`` is the total number of epochs to run (including those already finished).

Unless ``--valid-json`` is set, the train corpus is split into train/validation
by ``--valid-fraction`` (per ``(homograph, homograph_wordid)`` group, seeded with
``--seed``) so labels in dev always exist in the training index.

Data files are fetched from the Hugging Face dataset repo via huggingface_hub
(see flexthink/librig2p-nostress-space ``dataset/homograph_*.json``) when
``--train-json`` is omitted.

Training defaults: random context windows / left-padding, safe surface noise
outside the homograph span, **capping per-homograph class imbalance** so no two
alternatives differ by more than ``--max-alternative-spread`` examples (excess
rows from larger classes are dropped); candidate indices follow **training**
frequency (most common ``homograph_wordid`` → slot 0) after that cap;
inverse-frequency balancing over
``(homograph, homograph_wordid)``. Disable augmentation / balancing with
``--no-train-augment`` / ``--no-balance-train``; disable the cap with
``--max-alternative-spread 0``.

Use ``--valid-debug-json PATH`` to write misclassified validation rows after the
final epoch's validation (context, labels, greedy decode, Levenshtein); optional
``--valid-debug-max-errors``.

By default, training and validation rows are limited to the
``--top-homographs-by-frequency`` homograph surface keys (default 100) with the
highest corpus frequencies from ``--corpus-frequency-tsv`` (default
``data/en_us/dict_frequency.tsv``). Homographs in the JSON are ranked by that
file; set ``--top-homographs-by-frequency 0`` to keep all homographs.

Validation (accuracy, ``model.pt`` when it improves, and valid-debug output) runs only
on the **last** epoch of each run (including after ``--resume``).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from g2p_common import HETERONYM_CONTEXT_MAX_CHARS
from heteronym.infer import greedy_decode_phoneme_strings
from heteronym.ipa_postprocess import (
    ipa_string_to_phoneme_tokens,
    levenshtein_distance,
    pick_closest_alternative_index,
)
from heteronym.librig2p import (
    HomographRecord,
    build_char_vocab_from_homograph_records,
    build_homograph_candidate_tables,
    build_phoneme_vocab_from_ordered_ipa,
    cap_alternative_class_spread,
    download_librig2p_homograph_split,
    filter_homograph_records_by_group_keys,
    homograph_group_keys_in_records,
    iter_encoded_batches,
    load_homograph_corpus_frequency_tsv,
    load_homograph_json,
    load_training_artifacts,
    max_encoded_phoneme_len,
    save_training_artifacts,
    top_homograph_group_keys_by_corpus_frequency,
)
from heteronym.model import TinyHeteronymTransformer
from oov.model import decoder_ce_loss

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "checkpoint.pt"


# Saved in checkpoint.pt so resume can verify CLI matches the original run.
_CKPT_ARG_KEYS = (
    "max_seq_len",
    "max_candidates",
    "max_phoneme_len",
    "group_key",
    "d_model",
    "n_heads",
    "n_layers",
    "n_decoder_layers",
    "ffn_dim",
    "dropout",
    "seed",
    "device",
    "train_augment",
    "balance_training",
    "surface_noise_prob",
    "train_json",
    "valid_json",
    "valid_fraction",
    "max_alternative_spread",
    "levenshtein_extra_phonemes",
    "top_homographs_by_frequency",
    "corpus_frequency_tsv",
)


def _args_snapshot(args: argparse.Namespace) -> dict[str, object]:
    d = vars(args).copy()
    d["train_augment"] = not d.get("no_train_augment", False)
    d["balance_training"] = not d.get("no_balance_train", False)
    out: dict[str, object] = {}
    for k in _CKPT_ARG_KEYS:
        v = d.get(k)
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def _check_resume_args(saved: dict[str, object] | None, args: argparse.Namespace) -> None:
    if not isinstance(saved, dict):
        raise SystemExit("Checkpoint is missing args_snapshot (cannot --resume).")
    cur = _args_snapshot(args)
    mismatches: list[str] = []
    for k in _CKPT_ARG_KEYS:
        s, c = saved.get(k), cur.get(k)
        if s == c:
            continue
        # Checkpoints from before valid_fraction was tracked
        if k == "valid_fraction" and s is None:
            continue
        if k == "max_alternative_spread" and s is None:
            continue
        if k == "levenshtein_extra_phonemes" and s is None:
            continue
        if k == "n_decoder_layers" and s is None:
            continue
        if k == "top_homographs_by_frequency" and s is None:
            continue
        if k == "corpus_frequency_tsv" and s is None:
            continue
        mismatches.append(k)
    if mismatches:
        raise SystemExit(
            "Refusing to resume: the following flags differ from the checkpoint "
            f"({', '.join(mismatches)}). Use the same hyperparameters and data paths "
            "as the original run, or train in a fresh --out directory."
        )


def _save_checkpoint(
    out: Path,
    *,
    model: TinyHeteronymTransformer,
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
    model: TinyHeteronymTransformer,
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
    p.add_argument("--out", type=Path, default=Path("heteronym_out"), help="output directory")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=HETERONYM_CONTEXT_MAX_CHARS,
        help=f"encoder sequence length (positional embeddings); default {HETERONYM_CONTEXT_MAX_CHARS} "
        "matches the heteronym surface context window",
    )
    p.add_argument("--max-candidates", type=int, default=4)
    p.add_argument(
        "--max-phoneme-len",
        type=int,
        default=64,
        help="max phoneme tokens in decoder (BOS/EOS inclusive cap on teacher-forcing length)",
    )
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument(
        "--n-decoder-layers",
        type=int,
        default=None,
        help="decoder depth (default: same as --n-layers)",
    )
    p.add_argument(
        "--levenshtein-extra-phonemes",
        type=int,
        default=4,
        metavar="N",
        help="validation / inference: compare each candidate to at most len(candidate)+N "
        "decoded phoneme tokens (reduces impact of repetitive hallucinations)",
    )
    p.add_argument("--ffn-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="override path to homograph_train.json",
    )
    p.add_argument(
        "--valid-json",
        type=Path,
        default=None,
        help="optional separate validation JSON; if omitted, --valid-fraction is held out "
        "from the train corpus (same labeling scheme)",
    )
    p.add_argument("--group-key", choices=("lower", "exact"), default="lower")
    p.add_argument(
        "--no-train-augment",
        action="store_true",
        help="disable random context windows, left-padding, and safe surface noise",
    )
    p.add_argument(
        "--no-balance-train",
        action="store_true",
        help="disable inverse-frequency oversampling of (homograph, wordid) pairs",
    )
    p.add_argument(
        "--surface-noise-prob",
        type=float,
        default=0.3,
        help="per-example chance to apply 1–2 safe edits outside the homograph span",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=f"continue from {CHECKPOINT_NAME} under --out (full checkpoint required)",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="log metrics to Weights & Biases (requires wandb login or WANDB_API_KEY)",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="heteronym",
        help="W&B project name (used with --wandb)",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (optional)",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team or username; optional, defaults from wandb settings)",
    )
    p.add_argument(
        "--valid-fraction",
        type=float,
        default=0.1,
        metavar="F",
        help="when --valid-json is omitted: fraction of each (homograph, wordid) group "
        "held out for validation (deterministic given --seed)",
    )
    p.add_argument(
        "--valid-debug-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="after validation on the final epoch, write misclassified examples (context, labels, "
        "greedy decode, Levenshtein) to this JSON file for error analysis",
    )
    p.add_argument(
        "--valid-debug-max-errors",
        type=int,
        default=5000,
        metavar="M",
        help="cap rows written to --valid-debug-json (remaining errors counted only in summary)",
    )
    p.add_argument(
        "--max-alternative-spread",
        type=int,
        default=10,
        metavar="D",
        help="per homograph surface key, after splitting train/valid: each homograph_wordid "
        "keeps at most (min wordid count + D) training rows; extra rows from larger classes "
        "are discarded (random, seeded). 0 disables.",
    )
    p.add_argument(
        "--top-homographs-by-frequency",
        type=int,
        default=100,
        metavar="N",
        help="keep only the N homograph surface keys with highest corpus frequency "
        "(from --corpus-frequency-tsv) among keys present in train+valid data; 0 = no limit",
    )
    p.add_argument(
        "--corpus-frequency-tsv",
        type=Path,
        default=Path("data/en_us/dict_frequency.tsv"),
        help="TSV word<TAB>ipa<TAB>frequency (e.g. from scripts/build_dict_corpus_frequency.py); "
        "used when --top-homographs-by-frequency > 0",
    )
    return p.parse_args(argv)


def _group_homograph(r: HomographRecord, group_key: str) -> str:
    return r.homograph.lower() if group_key == "lower" else r.homograph


def _split_train_valid(
    records: list[HomographRecord],
    *,
    group_key: str,
    valid_fraction: float,
    seed: int,
) -> tuple[list[HomographRecord], list[HomographRecord]]:
    """
    Stratify by (homograph key, homograph_wordid): each multi-example group keeps at
    least one row in train so the dev labels still appear in the training index.
    """
    if not (0.0 < valid_fraction < 1.0):
        raise SystemExit("--valid-fraction must be strictly between 0 and 1.")
    rng = random.Random(seed)
    by_kw: dict[tuple[str, str], list[HomographRecord]] = defaultdict(list)
    for r in records:
        g = _group_homograph(r, group_key)
        by_kw[(g, r.homograph_wordid)].append(r)
    train_out: list[HomographRecord] = []
    valid_out: list[HomographRecord] = []
    for recs in by_kw.values():
        rng.shuffle(recs)
        n = len(recs)
        if n <= 1:
            train_out.extend(recs)
            continue
        n_val = max(1, int(round(n * valid_fraction)))
        n_val = min(n_val, n - 1)
        valid_out.extend(recs[:n_val])
        train_out.extend(recs[n_val:])
    rng.shuffle(train_out)
    rng.shuffle(valid_out)
    return train_out, valid_out


def _run_validation(
    model: TinyHeteronymTransformer,
    records,
    *,
    char_vocab,
    phoneme_vocab,
    ordered,
    ordered_ipa,
    label_maps,
    group_key: str,
    max_seq_len: int,
    max_candidates: int,
    max_phoneme_len: int,
    batch_size: int,
    device: torch.device,
    collect_errors: bool,
    max_errors: int,
    levenshtein_extra_phonemes: int,
    pbar_desc: str | None = None,
) -> tuple[float, int, int, list[dict[str, Any]] | None, int]:
    """
    Returns (accuracy, n_ok, n_tot, errors_or_none, n_errors_total).

    Predictions: greedy phoneme decode per row, then closest training-slot IPA by
    Levenshtein (candidate length + *levenshtein_extra_phonemes* on the prediction prefix).
    """
    model.eval()
    n_ok = 0
    n_tot = 0
    errors: list[dict[str, Any]] | None = [] if collect_errors else None
    n_errors_total = 0
    with torch.no_grad():
        batch_iter = iter_encoded_batches(
            records,
            char_vocab=char_vocab,
            phoneme_vocab=phoneme_vocab,
            ordered_candidates=ordered,
            ordered_ipa=ordered_ipa,
            label_maps=label_maps,
            group_key=group_key,
            max_seq_len=max_seq_len,
            max_candidates=max_candidates,
            max_phoneme_len=max_phoneme_len,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            include_group_keys=True,
            include_row_debug=collect_errors,
        )
        if pbar_desc is not None:
            iterator = tqdm(batch_iter, desc=pbar_desc, unit="batch", leave=True)
        else:
            iterator = batch_iter
        for batch in iterator:
            cm = batch["candidate_mask"].to(device)
            y = batch["labels"].to(device)
            valid = cm.gather(1, y.unsqueeze(1)).squeeze(1).bool()
            inp = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            sm = batch["span_mask"].to(device)
            gkeys = batch["group_keys"]
            dbg_rows = batch["row_debug"] if collect_errors else None

            for row in range(inp.shape[0]):
                if not bool(valid[row].item()):
                    continue
                yi = int(y[row].item())
                n_valid = int(cm[row].sum().item())
                pred_tokens = greedy_decode_phoneme_strings(
                    model,
                    input_ids=inp[row : row + 1],
                    attention_mask=am[row : row + 1],
                    span_mask=sm[row : row + 1],
                    phoneme_vocab=phoneme_vocab,
                    max_phoneme_len=max_phoneme_len,
                    device=device,
                )
                gk = gkeys[row]
                ipa_slots = ordered_ipa[gk]
                pi = pick_closest_alternative_index(
                    pred_tokens,
                    ipa_slots,
                    n_valid=n_valid,
                    extra_phonemes=levenshtein_extra_phonemes,
                )
                n_tot += 1
                if pi == yi:
                    n_ok += 1
                else:
                    n_errors_total += 1
                    if not collect_errors or errors is None or len(errors) >= max_errors:
                        continue
                    cands_w = ordered[gk]
                    cands_ipa = ordered_ipa[gk]
                    k_real = len(cands_w)
                    dists = []
                    for i in range(k_real):
                        cand_tok = ipa_string_to_phoneme_tokens(cands_ipa[i])
                        lim = len(cand_tok) + max(0, int(levenshtein_extra_phonemes))
                        dists.append(levenshtein_distance(cand_tok, pred_tokens[:lim]))
                    rec: dict[str, Any] = {
                        **dbg_rows[row],
                        "gold_label_index": yi,
                        "pred_label_index": pi,
                        "greedy_decoded_phoneme_tokens": pred_tokens,
                        "levenshtein_extra_phonemes": levenshtein_extra_phonemes,
                        "levenshtein_distances_to_candidates": dists,
                        "gold_homograph_wordid": cands_w[yi] if yi < k_real else None,
                        "pred_homograph_wordid": cands_w[pi] if pi < k_real else None,
                        "gold_ipa": cands_ipa[yi] if yi < k_real else None,
                        "pred_ipa": cands_ipa[pi] if pi < k_real else None,
                        "alternatives": [
                            {
                                "index": i,
                                "homograph_wordid": cands_w[i],
                                "ipa": cands_ipa[i],
                                "levenshtein_to_greedy_prefix": dists[i] if i < len(dists) else None,
                            }
                            for i in range(k_real)
                        ],
                    }
                    errors.append(rec)
            if pbar_desc is not None:
                iterator.set_postfix(acc=f"{n_ok / max(n_tot, 1):.4f}", rows=n_tot)
    acc = n_ok / max(n_tot, 1)
    if collect_errors and errors is not None:
        return acc, n_ok, n_tot, errors, n_errors_total
    return acc, n_ok, n_tot, None, 0


def _evaluate(
    model: TinyHeteronymTransformer,
    records,
    *,
    char_vocab,
    phoneme_vocab,
    ordered,
    ordered_ipa,
    label_maps,
    group_key: str,
    max_seq_len: int,
    max_candidates: int,
    max_phoneme_len: int,
    batch_size: int,
    device: torch.device,
    levenshtein_extra_phonemes: int,
    pbar_desc: str | None = None,
) -> float:
    acc, _, _, _, _ = _run_validation(
        model,
        records,
        char_vocab=char_vocab,
        phoneme_vocab=phoneme_vocab,
        ordered=ordered,
        ordered_ipa=ordered_ipa,
        label_maps=label_maps,
        group_key=group_key,
        max_seq_len=max_seq_len,
        max_candidates=max_candidates,
        max_phoneme_len=max_phoneme_len,
        batch_size=batch_size,
        device=device,
        collect_errors=False,
        max_errors=0,
        levenshtein_extra_phonemes=levenshtein_extra_phonemes,
        pbar_desc=pbar_desc,
    )
    return acc


def _write_valid_debug_json(
    path: Path,
    *,
    epoch: int,
    accuracy: float,
    n_ok: int,
    n_evaluated: int,
    errors: list[dict[str, Any]],
    n_errors_total: int,
    max_errors: int,
    args: argparse.Namespace,
) -> None:
    by_key: Counter[str] = Counter()
    gold_vs_pred: Counter[tuple[str, str]] = Counter()
    for e in errors:
        gk = str(e.get("homograph_group_key", ""))
        by_key[gk] += 1
        gw = str(e.get("gold_homograph_wordid", ""))
        pw = str(e.get("pred_homograph_wordid", ""))
        gold_vs_pred[(gw, pw)] += 1
    top_keys = by_key.most_common(40)
    top_confusions = gold_vs_pred.most_common(30)
    payload = {
        "schema": "heteronym_valid_debug_v4_greedy_decode",
        "epoch": epoch,
        "valid_accuracy": accuracy,
        "n_ok": n_ok,
        "n_evaluated": n_evaluated,
        "n_errors_total": n_errors_total,
        "errors_in_file": len(errors),
        "errors_truncated": n_errors_total > len(errors),
        "max_errors_cap": max_errors,
        "summary_errors_by_homograph_key_top40": [{"homograph_group_key": k, "count": c} for k, c in top_keys],
        "summary_gold_to_pred_wordid_top30": [
            {"gold_homograph_wordid": a, "pred_homograph_wordid": b, "count": c} for (a, b), c in top_confusions
        ],
        "summary_note": (
            "summary_errors_by_homograph_key_* and summary_gold_to_pred_* counts are computed "
            "only from error rows listed in `errors` (after --valid-debug-max-errors cap), "
            "not from the full validation set unless every error fits in the file."
        ),
        "train_config_snapshot": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "errors": errors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.n_decoder_layers is None:
        args.n_decoder_layers = args.n_layers
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_path = args.train_json
    if train_path is None:
        train_path = download_librig2p_homograph_split("train")

    corpus = load_homograph_json(train_path)
    if args.valid_json is None:
        train_recs, valid_recs = _split_train_valid(
            corpus,
            group_key=args.group_key,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
        )
        logger.info(
            "auto-split train data: %d train / %d valid (fraction=%.4f, seed=%d)",
            len(train_recs),
            len(valid_recs),
            args.valid_fraction,
            args.seed,
        )
        if not valid_recs:
            raise SystemExit(
                "Auto-split produced an empty validation set: every "
                "(homograph, homograph_wordid) group has only one example. "
                "Add more data per label, set a separate --valid-json, or lower "
                "the bar by merging duplicates (not supported here)."
            )
    else:
        train_recs = corpus
        valid_recs = load_homograph_json(args.valid_json)

    if args.top_homographs_by_frequency > 0:
        freq_tsv = args.corpus_frequency_tsv.resolve()
        if not freq_tsv.is_file():
            raise SystemExit(
                f"--corpus-frequency-tsv not found: {freq_tsv} "
                "(build it with scripts/build_dict_corpus_frequency.py, or set "
                "--top-homographs-by-frequency 0)"
            )
        freq_map = load_homograph_corpus_frequency_tsv(freq_tsv)
        key_union = homograph_group_keys_in_records(
            train_recs, args.group_key
        ) | homograph_group_keys_in_records(valid_recs, args.group_key)
        allowed = top_homograph_group_keys_by_corpus_frequency(
            key_union,
            freq_map,
            args.top_homographs_by_frequency,
        )
        n_tr0, n_va0 = len(train_recs), len(valid_recs)
        train_recs = filter_homograph_records_by_group_keys(
            train_recs, group_key=args.group_key, allowed_group_keys=allowed
        )
        valid_recs = filter_homograph_records_by_group_keys(
            valid_recs, group_key=args.group_key, allowed_group_keys=allowed
        )
        logger.info(
            "corpus-frequency filter: kept %d / %d homograph keys (top %d by %s); "
            "train rows %d -> %d, valid rows %d -> %d",
            len(allowed),
            len(key_union),
            args.top_homographs_by_frequency,
            freq_tsv,
            n_tr0,
            len(train_recs),
            n_va0,
            len(valid_recs),
        )
        if not train_recs:
            raise SystemExit(
                "Training is empty after --top-homographs-by-frequency filtering."
            )
        if not valid_recs:
            raise SystemExit(
                "Validation is empty after --top-homographs-by-frequency filtering."
            )

    if args.max_alternative_spread > 0:
        n_tr_before = len(train_recs)
        train_recs, n_disc = cap_alternative_class_spread(
            train_recs,
            group_key=args.group_key,
            max_spread=args.max_alternative_spread,
            seed=args.seed,
        )
        logger.info(
            "train cap: per-homograph class spread <= %d → %d rows (discarded %d from train)",
            args.max_alternative_spread,
            len(train_recs),
            n_disc,
        )
        if not train_recs:
            raise SystemExit(
                "Training is empty after --max-alternative-spread capping (unexpected)."
            )

    if args.resume:
        char_vocab, phoneme_vocab, ordered, ordered_ipa, label_maps, mc, gk = load_training_artifacts(out)
        if mc != args.max_candidates or gk != args.group_key:
            raise SystemExit(
                "Refusing to resume: --max-candidates / --group-key must match "
                f"homograph_index.json in {out} (got max_candidates={args.max_candidates}, "
                f"group_key={args.group_key!r}; saved max_candidates={mc}, group_key={gk!r})."
            )
    else:
        ordered, label_maps, ordered_ipa = build_homograph_candidate_tables(
            train_recs,
            max_candidates=args.max_candidates,
            group_key=args.group_key,
        )
        extra = ".,;:'\"-" if not args.no_train_augment else None
        char_vocab = build_char_vocab_from_homograph_records(train_recs, extra_chars=extra)
        phoneme_vocab = build_phoneme_vocab_from_ordered_ipa(ordered_ipa)
        with (out / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, default=str, indent=2)

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
    if not valid_recs:
        raise SystemExit(
            "Validation is empty after filtering: no rows share both `homograph` and "
            "`homograph_wordid` with the candidate index built from training data. "
            "Common cause: `--train-json` uses IPA-style word ids (e.g. from eSpeak) while "
            "`--valid-json` uses LibriG2P-style ids (`*_noun`, `*_vrb`, …) — they never "
            "match. Use a validation file with the same labeling scheme as training, split "
            "held-out examples from the train file, or point both splits at LibriG2P JSON."
        )

    need_ph = max_encoded_phoneme_len(ordered_ipa, phoneme_vocab, cap=10**9)
    if need_ph > args.max_phoneme_len:
        logger.warning(
            "longest gold phoneme sequence in the train index needs %d decoder steps; "
            "--max-phoneme-len=%d may skip or truncate some rows",
            need_ph,
            args.max_phoneme_len,
        )

    if args.max_seq_len > HETERONYM_CONTEXT_MAX_CHARS:
        logger.warning(
            "--max-seq-len=%d exceeds heteronym surface context (%d chars); "
            "only the first %d positions carry real text, the rest are pad",
            args.max_seq_len,
            HETERONYM_CONTEXT_MAX_CHARS,
            HETERONYM_CONTEXT_MAX_CHARS,
        )

    model = TinyHeteronymTransformer(
        char_vocab_size=len(char_vocab),
        phoneme_vocab_size=len(phoneme_vocab),
        max_seq_len=args.max_seq_len,
        max_phoneme_len=args.max_phoneme_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
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

    train_augment = not args.no_train_augment
    balance_training = not args.no_balance_train

    wandb_run = None
    if args.wandb:
        import wandb

        wb_config = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=wb_config,
            dir=str(out),
        )

    try:
        best_acc = _train_loop(
            args=args,
            out=out,
            device=device,
            model=model,
            optimizer=opt,
            train_recs=train_recs,
            valid_recs=valid_recs,
            char_vocab=char_vocab,
            phoneme_vocab=phoneme_vocab,
            ordered=ordered,
            ordered_ipa=ordered_ipa,
            label_maps=label_maps,
            start_epoch=start_epoch,
            best_acc=best_acc,
            train_augment=train_augment,
            balance_training=balance_training,
            log_wandb=bool(args.wandb),
        )
    finally:
        if wandb_run is not None:
            import wandb

            wandb.finish()

    logger.info("done. best valid acc %.4f -> %s", best_acc, out / "model.pt")


def _train_loop(
    *,
    args: argparse.Namespace,
    out: Path,
    device: torch.device,
    model: TinyHeteronymTransformer,
    optimizer: AdamW,
    train_recs,
    valid_recs,
    char_vocab,
    phoneme_vocab,
    ordered,
    ordered_ipa,
    label_maps,
    start_epoch: int,
    best_acc: float,
    train_augment: bool,
    balance_training: bool,
    log_wandb: bool,
) -> float:
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        batch_iter = iter_encoded_batches(
            train_recs,
            char_vocab=char_vocab,
            phoneme_vocab=phoneme_vocab,
            ordered_candidates=ordered,
            ordered_ipa=ordered_ipa,
            label_maps=label_maps,
            group_key=args.group_key,
            max_seq_len=args.max_seq_len,
            max_candidates=args.max_candidates,
            max_phoneme_len=args.max_phoneme_len,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            train_augment=train_augment,
            balance_training=balance_training,
            surface_noise_prob=args.surface_noise_prob,
        )
        pbar = tqdm(
            batch_iter,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=True,
        )
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["span_mask"].to(device),
                batch["decoder_input_ids"].to(device),
                batch["decoder_attention_mask"].to(device),
            )
            loss = decoder_ce_loss(logits, batch["decoder_labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lv = float(loss.item())
            losses.append(lv)
            pbar.set_postfix(loss=f"{lv:.4f}", avg=f"{sum(losses) / len(losses):.4f}")
        lev_n = int(args.levenshtein_extra_phonemes)
        is_last_epoch = epoch + 1 == args.epochs
        acc = best_acc
        if is_last_epoch:
            valid_pbar = f"valid epoch {epoch + 1}/{args.epochs}"
            if args.valid_debug_json is not None:
                acc, n_ok_v, n_tot_v, err_rows, n_err_tot = _run_validation(
                    model,
                    valid_recs,
                    char_vocab=char_vocab,
                    phoneme_vocab=phoneme_vocab,
                    ordered=ordered,
                    ordered_ipa=ordered_ipa,
                    label_maps=label_maps,
                    group_key=args.group_key,
                    max_seq_len=args.max_seq_len,
                    max_candidates=args.max_candidates,
                    max_phoneme_len=args.max_phoneme_len,
                    batch_size=args.batch_size,
                    device=device,
                    collect_errors=True,
                    max_errors=max(0, int(args.valid_debug_max_errors)),
                    levenshtein_extra_phonemes=lev_n,
                    pbar_desc=valid_pbar,
                )
                _write_valid_debug_json(
                    args.valid_debug_json,
                    epoch=epoch + 1,
                    accuracy=acc,
                    n_ok=n_ok_v,
                    n_evaluated=n_tot_v,
                    errors=err_rows or [],
                    n_errors_total=n_err_tot,
                    max_errors=max(0, int(args.valid_debug_max_errors)),
                    args=args,
                )
                logger.info(
                    "epoch %d | valid debug → %s (%d error rows written, %d wrong total)",
                    epoch + 1,
                    args.valid_debug_json,
                    len(err_rows or []),
                    n_err_tot,
                )
            else:
                acc = _evaluate(
                    model,
                    valid_recs,
                    char_vocab=char_vocab,
                    phoneme_vocab=phoneme_vocab,
                    ordered=ordered,
                    ordered_ipa=ordered_ipa,
                    label_maps=label_maps,
                    group_key=args.group_key,
                    max_seq_len=args.max_seq_len,
                    max_candidates=args.max_candidates,
                    max_phoneme_len=args.max_phoneme_len,
                    batch_size=args.batch_size,
                    device=device,
                    levenshtein_extra_phonemes=lev_n,
                    pbar_desc=valid_pbar,
                )
        mean_loss = sum(losses) / max(len(losses), 1)
        if is_last_epoch:
            logger.info(
                "epoch %d | train loss %.4f | valid acc %.4f",
                epoch + 1,
                mean_loss,
                acc,
            )
        else:
            logger.info("epoch %d | train loss %.4f", epoch + 1, mean_loss)
        if is_last_epoch and acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out / "model.pt")
        save_training_artifacts(
            out,
            char_vocab=char_vocab,
            phoneme_vocab=phoneme_vocab,
            ordered_candidates=ordered,
            ordered_candidate_ipa=ordered_ipa,
            label_maps=label_maps,
            max_candidates=args.max_candidates,
            group_key=args.group_key,
        )
        with (out / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, default=str, indent=2)

        _save_checkpoint(
            out,
            model=model,
            optimizer=optimizer,
            completed_epochs=epoch + 1,
            best_acc=best_acc,
            args=args,
        )
        if log_wandb:
            import wandb

            payload: dict[str, float] = {"train/loss": mean_loss}
            if is_last_epoch:
                payload["valid/accuracy"] = acc
                payload["valid/best_accuracy"] = best_acc
            wandb.log(payload, step=epoch + 1)

    return best_acc


if __name__ == "__main__":
    main(sys.argv[1:])
