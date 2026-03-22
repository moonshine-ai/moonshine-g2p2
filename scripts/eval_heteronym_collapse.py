#!/usr/bin/env python3
"""
Measure heteronym classifier collapse on a labeled JSON corpus.

Loads a trained checkpoint (``model.pt`` or ``checkpoint.pt``) and sibling
artifacts (``char_vocab.json``, ``ipa_char_vocab.json``, ``homograph_index.json``),
then runs the same batched forward pass as training validation.

Reports:

* Global accuracy and histograms of **gold** vs **predicted** class indices.
* Whether predictions are dominated by a single index (e.g. always 0).
* Per homograph key: support, accuracy, gold/pred distributions, predicted IPA
  string distribution, and a simple **collapse score** (max mass on one pred class).

Example::

    python scripts/eval_heteronym_collapse.py \\
        --checkpoint heteronym_runs/run1/model.pt \\
        --eval-json data/en_us/heteronym-training/homograph_valid.json

Pass ``--train-json`` with the same homograph JSON used to build ``homograph_index.json``
to add **tr_dom**: the maximum marginal mass of the **gold** label on the training split
per key (prevalence of the most common alternative). Compare **tr_dom** to **p_dom** /
**ipa_dom** on eval to see whether the model is simply matching training skew.
Use ``--json-out collapse.json`` for machine-readable summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from heteronym.infer import _resolve_snap_and_state_dict
from heteronym.librig2p import iter_encoded_batches, load_homograph_json, load_training_artifacts
from heteronym.model import TinyHeteronymTransformer


def _snap_int(snap: dict[str, object], key: str, default: int) -> int:
    v = snap.get(key, default)
    return int(v) if not isinstance(v, bool) else int(default)


def _entropy(counts: Counter[int], *, n_slot: int) -> float:
    """Shannon entropy over indices 0..n_slot-1 (zero mass for missing keys)."""
    tot = sum(counts[i] for i in range(n_slot))
    if tot <= 0:
        return 0.0
    h = 0.0
    for i in range(n_slot):
        c = counts[i]
        if c <= 0:
            continue
        p = c / tot
        h -= p * math.log(p + 1e-30)
    return h


def _max_mass(counts: Counter[int], *, n_slot: int) -> tuple[int, float]:
    tot = sum(counts[i] for i in range(n_slot))
    if tot <= 0:
        return -1, 0.0
    best_i, best_c = -1, -1
    for i in range(n_slot):
        c = counts[i]
        if c > best_c:
            best_c = c
            best_i = i
    return best_i, best_c / tot


def _group_key_record(r, group_key: str) -> str:
    return r.homograph.lower() if group_key == "lower" else r.homograph


def _filter_eval_records(records, *, label_maps: dict, group_key: str):
    kept = []
    for r in records:
        g = _group_key_record(r, group_key)
        if g not in label_maps or r.homograph_wordid not in label_maps[g]:
            continue
        kept.append(r)
    return kept


def _label_histograms_for_split(
    records, *, label_maps: dict, group_key: str
) -> dict[str, Counter[int]]:
    """Count gold class indices per homograph key (same filter as eval)."""
    h: dict[str, Counter[int]] = defaultdict(Counter)
    for r in records:
        gk = _group_key_record(r, group_key)
        if gk not in label_maps or r.homograph_wordid not in label_maps[gk]:
            continue
        li = label_maps[gk][r.homograph_wordid]
        h[gk][li] += 1
    return h


def _ipa_histogram_from_label_counts(
    label_counts: Counter[int], ipa_slots: list[str]
) -> Counter[str]:
    out: Counter[str] = Counter()
    for li, c in label_counts.items():
        if 0 <= li < len(ipa_slots):
            out[ipa_slots[li]] += c
    return out


def _max_mass_counter(counter: Counter[str]) -> tuple[str, float]:
    tot = sum(counter.values())
    if tot <= 0:
        return "", 0.0
    s, c0 = counter.most_common(1)[0]
    return s, c0 / tot


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="model.pt or checkpoint.pt (artifacts live in the same directory)",
    )
    p.add_argument(
        "--eval-json",
        type=Path,
        required=True,
        help="homograph JSON with rows in the training label scheme",
    )
    p.add_argument(
        "--train-json",
        type=Path,
        default=None,
        help="training homograph JSON (same label scheme); enables tr_dom column vs training skew",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--min-support",
        type=int,
        default=20,
        metavar="N",
        help="only list per-key collapse table for keys with at least N eval rows",
    )
    p.add_argument(
        "--top",
        type=int,
        default=25,
        help="how many homograph keys to print in the collapse table",
    )
    p.add_argument("--json-out", type=Path, default=None, help="write full metrics JSON here")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ckpt_path: Path = args.checkpoint
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    if not args.eval_json.is_file():
        raise SystemExit(f"--eval-json not found: {args.eval_json}")
    if args.train_json is not None and not args.train_json.is_file():
        raise SystemExit(f"--train-json not found: {args.train_json}")

    device = torch.device(args.device)
    artifacts_dir = ckpt_path.parent
    char_vocab, ipa_cv, ordered, ordered_ipa, label_maps, max_candidates, group_key = (
        load_training_artifacts(artifacts_dir)
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit("checkpoint must be a dict (full checkpoint or state_dict)")
    snap, state_dict = _resolve_snap_and_state_dict(ckpt_path, ckpt)

    max_seq_len = _snap_int(snap, "max_seq_len", 384)
    max_ipa_len = _snap_int(snap, "max_ipa_len", 64)
    snap_mc = _snap_int(snap, "max_candidates", max_candidates)
    if snap_mc != max_candidates:
        raise SystemExit(
            f"checkpoint max_candidates {snap_mc} != homograph_index.json {max_candidates}"
        )
    sk = str(snap.get("group_key", group_key))
    if sk != group_key:
        raise SystemExit(f"checkpoint group_key {sk!r} != homograph_index.json {group_key!r}")

    model = TinyHeteronymTransformer(
        vocab_size=len(char_vocab),
        max_seq_len=max_seq_len,
        ipa_vocab_size=len(ipa_cv),
        max_ipa_len=max_ipa_len,
        d_model=_snap_int(snap, "d_model", 128),
        n_heads=_snap_int(snap, "n_heads", 4),
        n_layers=_snap_int(snap, "n_layers", 2),
        dim_feedforward=_snap_int(snap, "ffn_dim", 256),
        max_candidates=max_candidates,
        dropout=float(snap.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    records = load_homograph_json(args.eval_json)
    records = _filter_eval_records(records, label_maps=label_maps, group_key=group_key)
    if not records:
        raise SystemExit("no eval rows left after filtering to homograph_index label_maps")

    train_hist: dict[str, Counter[int]] | None = None
    train_ipa_dom: dict[str, float] = {}
    train_idx_dom: dict[str, float] = {}
    train_n_rows: dict[str, int] = {}
    if args.train_json is not None:
        train_recs = load_homograph_json(args.train_json)
        train_hist = _label_histograms_for_split(
            train_recs, label_maps=label_maps, group_key=group_key
        )
        for gk, ctr in train_hist.items():
            slots = ordered_ipa[gk]
            _, midx = _max_mass(ctr, n_slot=max_candidates)
            train_idx_dom[gk] = midx
            ipa_ctr = _ipa_histogram_from_label_counts(ctr, slots)
            _, ipa_m = _max_mass_counter(ipa_ctr)
            train_ipa_dom[gk] = ipa_m
            train_n_rows[gk] = sum(ctr.values())

    neg_inf = torch.finfo(torch.float32).min / 4

    gold_global: Counter[int] = Counter()
    pred_global: Counter[int] = Counter()
    pred_ipa_global: Counter[str] = Counter()

    per_key_gold: dict[str, Counter[int]] = defaultdict(Counter)
    per_key_pred: dict[str, Counter[int]] = defaultdict(Counter)
    per_key_pred_ipa: dict[str, Counter[str]] = defaultdict(Counter)
    per_key_correct: dict[str, int] = defaultdict(int)
    per_key_total: dict[str, int] = defaultdict(int)

    n_ok = 0
    n_tot = 0
    always0_ok = 0

    with torch.no_grad():
        for batch in iter_encoded_batches(
            records,
            char_vocab=char_vocab,
            ipa_char_vocab=ipa_cv,
            ordered_candidates=ordered,
            ordered_ipa=ordered_ipa,
            label_maps=label_maps,
            group_key=group_key,
            max_seq_len=max_seq_len,
            max_candidates=max_candidates,
            max_ipa_len=max_ipa_len,
            batch_size=args.batch_size,
            shuffle=False,
            seed=0,
            train_augment=False,
            balance_training=False,
            include_group_keys=True,
        ):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["span_mask"].to(device),
                batch["ipa_input_ids"].to(device),
                batch["ipa_attention_mask"].to(device),
            )
            cm = batch["candidate_mask"].to(device)
            masked = logits.masked_fill(~cm, neg_inf)
            pred = masked.argmax(dim=-1)
            y = batch["labels"].to(device)
            valid = cm.gather(1, y.unsqueeze(1)).squeeze(1).bool()
            gkeys = batch["group_keys"]

            for row in range(pred.shape[0]):
                if not bool(valid[row].item()):
                    continue
                gk = gkeys[row]
                yi = int(y[row].item())
                pi = int(pred[row].item())
                gold_global[yi] += 1
                pred_global[pi] += 1
                per_key_gold[gk][yi] += 1
                per_key_pred[gk][pi] += 1
                ipa_slots = ordered_ipa[gk]
                pred_ipa = ipa_slots[pi] if 0 <= pi < len(ipa_slots) else f"<oob:{pi}>"
                pred_ipa_global[pred_ipa] += 1
                per_key_pred_ipa[gk][pred_ipa] += 1

                per_key_total[gk] += 1
                if pi == yi:
                    per_key_correct[gk] += 1
                    n_ok += 1
                n_tot += 1
                if yi == 0 and pi == 0:
                    always0_ok += 1

    acc = n_ok / max(n_tot, 1)
    gold_ent = _entropy(gold_global, n_slot=max_candidates)
    pred_ent = _entropy(pred_global, n_slot=max_candidates)
    max_ent = math.log(max_candidates) if max_candidates > 1 else 1.0

    gi, pred_max_mass = _max_mass(pred_global, n_slot=max_candidates)
    g_gold, gold_max_mass = _max_mass(gold_global, n_slot=max_candidates)

    always0_denom = sum(per_key_gold[gk][0] for gk in per_key_gold)
    always0_acc = always0_ok / max(always0_denom, 1)

    pred0_rate = pred_global[0] / max(n_tot, 1)

    key_rows: list[dict[str, object]] = []
    for gk in sorted(per_key_total.keys()):
        n = per_key_total[gk]
        n_cand = len(ordered[gk])
        pk = per_key_pred[gk]
        _, collapse = _max_mass(pk, n_slot=max_candidates)
        acc_k = per_key_correct[gk] / n
        ipa_ctr = per_key_pred_ipa[gk]
        tot_i = sum(ipa_ctr.values())
        if tot_i <= 0:
            pred_ipa_dominant, ipa_collapse = "", 0.0
        else:
            s0, c0 = ipa_ctr.most_common(1)[0]
            pred_ipa_dominant = s0
            ipa_collapse = c0 / tot_i
        tr_idx = train_idx_dom.get(gk) if train_hist is not None else None
        tr_ipa = train_ipa_dom.get(gk) if train_hist is not None else None
        tr_n = train_n_rows.get(gk) if train_hist is not None else None
        key_rows.append(
            {
                "homograph_key": gk,
                "n": n,
                "n_candidates": n_cand,
                "accuracy": acc_k,
                "pred_index_collapse_mass": collapse,
                "pred_ipa_collapse_mass": ipa_collapse,
                "pred_ipa_dominant": pred_ipa_dominant,
                "train_rows_for_key": tr_n,
                "train_gold_index_dominance_mass": tr_idx,
                "train_gold_ipa_dominance_mass": tr_ipa,
                "gold_histogram": {str(i): per_key_gold[gk][i] for i in range(n_cand)},
                "pred_histogram": {str(i): per_key_pred[gk][i] for i in range(n_cand)},
            }
        )

    key_rows.sort(key=lambda r: (r["pred_index_collapse_mass"], r["n"]), reverse=True)

    print("heteronym collapse / calibration report")
    print(f"  checkpoint:   {ckpt_path}")
    print(f"  eval-json:    {args.eval_json}")
    if args.train_json is not None:
        print(f"  train-json:   {args.train_json}  (tr_dom = training majority-class mass)")
    print(f"  rows used:    {n_tot} (after label filter + span/window rules)")
    print(f"  accuracy:     {acc:.4f}")
    print()
    print("global class index histograms (counts over valid eval rows)")
    for i in range(max_candidates):
        g = gold_global[i]
        p = pred_global[i]
        if g or p:
            print(f"  slot {i}: gold={g:6d}  pred={p:6d}")
    print()
    print("global prediction dominance")
    print(f"  pred argmax mass on single index: {pred_max_mass:.4f} (index {gi})")
    print(f"  gold marginal mass on single index: {gold_max_mass:.4f} (index {g_gold})")
    print(f"  pred index == 0 rate:               {pred0_rate:.4f}")
    print(f"  normalized pred entropy / log(K):   {pred_ent / max_ent:.4f}  (gold {gold_ent / max_ent:.4f})")
    print()
    print("always-class-0 baseline (when gold label is 0)")
    print(f"  accuracy if always predict index 0: {always0_acc:.4f}  (on {always0_denom} gold-0 rows)")
    print()
    print("global predicted IPA string (training slot text) — top 15")
    for ipa, c in pred_ipa_global.most_common(15):
        print(f"  {c:6d}  {ipa!r}")
    print()

    print(
        f"ambiguous keys (>=2 pronunciation slots), n >= {args.min_support}, "
        f"by pred_index_collapse_mass (top {args.top})"
    )
    if args.train_json is not None:
        hdr = (
            f"  {'key':<16} {'n':>5} {'k':>2} {'acc':>5} {'p_dom':>6} {'ipa_dom':>7} {'tr_dom':>7}  "
            "dominant_pred_ipa"
        )
    else:
        hdr = f"  {'key':<18} {'n':>6} {'k':>3} {'acc':>6} {'p_dom':>6} {'ipa_dom':>6}  ipa"
    print(hdr)
    shown = 0
    for r in key_rows:
        if int(r["n_candidates"]) < 2:
            continue
        if int(r["n"]) < args.min_support:
            continue
        if shown >= args.top:
            break
        if args.train_json is not None:
            tr = r["train_gold_index_dominance_mass"]
            tr_s = f"{float(tr):.3f}" if isinstance(tr, (int, float)) else "  —  "
            print(
                f"  {str(r['homograph_key']):<16} {int(r['n']):>5} {int(r['n_candidates']):>2} "
                f"{float(r['accuracy']):>5.3f} {float(r['pred_index_collapse_mass']):>6.3f} "
                f"{float(r['pred_ipa_collapse_mass']):>7.3f} {tr_s:>7}  {r['pred_ipa_dominant']!r}"
            )
        else:
            print(
                f"  {str(r['homograph_key']):<18} {int(r['n']):>6} {int(r['n_candidates']):>3} "
                f"{float(r['accuracy']):>6.3f} {float(r['pred_index_collapse_mass']):>6.3f} "
                f"{float(r['pred_ipa_collapse_mass']):>6.3f}  {r['pred_ipa_dominant']!r}"
            )
        shown += 1
    if shown == 0:
        print("  (none)")
    if args.train_json is not None:
        print()
        print(
            "  tr_dom = share of training rows for this homograph whose gold label is the "
            "single most frequent class (same index ordering as homograph_index.json). "
            "Compare to p_dom / ipa_dom on eval."
        )

    ambiguous_keys = [r for r in key_rows if int(r["n_candidates"]) >= 2]
    fully_collapsed = [
        r
        for r in ambiguous_keys
        if float(r["pred_index_collapse_mass"]) >= 1.0 - 1e-9
    ]
    print()
    print(
        f"ambiguous keys (>=2 candidates): {len(ambiguous_keys)} / {len(key_rows)} total keys"
    )
    print(
        f"among ambiguous keys, 100% of preds on one class index: "
        f"{len(fully_collapsed)} / {len(ambiguous_keys)}"
    )

    if args.json_out is not None:
        out = {
            "checkpoint": str(ckpt_path),
            "eval_json": str(args.eval_json),
            "train_json": str(args.train_json) if args.train_json else None,
            "n_eval_rows": n_tot,
            "accuracy": acc,
            "global_gold_histogram": {str(i): gold_global[i] for i in range(max_candidates)},
            "global_pred_histogram": {str(i): pred_global[i] for i in range(max_candidates)},
            "pred_index_dominant": gi,
            "pred_index_dominant_mass": pred_max_mass,
            "pred_index_0_rate": pred0_rate,
            "normalized_pred_entropy": pred_ent / max_ent,
            "normalized_gold_entropy": gold_ent / max_ent,
            "always_predict_0_accuracy_on_gold_0": always0_acc,
            "n_gold_0_rows": always0_denom,
            "n_ambiguous_keys": len(ambiguous_keys),
            "n_keys_fully_collapsed_on_index": len(fully_collapsed),
            "per_key": key_rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print()
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main(sys.argv[1:])
