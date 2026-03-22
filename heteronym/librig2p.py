"""
Load LibriG2P homograph JSON from the Hugging Face dataset repo (raw files).

The published ``datasets.load_dataset`` hub entry can fail on newer library
versions; this module uses ``huggingface_hub.hf_hub_download`` and reads
``dataset/homograph_{train,valid,test}.json`` (JSON objects keyed by sample id;
see `flexthink/librig2p-nostress-space <https://huggingface.co/datasets/flexthink/librig2p-nostress-space>`_).

For other corpora (e.g. `IPA-CHILDES <https://huggingface.co/datasets/phonemetransformers/IPA-CHILDES>`_),
add a thin adapter that yields :class:`HomographRecord` (sentence text, surface
homograph, word-id label, character span). Grouping and labels stay
data-driven via :func:`build_homograph_candidate_tables`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from g2p_common import SPECIAL_PAD, CharVocab, inference_context_window

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:  # pragma: no cover
    hf_hub_download = None  # type: ignore[misc, assignment]
    _HF_IMPORT_ERROR = e
else:
    _HF_IMPORT_ERROR = None


@dataclass(frozen=True)
class HomographRecord:
    """One training/eval example."""

    char_text: str
    homograph: str
    homograph_wordid: str
    homograph_char_start: int
    homograph_char_end: int
    # Optional IPA string for this label (LibriG2P-style abstract wordids).
    # When omitted, the pronunciation text for modeling is ``homograph_wordid``
    # (eSpeak-style corpora store IPA there).
    homograph_wordid_ipa: str | None = None


def load_homograph_json(path: Path | str) -> list[HomographRecord]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        blob: dict[str, Any] = json.load(f)
    out: list[HomographRecord] = []
    for _sid, row in blob.items():
        ipa_opt = row.get("homograph_wordid_ipa")
        ipa_str = str(ipa_opt).strip() if ipa_opt is not None else None
        if ipa_str == "":
            ipa_str = None
        out.append(
            HomographRecord(
                char_text=row["char"],
                homograph=row["homograph"],
                homograph_wordid=row["homograph_wordid"],
                homograph_char_start=int(row["homograph_char_start"]),
                homograph_char_end=int(row["homograph_char_end"]),
                homograph_wordid_ipa=ipa_str,
            )
        )
    return out


def download_librig2p_homograph_split(
    split: str,
    *,
    cache_dir: str | None = None,
    repo_id: str = "flexthink/librig2p-nostress-space",
) -> Path:
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    if hf_hub_download is None:
        raise ImportError(
            "huggingface_hub is required to download LibriG2P. "
            f"({ _HF_IMPORT_ERROR })"
        )
    name = f"dataset/homograph_{split}.json"
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=name,
            cache_dir=cache_dir,
        )
    )


def _wordid_to_ipa_map(records: list[HomographRecord], *, group_key: str) -> dict[tuple[str, str], str]:
    """
    Map (homograph_group_key, homograph_wordid) -> IPA (or pronunciation) string.

    If ``homograph_wordid_ipa`` is set on any row, all rows for that pair must agree.
    Otherwise the IPA text defaults to ``homograph_wordid`` (eSpeak / IPA-as-id corpora).
    """
    out: dict[tuple[str, str], str] = {}
    for r in records:
        g = _group_key_for_record(r, group_key)
        wid = r.homograph_wordid
        ipa = (r.homograph_wordid_ipa or wid).strip()
        k = (g, wid)
        prev = out.get(k)
        if prev is not None and prev != ipa:
            raise ValueError(
                f"Inconsistent homograph_wordid_ipa for homograph={g!r} wordid={wid!r}: "
                f"{prev!r} vs {ipa!r}"
            )
        out[k] = ipa
    return out


def build_homograph_candidate_tables(
    records: list[HomographRecord],
    *,
    max_candidates: int,
    group_key: str = "lower",
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]], dict[str, list[str]]]:
    """
    For each surface homograph key, collect sorted unique `homograph_wordid`
    strings and map each id to a class index in 0..K-1.

    Also returns ``ordered_ipa``: for each key, parallel list of IPA (phoneme)
    strings in candidate index order (same slots as ``ordered``).

    group_key
        ``lower`` — fold homograph string with ``str.lower()`` for grouping
        (recommended for LibriG2P uppercase sentences).
        ``exact`` — use homograph field verbatim.
    """
    wid_ipa = _wordid_to_ipa_map(records, group_key=group_key)
    groups: dict[str, set[str]] = {}
    for r in records:
        key = r.homograph.lower() if group_key == "lower" else r.homograph
        groups.setdefault(key, set()).add(r.homograph_wordid)

    ordered: dict[str, list[str]] = {}
    ordered_ipa: dict[str, list[str]] = {}
    label_maps: dict[str, dict[str, int]] = {}
    for key, ids in groups.items():
        lst = sorted(ids)
        if len(lst) > max_candidates:
            raise ValueError(
                f"homograph {key!r} has {len(lst)} pronunciations "
                f"(> max_candidates={max_candidates}): {lst}"
            )
        ordered[key] = lst
        ordered_ipa[key] = [wid_ipa[(key, w)] for w in lst]
        label_maps[key] = {wid: i for i, wid in enumerate(lst)}
    return ordered, label_maps, ordered_ipa


def save_training_artifacts(
    out_dir: Path | str,
    *,
    char_vocab: CharVocab,
    ipa_char_vocab: CharVocab,
    ordered_candidates: dict[str, list[str]],
    ordered_candidate_ipa: dict[str, list[str]],
    label_maps: dict[str, dict[str, int]],
    max_candidates: int,
    group_key: str,
) -> None:
    """Write vocab and homograph candidate tables for inference."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "char_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(char_vocab.stoi, f, ensure_ascii=False, indent=2)
    with (out_dir / "ipa_char_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(ipa_char_vocab.stoi, f, ensure_ascii=False, indent=2)
    payload = {
        "max_candidates": max_candidates,
        "group_key": group_key,
        "ordered_candidates": ordered_candidates,
        "ordered_candidate_ipa": ordered_candidate_ipa,
        "label_maps": label_maps,
    }
    with (out_dir / "homograph_index.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2    )


def build_char_vocab_from_homograph_records(
    records: list[HomographRecord],
    *,
    extra_chars: str | None = None,
) -> CharVocab:
    """Build a :class:`~g2p_common.CharVocab` from all characters seen in *records*."""
    seen: set[str] = set()
    for r in records:
        for ch in r.char_text:
            seen.add(ch)
    if extra_chars:
        seen.update(extra_chars)
    return CharVocab(sorted(seen))


def build_ipa_char_vocab_from_ordered_ipa(
    ordered_ipa: dict[str, list[str]],
    *,
    extra_chars: str | None = None,
) -> CharVocab:
    """Character vocabulary for IPA strings (per-candidate pronunciation text)."""
    seen: set[str] = set()
    for lst in ordered_ipa.values():
        for s in lst:
            for ch in s:
                seen.add(ch)
    if extra_chars:
        seen.update(extra_chars)
    return CharVocab(sorted(seen))


def max_encoded_ipa_len(
    ordered_ipa: dict[str, list[str]],
    ipa_char_vocab: CharVocab,
    *,
    cap: int,
) -> int:
    """Longest IPA encoding length in the index, capped (at least 1)."""
    m = 1
    for lst in ordered_ipa.values():
        for s in lst:
            m = max(m, len(ipa_char_vocab.encode(s)))
    return min(max(m, 1), cap)


def load_training_artifacts(
    dir_path: Path | str,
) -> tuple[CharVocab, CharVocab, dict[str, list[str]], dict[str, list[str]], dict[str, dict[str, int]], int, str]:
    dir_path = Path(dir_path)
    with (dir_path / "char_vocab.json").open(encoding="utf-8") as f:
        stoi: dict[str, int] = json.load(f)
    cv = CharVocab.from_stoi(stoi)
    ipa_path = dir_path / "ipa_char_vocab.json"
    if not ipa_path.is_file():
        raise FileNotFoundError(
            f"{ipa_path} not found (required for heteronym IPA conditioning; "
            "re-save artifacts with save_training_artifacts or retrain)."
        )
    with ipa_path.open(encoding="utf-8") as f:
        ipa_stoi: dict[str, int] = json.load(f)
    ipa_cv = CharVocab.from_stoi(ipa_stoi)
    with (dir_path / "homograph_index.json").open(encoding="utf-8") as f:
        payload = json.load(f)
    ordered = payload["ordered_candidates"]
    oci = payload.get("ordered_candidate_ipa")
    if not isinstance(oci, dict):
        oci = {k: [ordered[k][i] for i in range(len(ordered[k]))] for k in ordered}
    return (
        cv,
        ipa_cv,
        ordered,
        oci,
        payload["label_maps"],
        int(payload["max_candidates"]),
        str(payload["group_key"]),
    )


def _group_key_for_record(r: HomographRecord, group_key: str) -> str:
    return r.homograph.lower() if group_key == "lower" else r.homograph


def _allowed_noise_inserts(char_vocab: CharVocab) -> set[str]:
    return {c for c in ".,;:'\"-" if c in char_vocab.stoi}


def _safe_surface_noise(
    text: str,
    span_s: int,
    span_e: int,
    rng: random.Random,
    prob: float,
    allowed_punct: set[str],
) -> tuple[str, int, int]:
    """
    Light edits strictly outside ``[span_s, span_e)``: duplicate/collapse spaces,
    or insert punctuation after a space at a safe boundary.
    """
    if prob <= 0 or rng.random() > prob:
        return text, span_s, span_e
    ch = list(text)
    s, e = span_s, span_e
    n_ops = rng.randint(1, 2)
    for _ in range(n_ops):
        if not ch:
            break
        kind = rng.choice(["dup_space", "collapse_space", "punct"])
        if kind == "dup_space":
            candidates = [
                i
                for i, c in enumerate(ch)
                if c == " " and (i < s or i >= e) and (i + 1 <= s or i + 1 >= e)
            ]
            if not candidates:
                continue
            i = rng.choice(candidates)
            k = i + 1
            ch.insert(k, " ")
            if k <= s:
                s += 1
                e += 1
        elif kind == "collapse_space":
            doubles = [
                i
                for i in range(len(ch) - 1)
                if ch[i] == " " and ch[i + 1] == " " and (i + 1 < s or i >= e)
            ]
            if not doubles:
                continue
            i = rng.choice(doubles)
            del ch[i]
            if i < s:
                s -= 1
                e -= 1
        elif kind == "punct" and allowed_punct:
            slots = [
                i
                for i, c in enumerate(ch)
                if c == " "
                and (i < s or i >= e)
                and (i + 1 <= s or i + 1 >= e)
            ]
            if not slots:
                continue
            i = rng.choice(slots)
            p = rng.choice(list(allowed_punct))
            k = i + 1
            ch.insert(k, p)
            if k <= s:
                s += 1
                e += 1
    s = min(max(0, s), len(ch))
    e = min(max(s, e), len(ch))
    return "".join(ch), s, e


def _random_context_window(
    text: str,
    span_s: int,
    span_e: int,
    max_seq_len: int,
    rng: random.Random,
    *,
    max_left_pad: int = 48,
) -> tuple[str, int, int] | None:
    """Random crop if ``len(text) > max_seq_len``; optional left-padding if shorter."""
    L = len(text)
    s, e = span_s, span_e
    if e > L or s < 0 or s >= e:
        return None
    if L > max_seq_len:
        lo = max(0, e - max_seq_len)
        hi = min(s, L - max_seq_len)
        if lo > hi:
            return None
        w0 = rng.randint(lo, hi)
        text = text[w0 : w0 + max_seq_len]
        s -= w0
        e -= w0
        L = len(text)
    if L < max_seq_len:
        budget = max_seq_len - L
        left = rng.randint(0, min(budget, max_left_pad)) if budget > 0 else 0
        if left:
            text = " " * left + text
            s += left
            e += left
    return text, s, e


def apply_train_augmentation(
    r: HomographRecord,
    *,
    char_vocab: CharVocab,
    max_seq_len: int,
    rng: random.Random,
    surface_noise_prob: float,
) -> tuple[str, int, int] | None:
    """
    Return augmented ``(char_text, homograph_char_start, homograph_char_end)``,
    or ``None`` if the span cannot be placed in a ``max_seq_len`` window.
    """
    text = r.char_text
    s = min(max(0, r.homograph_char_start), len(text))
    e = min(max(s, r.homograph_char_end), len(text))
    if s >= e:
        return None
    punct = _allowed_noise_inserts(char_vocab)
    text, s, e = _safe_surface_noise(text, s, e, rng, surface_noise_prob, punct)
    out = _random_context_window(text, s, e, max_seq_len, rng)
    return out


def _training_index_order(
    records: list[HomographRecord],
    *,
    label_maps: dict[str, dict[str, int]],
    group_key: str,
    shuffle: bool,
    balance_training: bool,
    rng: random.Random,
) -> list[int]:
    if not shuffle:
        return list(range(len(records)))
    valid_ix: list[int] = []
    for i, r in enumerate(records):
        g = _group_key_for_record(r, group_key)
        if g not in label_maps:
            continue
        if r.homograph_wordid not in label_maps[g]:
            continue
        valid_ix.append(i)
    if balance_training and valid_ix:
        keys = [
            (_group_key_for_record(records[i], group_key), records[i].homograph_wordid)
            for i in valid_ix
        ]
        counts = Counter(keys)
        weights = [1.0 / counts[k] for k in keys]
        return rng.choices(valid_ix, weights=weights, k=len(records))
    if balance_training and not valid_ix:
        logger.warning("balance_training: no valid rows, using uniform shuffle")
    idx = list(range(len(records)))
    rng.shuffle(idx)
    return idx


def encode_ipa_candidate_slots(
    ipa_char_vocab: CharVocab,
    ipa_per_slot: list[str],
    *,
    max_candidates: int,
    max_ipa_len: int,
) -> tuple[list[list[int]], list[list[bool]]]:
    """
    Encode parallel IPA strings for each candidate index (length ``max_candidates``).

    Invalid / padding candidate indices use an all-pad sequence with mask False.
    """
    pad_id = ipa_char_vocab.stoi[SPECIAL_PAD]
    ids: list[list[int]] = []
    attn: list[list[bool]] = []
    for k in range(max_candidates):
        if k < len(ipa_per_slot):
            raw = ipa_char_vocab.encode(ipa_per_slot[k])[:max_ipa_len]
        else:
            raw = []
        row = raw + [pad_id] * (max_ipa_len - len(raw))
        ids.append(row[:max_ipa_len])
        attn.append([j < len(raw) for j in range(max_ipa_len)])
    return ids, attn


def iter_encoded_batches(
    records: list[HomographRecord],
    *,
    char_vocab: CharVocab,
    ipa_char_vocab: CharVocab,
    ordered_candidates: dict[str, list[str]],
    ordered_ipa: dict[str, list[str]],
    label_maps: dict[str, dict[str, int]],
    group_key: str,
    max_seq_len: int,
    max_candidates: int,
    max_ipa_len: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    train_augment: bool = False,
    balance_training: bool = False,
    surface_noise_prob: float = 0.3,
    include_group_keys: bool = False,
) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    train_augment = bool(train_augment and shuffle)
    balance_training = bool(balance_training and shuffle)
    idx = _training_index_order(
        records,
        label_maps=label_maps,
        group_key=group_key,
        shuffle=shuffle,
        balance_training=balance_training,
        rng=rng,
    )

    batch_ids: list[list[int]] = []
    batch_span: list[list[float]] = []
    batch_cm: list[list[bool]] = []
    batch_y: list[int] = []
    batch_ipa: list[list[list[int]]] = []
    batch_ipa_attn: list[list[list[bool]]] = []
    batch_gkeys: list[str] = []

    def flush() -> dict[str, Any] | None:
        if not batch_y:
            return None
        import torch

        t_max = max(len(x) for x in batch_ids)
        t_max = min(t_max, max_seq_len)
        pad_id = char_vocab.stoi[SPECIAL_PAD]
        ids = []
        mask = []
        span = []
        for row_ids, row_span in zip(batch_ids, batch_span):
            row_ids = row_ids[:max_seq_len]
            row_span = row_span[:max_seq_len]
            pad = t_max - len(row_ids)
            ids.append(row_ids + [pad_id] * pad)
            mask.append([1] * len(row_ids) + [0] * pad)
            span.append(row_span + [0.0] * pad)
        cm = []
        for row in batch_cm:
            cm.append(row + [False] * (max_candidates - len(row)))
        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "span_mask": torch.tensor(span, dtype=torch.float32),
            "ipa_input_ids": torch.tensor(batch_ipa, dtype=torch.long),
            "ipa_attention_mask": torch.tensor(batch_ipa_attn, dtype=torch.bool),
            "candidate_mask": torch.tensor(cm, dtype=torch.bool),
            "labels": torch.tensor(batch_y, dtype=torch.long),
        }
        if include_group_keys:
            out["group_keys"] = list(batch_gkeys)
        batch_ids.clear()
        batch_span.clear()
        batch_cm.clear()
        batch_y.clear()
        batch_ipa.clear()
        batch_ipa_attn.clear()
        batch_gkeys.clear()
        return out

    for i in idx:
        r = records[i]
        gkey = _group_key_for_record(r, group_key)
        lm = label_maps[gkey]
        if r.homograph_wordid not in lm:
            # logger.warning("skip row: unknown wordid %s for %s", r.homograph_wordid, gkey)
            continue
        y = lm[r.homograph_wordid]
        if train_augment:
            aug = apply_train_augmentation(
                r,
                char_vocab=char_vocab,
                max_seq_len=max_seq_len,
                rng=rng,
                surface_noise_prob=surface_noise_prob,
            )
            if aug is None:
                # logger.warning("skip row: homograph span does not fit max_seq_len for %s", gkey)
                continue
            char_text, s, e = aug
        else:
            # Deterministic crop/pad so the homograph stays in-window (same as inference).
            # Plain ``text[:max_seq_len]`` drops late-span rows and yields n_tot=0 / acc 0.0.
            win = inference_context_window(
                r.char_text,
                r.homograph_char_start,
                r.homograph_char_end,
                max_seq_len,
            )
            if win is None:
                continue
            char_text, s, e = win
        ids = char_vocab.encode(char_text)
        span = [0.0] * len(ids)
        e = min(e, len(ids))
        s = min(max(0, s), len(ids))
        for j in range(s, e):
            span[j] = 1.0
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            span = span[:max_seq_len]
        if sum(span) < 1.0:
            # logger.warning("skip row: homograph span outside truncated window for %s", gkey)
            continue
        cands = ordered_candidates[gkey]
        ipa_slots = ordered_ipa[gkey]
        cm = [True] * len(cands) + [False] * (max_candidates - len(cands))
        ipa_ids, ipa_m = encode_ipa_candidate_slots(
            ipa_char_vocab,
            ipa_slots,
            max_candidates=max_candidates,
            max_ipa_len=max_ipa_len,
        )
        batch_ids.append(ids)
        batch_span.append(span)
        batch_cm.append(cm)
        batch_y.append(y)
        batch_ipa.append(ipa_ids)
        batch_ipa_attn.append(ipa_m)
        if include_group_keys:
            batch_gkeys.append(gkey)
        if len(batch_y) >= batch_size:
            b = flush()
            if b is not None:
                yield b
    b = flush()
    if b is not None:
        yield b
