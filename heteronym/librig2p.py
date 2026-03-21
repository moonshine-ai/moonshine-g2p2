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
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:  # pragma: no cover
    hf_hub_download = None  # type: ignore[misc, assignment]
    _HF_IMPORT_ERROR = e
else:
    _HF_IMPORT_ERROR = None


SPECIAL_PAD = "<pad>"
SPECIAL_UNK = "<unk>"


@dataclass(frozen=True)
class HomographRecord:
    """One training/eval example."""

    char_text: str
    homograph: str
    homograph_wordid: str
    homograph_char_start: int
    homograph_char_end: int


def load_homograph_json(path: Path | str) -> list[HomographRecord]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        blob: dict[str, Any] = json.load(f)
    out: list[HomographRecord] = []
    for _sid, row in blob.items():
        out.append(
            HomographRecord(
                char_text=row["char"],
                homograph=row["homograph"],
                homograph_wordid=row["homograph_wordid"],
                homograph_char_start=int(row["homograph_char_start"]),
                homograph_char_end=int(row["homograph_char_end"]),
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


def build_homograph_candidate_tables(
    records: list[HomographRecord],
    *,
    max_candidates: int,
    group_key: str = "lower",
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """
    For each surface homograph key, collect sorted unique `homograph_wordid`
    strings and map each id to a class index in 0..K-1.

    group_key
        ``lower`` — fold homograph string with ``str.lower()`` for grouping
        (recommended for LibriG2P uppercase sentences).
        ``exact`` — use homograph field verbatim.
    """
    groups: dict[str, set[str]] = {}
    for r in records:
        key = r.homograph.lower() if group_key == "lower" else r.homograph
        groups.setdefault(key, set()).add(r.homograph_wordid)

    ordered: dict[str, list[str]] = {}
    label_maps: dict[str, dict[str, int]] = {}
    for key, ids in groups.items():
        lst = sorted(ids)
        if len(lst) > max_candidates:
            raise ValueError(
                f"homograph {key!r} has {len(lst)} pronunciations "
                f"(> max_candidates={max_candidates}): {lst}"
            )
        ordered[key] = lst
        label_maps[key] = {wid: i for i, wid in enumerate(lst)}
    return ordered, label_maps


def save_training_artifacts(
    out_dir: Path | str,
    *,
    char_vocab: CharVocab,
    ordered_candidates: dict[str, list[str]],
    label_maps: dict[str, dict[str, int]],
    max_candidates: int,
    group_key: str,
) -> None:
    """Write vocab and homograph candidate tables for inference."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "char_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(char_vocab.stoi, f, ensure_ascii=False, indent=2)
    payload = {
        "max_candidates": max_candidates,
        "group_key": group_key,
        "ordered_candidates": ordered_candidates,
        "label_maps": label_maps,
    }
    with (out_dir / "homograph_index.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_training_artifacts(dir_path: Path | str) -> tuple[CharVocab, dict[str, list[str]], dict[str, dict[str, int]], int, str]:
    dir_path = Path(dir_path)
    with (dir_path / "char_vocab.json").open(encoding="utf-8") as f:
        stoi: dict[str, int] = json.load(f)
    cv = CharVocab.from_stoi(stoi)
    with (dir_path / "homograph_index.json").open(encoding="utf-8") as f:
        payload = json.load(f)
    return (
        cv,
        payload["ordered_candidates"],
        payload["label_maps"],
        int(payload["max_candidates"]),
        str(payload["group_key"]),
    )


class CharVocab:
    """Char -> id with fixed specials at 0 and 1."""

    def __init__(self, chars: list[str]) -> None:
        self.stoi: dict[str, int] = {SPECIAL_PAD: 0, SPECIAL_UNK: 1}
        for c in chars:
            if c not in self.stoi:
                self.stoi[c] = len(self.stoi)
        self._sync_itos()

    def _sync_itos(self) -> None:
        self.itos = [""] * len(self.stoi)
        for s, i in self.stoi.items():
            self.itos[i] = s

    @classmethod
    def from_stoi(cls, stoi: dict[str, int]) -> CharVocab:
        self = object.__new__(cls)
        self.stoi = dict(stoi)
        self._sync_itos()
        return self

    @classmethod
    def from_records(cls, records: list[HomographRecord]) -> CharVocab:
        seen: set[str] = set()
        for r in records:
            for ch in r.char_text:
                seen.add(ch)
        return cls(sorted(seen))

    def to_jsonable(self) -> dict[str, int]:
        return dict(self.stoi)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[SPECIAL_UNK]
        return [self.stoi.get(c, unk) for c in text]

    def __len__(self) -> int:
        return len(self.stoi)


def iter_encoded_batches(
    records: list[HomographRecord],
    *,
    char_vocab: CharVocab,
    ordered_candidates: dict[str, list[str]],
    label_maps: dict[str, dict[str, int]],
    group_key: str,
    max_seq_len: int,
    max_candidates: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    import random

    idx = list(range(len(records)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)

    batch_ids: list[list[int]] = []
    batch_span: list[list[float]] = []
    batch_cm: list[list[bool]] = []
    batch_y: list[int] = []

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
            "candidate_mask": torch.tensor(cm, dtype=torch.bool),
            "labels": torch.tensor(batch_y, dtype=torch.long),
        }
        batch_ids.clear()
        batch_span.clear()
        batch_cm.clear()
        batch_y.clear()
        return out

    for i in idx:
        r = records[i]
        gkey = r.homograph.lower() if group_key == "lower" else r.homograph
        lm = label_maps[gkey]
        if r.homograph_wordid not in lm:
            logger.warning("skip row: unknown wordid %s for %s", r.homograph_wordid, gkey)
            continue
        y = lm[r.homograph_wordid]
        ids = char_vocab.encode(r.char_text)
        span = [0.0] * len(ids)
        s, e = r.homograph_char_start, r.homograph_char_end
        e = min(e, len(ids))
        s = min(max(0, s), len(ids))
        for j in range(s, e):
            span[j] = 1.0
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
            span = span[:max_seq_len]
        if sum(span) < 1.0:
            logger.warning("skip row: homograph span outside truncated window for %s", gkey)
            continue
        cands = ordered_candidates[gkey]
        cm = [True] * len(cands) + [False] * (max_candidates - len(cands))
        batch_ids.append(ids)
        batch_span.append(span)
        batch_cm.append(cm)
        batch_y.append(y)
        if len(batch_y) >= batch_size:
            b = flush()
            if b is not None:
                yield b
    b = flush()
    if b is not None:
        yield b
