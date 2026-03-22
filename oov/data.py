"""
Load OOV G2P JSON, char / phoneme vocabularies, and training batches.

Each record is a **word-only** grapheme string (slice via ``word_char_start``/``end``
when valid) and an eSpeak phone sequence. Batches use teacher forcing:
decoder input ``[BOS, p0, …, p_{L-1}]``, targets ``[p0, …, p_{L-1}, EOS]``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterator

from g2p_common import CharVocab, SPECIAL_PAD

SPECIAL_PHON_PAD = "<pad>"
SPECIAL_PHON_UNK = "<unk>"
SPECIAL_PHON_BOS = "<bos>"
SPECIAL_PHON_EOS = "<eos>"


@dataclass(frozen=True)
class OovRecord:
    char_text: str
    word_char_start: int
    word_char_end: int
    phonemes: tuple[str, ...]
    source: str


def grapheme_string_for_record(r: OovRecord) -> str:
    """Grapheme input only: slice ``char_text[start:end]`` when valid, else full string."""
    t = r.char_text
    s, e = r.word_char_start, r.word_char_end
    if 0 <= s < e <= len(t):
        return t[s:e]
    return t


def load_oov_json(path: Path | str) -> list[OovRecord]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        blob: dict[str, Any] = json.load(f)
    out: list[OovRecord] = []
    for _sid, row in blob.items():
        phones = row["phonemes"]
        if isinstance(phones, str):
            phones = json.loads(phones)
        out.append(
            OovRecord(
                char_text=row["char"],
                word_char_start=int(row["word_char_start"]),
                word_char_end=int(row["word_char_end"]),
                phonemes=tuple(str(p) for p in phones),
                source=str(row.get("source", "")),
            )
        )
    return out


def save_training_artifacts(
    out_dir: Path | str,
    *,
    char_vocab: CharVocab,
    phoneme_vocab_stoi: dict[str, int],
    max_phoneme_len: int,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "char_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(char_vocab.stoi, f, ensure_ascii=False, indent=2)
    with (out_dir / "phoneme_vocab.json").open("w", encoding="utf-8") as f:
        json.dump(phoneme_vocab_stoi, f, ensure_ascii=False, indent=2)
    meta = {"max_phoneme_len": max_phoneme_len}
    with (out_dir / "oov_index.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_training_artifacts(
    dir_path: Path | str,
) -> tuple[CharVocab, dict[str, int], int]:
    dir_path = Path(dir_path)
    with (dir_path / "char_vocab.json").open(encoding="utf-8") as f:
        char_vocab = CharVocab.from_stoi(json.load(f))
    with (dir_path / "phoneme_vocab.json").open(encoding="utf-8") as f:
        phon_stoi: dict[str, int] = json.load(f)
    with (dir_path / "oov_index.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    mpl = meta.get("max_phoneme_len")
    if mpl is None:
        raise KeyError("oov_index.json must contain max_phoneme_len")
    return char_vocab, phon_stoi, int(mpl)


class PhonemeVocab:
    """Phoneme token -> id with fixed specials."""

    def __init__(self, tokens: list[str]) -> None:
        self.stoi: dict[str, int] = {
            SPECIAL_PHON_PAD: 0,
            SPECIAL_PHON_UNK: 1,
            SPECIAL_PHON_BOS: 2,
            SPECIAL_PHON_EOS: 3,
        }
        for t in tokens:
            if t not in self.stoi:
                self.stoi[t] = len(self.stoi)
        self._sync_itos()

    def _sync_itos(self) -> None:
        self.itos = [""] * len(self.stoi)
        for s, i in self.stoi.items():
            self.itos[i] = s

    @classmethod
    def from_stoi(cls, stoi: dict[str, int]) -> PhonemeVocab:
        self = object.__new__(cls)
        self.stoi = dict(stoi)
        self._sync_itos()
        return self

    @classmethod
    def from_records(cls, records: list[OovRecord]) -> PhonemeVocab:
        seen: set[str] = set()
        for r in records:
            seen.update(r.phonemes)
        return cls(sorted(seen))

    def encode_sequence(self, phones: tuple[str, ...]) -> list[int]:
        unk = self.stoi[SPECIAL_PHON_UNK]
        return [self.stoi.get(p, unk) for p in phones]

    def __len__(self) -> int:
        return len(self.stoi)


def build_char_vocab_from_records(
    records: list[OovRecord],
    *,
    extra_chars: str | None = None,
) -> CharVocab:
    seen: set[str] = set()
    for r in records:
        for ch in grapheme_string_for_record(r):
            seen.add(ch)
    if extra_chars:
        seen.update(extra_chars)
    return CharVocab(sorted(seen))


def iter_encoded_batches(
    records: list[OovRecord],
    *,
    char_vocab: CharVocab,
    phoneme_vocab: PhonemeVocab,
    max_seq_len: int,
    max_phoneme_len: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    on_record: Callable[[], None] | None = None,
) -> Iterator[dict[str, Any]]:
    import torch

    rng = random.Random(seed)
    idx = list(range(len(records)))
    if shuffle:
        rng.shuffle(idx)

    bos = phoneme_vocab.stoi[SPECIAL_PHON_BOS]
    eos = phoneme_vocab.stoi[SPECIAL_PHON_EOS]
    pad_p = phoneme_vocab.stoi[SPECIAL_PHON_PAD]

    batch_c_ids: list[list[int]] = []
    batch_dec_in: list[list[int]] = []
    batch_dec_tgt: list[list[int]] = []
    batch_dec_m: list[list[int]] = []

    def flush() -> dict[str, Any] | None:
        if not batch_dec_in:
            return None
        t_max = max(len(x) for x in batch_c_ids)
        t_max = min(t_max, max_seq_len)
        pad_c = char_vocab.stoi[SPECIAL_PAD]
        ids = []
        enc_mask = []
        for row_ids in batch_c_ids:
            row_ids = row_ids[:max_seq_len]
            pad = t_max - len(row_ids)
            ids.append(row_ids + [pad_c] * pad)
            enc_mask.append([1] * len(row_ids) + [0] * pad)

        p_max = min(max(len(x) for x in batch_dec_in), max_phoneme_len)
        dec_in = []
        dec_tgt = []
        dec_key = []
        for a, b, m in zip(batch_dec_in, batch_dec_tgt, batch_dec_m):
            a = a[:p_max]
            b = b[:p_max]
            m = m[:p_max]
            pad = p_max - len(a)
            dec_in.append(a + [pad_p] * pad)
            tgt_row = b + [-100] * pad
            dec_tgt.append(tgt_row)
            dec_key.append(m + [0] * pad)

        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(enc_mask, dtype=torch.bool),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_labels": torch.tensor(dec_tgt, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(dec_key, dtype=torch.bool),
        }
        batch_c_ids.clear()
        batch_dec_in.clear()
        batch_dec_tgt.clear()
        batch_dec_m.clear()
        return out

    for i in idx:
        if on_record is not None:
            on_record()
        r = records[i]
        g = grapheme_string_for_record(r)
        if not g:
            continue
        ids = char_vocab.encode(g)
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]

        enc_ph = phoneme_vocab.encode_sequence(r.phonemes)
        if not enc_ph:
            continue
        L = len(enc_ph)
        if L + 1 > max_phoneme_len:
            continue
        dec_in = [bos] + enc_ph
        dec_tgt = enc_ph + [eos]
        m = [1] * len(dec_in)

        batch_c_ids.append(ids)
        batch_dec_in.append(dec_in)
        batch_dec_tgt.append(dec_tgt)
        batch_dec_m.append(m)
        if len(batch_dec_in) >= batch_size:
            b = flush()
            if b is not None:
                yield b
    b = flush()
    if b is not None:
        yield b
