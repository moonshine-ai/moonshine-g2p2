"""Character vocabulary and specials shared by heteronym and OOV G2P."""

from __future__ import annotations

SPECIAL_PAD = "<pad>"
SPECIAL_UNK = "<unk>"


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

    def to_jsonable(self) -> dict[str, int]:
        return dict(self.stoi)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[SPECIAL_UNK]
        return [self.stoi.get(c, unk) for c in text]

    def __len__(self) -> int:
        return len(self.stoi)
