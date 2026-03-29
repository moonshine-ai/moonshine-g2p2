"""
From-scratch BERT-style **BasicTokenizer** + **WordPiece** using only ``vocab.txt``
(Hugging Face line order: id = line index). Matches ``transformers`` slow BERT
tokenization and the ``tokenizers`` BertNormalizer + WordPiece model for typical
text (including CTB9 Chinese).

Designed for a future C++ port: no ``transformers``/``tokenizers`` dependency
in the hot path—only stdlib + ``json`` for config I/O.

Offset mapping follows the fast tokenizer: each subword maps to ``(start, end)``
character spans in the **input string** passed to ``encode`` (exclusive ``end``).
``[CLS]`` / ``[SEP]`` use ``(0, 0)`` when ``add_special_tokens=True``.
"""

from __future__ import annotations

import json
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def load_vocab(vocab_file: Path | str) -> OrderedDict[str, int]:
    vocab: OrderedDict[str, int] = OrderedDict()
    path = Path(vocab_file)
    with path.open(encoding="utf-8") as reader:
        for index, line in enumerate(reader):
            token = line.rstrip("\n")
            vocab[token] = index
    return vocab


def _is_whitespace(char: str) -> bool:
    if char in " \t\n\r":
        return True
    return unicodedata.category(char) == "Zs"


def _is_control(char: str) -> bool:
    if char in "\t\n\r":
        return False
    return unicodedata.category(char).startswith("C")


def _is_punctuation(char: str) -> bool:
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def _is_chinese_char(cp: int) -> bool:
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def _nfc_annotate(seq: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Apply NFC per character, preserving (lo, hi) into the original string."""
    if not seq:
        return []
    s = "".join(t[0] for t in seq)
    s_nfc = unicodedata.normalize("NFC", s)
    spans_src = [(t[1], t[2]) for t in seq]
    if len(s_nfc) == len(s):
        return [(s_nfc[k], spans_src[k][0], spans_src[k][1]) for k in range(len(s_nfc))]
    out: list[tuple[str, int, int]] = []
    i = 0
    for out_c in s_nfc:
        if i >= len(s):
            break
        j = i + 1
        while j <= len(s):
            chunk = s[i:j]
            if unicodedata.normalize("NFC", chunk) == out_c:
                lo = min(spans_src[k][0] for k in range(i, j))
                hi = max(spans_src[k][1] for k in range(i, j))
                out.append((out_c, lo, hi))
                i = j
                break
            j += 1
        else:
            lo, hi = spans_src[i]
            out.append((out_c, lo, hi))
            i += 1
    return out


def _clean_text_ann(seq: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for char, lo, hi in seq:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            out.append((" ", lo, hi))
        else:
            out.append((char, lo, hi))
    return out


def _tokenize_chinese_chars_ann(seq: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for char, lo, hi in seq:
        cp = ord(char)
        if _is_chinese_char(cp):
            out.append((" ", lo, lo))
            out.append((char, lo, hi))
            out.append((" ", hi, hi))
        else:
            out.append((char, lo, hi))
    return out


def _strip_ann(seq: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    while seq and seq[0][0] == " ":
        seq = seq[1:]
    while seq and seq[-1][0] == " ":
        seq = seq[:-1]
    return seq


def _split_ws_ann(seq: list[tuple[str, int, int]]) -> list[list[tuple[str, int, int]]]:
    seq = _strip_ann(seq)
    if not seq:
        return []
    words: list[list[tuple[str, int, int]]] = []
    cur: list[tuple[str, int, int]] = []
    for t in seq:
        if t[0] == " ":
            if cur:
                words.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        words.append(cur)
    return words


def _lowercase_ann(seq: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    return [(c.lower(), lo, hi) for c, lo, hi in seq]


def _strip_accents_ann(seq: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for char, lo, hi in seq:
        for ch in unicodedata.normalize("NFD", char):
            if unicodedata.category(ch) == "Mn":
                continue
            out.append((ch, lo, hi))
    return out


def _run_split_on_punc_ann(
    piece: list[tuple[str, int, int]],
    never_split: set[str],
    *,
    do_split_on_punc: bool,
) -> list[list[tuple[str, int, int]]]:
    s = "".join(c for c, _, _ in piece)
    if not do_split_on_punc or s in never_split:
        return [piece]
    chars = list(piece)
    i = 0
    start_new_word = True
    output: list[list[tuple[str, int, int]]] = []
    while i < len(chars):
        char, lo, hi = chars[i]
        if _is_punctuation(char):
            output.append([(char, lo, hi)])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append((char, lo, hi))
        i += 1
    return output


def _final_ws_join_split(
    punc_pieces: list[list[tuple[str, int, int]]],
) -> list[list[tuple[str, int, int]]]:
    """Mirror ``whitespace_tokenize(" ".join(...))``; each word is char-level ``(c, lo, hi)``."""
    if not punc_pieces:
        return []
    merged: list[tuple[str, int, int]] = []
    for i, p in enumerate(punc_pieces):
        if not p:
            continue
        if i > 0 and merged:
            lo = p[0][1]
            merged.append((" ", lo, lo))
        merged.extend(p)
    merged = _strip_ann(merged)
    if not merged:
        return []
    words: list[list[tuple[str, int, int]]] = []
    cur: list[tuple[str, int, int]] = []
    for t in merged:
        if t[0] == " ":
            if cur:
                words.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        words.append(cur)
    return words


def basic_tokenize_with_spans(
    text: str,
    *,
    never_split: set[str],
    do_lower_case: bool,
    tokenize_chinese_chars: bool,
    strip_accents: bool | None,
    do_split_on_punc: bool = True,
) -> list[list[tuple[str, int, int]]]:
    ann = [(c, i, i + 1) for i, c in enumerate(text)]
    ann = _clean_text_ann(ann)
    if tokenize_chinese_chars:
        ann = _tokenize_chinese_chars_ann(ann)
    ann = _nfc_annotate(ann)
    words_ann = _split_ws_ann(ann)

    if strip_accents is None:
        effective_strip = do_lower_case
    else:
        effective_strip = strip_accents

    flat_after_punc: list[list[tuple[str, int, int]]] = []
    for w in words_ann:
        wstr = "".join(c for c, _, _ in w)
        if wstr in never_split:
            wtok = w
        else:
            wtok = w
            if do_lower_case:
                wtok = _lowercase_ann(wtok)
                if effective_strip:
                    wtok = _strip_accents_ann(wtok)
            elif effective_strip:
                wtok = _strip_accents_ann(wtok)
        for seg in _run_split_on_punc_ann(wtok, never_split, do_split_on_punc=do_split_on_punc):
            if seg:
                flat_after_punc.append(seg)

    return _final_ws_join_split(flat_after_punc)


def wordpiece_tokenize_with_spans(
    basic_words: list[list[tuple[str, int, int]]],
    vocab: dict[str, int],
    unk_token: str,
    *,
    max_input_chars_per_word: int = 100,
) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for chars in basic_words:
        if not chars:
            continue
        wlo = min(t[1] for t in chars)
        whi = max(t[2] for t in chars)
        if len(chars) > max_input_chars_per_word:
            out.append((unk_token, wlo, whi))
            continue
        start = 0
        is_bad = False
        while start < len(chars):
            end = len(chars)
            cur_substr: str | None = None
            cur_end = start
            while start < end:
                substr = "".join(chars[i][0] for i in range(start, end))
                if start > 0:
                    substr = "##" + substr
                if substr in vocab:
                    cur_substr = substr
                    cur_end = end
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            p_lo = chars[start][1]
            p_hi = chars[cur_end - 1][2]
            out.append((cur_substr, p_lo, p_hi))
            start = cur_end
        if is_bad:
            out.append((unk_token, wlo, whi))
    return out


@dataclass(frozen=True)
class VocabEncoding:
    ids: list[int]
    tokens: list[str]
    offsets: list[tuple[int, int]]


class BertWordPieceTokenizer:
    """BERT BasicTokenizer + WordPiece from ``vocab.txt`` only."""

    def __init__(
        self,
        vocab_path: Path | str,
        *,
        do_lower_case: bool = True,
        tokenize_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        never_split: Iterable[str] | None = None,
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "",
        mask_token: str = "[MASK]",
        do_split_on_punc: bool = True,
        max_input_chars_per_word: int = 100,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.vocab: OrderedDict[str, int] = load_vocab(self.vocab_path)
        self.ids_to_token: dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
        self.max_input_chars_per_word = max_input_chars_per_word
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        base_ns = set(never_split) if never_split else set()
        for t in (unk_token, cls_token, sep_token, pad_token, mask_token):
            base_ns.add(t)
        self.never_split = base_ns

    def token_to_id(self, token: str) -> int | None:
        i = self.vocab.get(token)
        return i if i is not None else None

    def encode(self, text: str, add_special_tokens: bool = True) -> VocabEncoding:
        basic = basic_tokenize_with_spans(
            text,
            never_split=self.never_split,
            do_lower_case=self.do_lower_case,
            tokenize_chinese_chars=self.tokenize_chinese_chars,
            strip_accents=self.strip_accents,
            do_split_on_punc=self.do_split_on_punc,
        )
        wp = wordpiece_tokenize_with_spans(
            basic,
            self.vocab,
            self.unk_token,
            max_input_chars_per_word=self.max_input_chars_per_word,
        )
        unk_id = self.vocab[self.unk_token]
        ids = [self.vocab.get(t, unk_id) for t, _, _ in wp]
        tokens = [t for t, _, _ in wp]
        offsets = [(b, e) for _, b, e in wp]

        if add_special_tokens:
            cls_id = self.vocab[self.cls_token]
            sep_id = self.vocab[self.sep_token]
            ids = [cls_id] + ids + [sep_id]
            tokens = [self.cls_token] + tokens + [self.sep_token]
            offsets = [(0, 0)] + offsets + [(0, 0)]

        return VocabEncoding(ids=ids, tokens=tokens, offsets=offsets)


def load_bert_wordpiece_tokenizer(tokenizer_dir: Path | str) -> BertWordPieceTokenizer:
    d = Path(tokenizer_dir)
    cfg_path = d / "tokenizer_config.json"
    cfg: dict = {}
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    never = cfg.get("never_split")
    return BertWordPieceTokenizer(
        d / "vocab.txt",
        do_lower_case=bool(cfg.get("do_lower_case", True)),
        tokenize_chinese_chars=bool(cfg.get("tokenize_chinese_chars", True)),
        strip_accents=cfg.get("strip_accents"),
        never_split=never if never else None,
        unk_token=str(cfg.get("unk_token", "[UNK]")),
        cls_token=str(cfg.get("cls_token", "[CLS]")),
        sep_token=str(cfg.get("sep_token", "[SEP]")),
        pad_token=str(cfg.get("pad_token", "")),
        mask_token=str(cfg.get("mask_token", "[MASK]")),
    )


def _compare_to_tokenizers_json(tokenizer_dir: Path, samples: list[str]) -> int:
    from tokenizers import Tokenizer

    ref = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    mine = load_bert_wordpiece_tokenizer(tokenizer_dir)
    bad = 0
    for s in samples:
        r = ref.encode(s, add_special_tokens=True)
        m = mine.encode(s, add_special_tokens=True)
        if list(r.ids) != m.ids or list(r.offsets) != m.offsets:
            bad += 1
            if bad <= 3:
                print("mismatch", repr(s[:60]))
                print("  ref ids", list(r.ids)[:20])
                print("  got ids", m.ids[:20])
                print("  ref off", list(r.offsets)[:12])
                print("  got off", m.offsets[:12])
    return bad


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parent.parent
    td = root / "models" / "zh_hans" / "hanlp_ctb9_electra_small" / "tokenizer"
    tests = [
        "你好世界",
        "Hello 中国",
        "café",
        "naïve",
        "a,b",
        "  spaced  ",
        "é",
        "test…",
    ]
    wiki = root / "data" / "zh" / "wiki_sample.txt"
    if wiki.is_file():
        for line in wiki.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t:
                tests.append(t)
            if len(tests) > 2000:
                break
    n = _compare_to_tokenizers_json(td, tests)
    print("compared", len(tests), "strings;", n, "mismatches")
    sys.exit(1 if n else 0)
