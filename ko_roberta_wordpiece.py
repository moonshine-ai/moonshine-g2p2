# Portions adapted from HuggingFace transformers (Apache-2.0):
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py
# Used for runtime Korean morph+UPOS without the ``tokenizers`` or ``transformers`` packages.

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple


def _load_vocab(vocab_file: str | Path) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    with open(vocab_file, encoding="utf-8") as reader:
        for index, line in enumerate(reader):
            vocab[line.rstrip("\n")] = index
    return vocab


def _whitespace_tokenize(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return text.split()


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


class _BasicTokenizer:
    def __init__(
        self,
        *,
        do_lower_case: bool,
        tokenize_chinese_chars: bool,
        strip_accents: bool | None,
    ) -> None:
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def _clean_text(self, text: str) -> str:
        out: List[str] = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            out.append(" " if _is_whitespace(char) else char)
        return "".join(out)

    def _run_strip_accents(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def _run_split_on_punc(self, text: str) -> List[str]:
        chars = list(text)
        i = 0
        start_new_word = True
        output: List[List[str]] = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text: str) -> str:
        out: List[str] = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                out.extend([" ", char, " "])
            else:
                out.append(char)
        return "".join(out)

    @staticmethod
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

    def tokenize(self, text: str) -> List[str]:
        text = self._clean_text(text)
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        orig_tokens = _whitespace_tokenize(unicode_normalized_text)
        split_tokens: List[str] = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                if self.strip_accents is not False:
                    token = self._run_strip_accents(token)
            elif self.strip_accents:
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        return _whitespace_tokenize(" ".join(split_tokens))


class _WordpieceTokenizer:
    def __init__(self, vocab: Dict[str, int], unk_token: str, max_input_chars_per_word: int = 100) -> None:
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> List[str]:
        output_tokens: List[str] = []
        for token in _whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens: List[str] = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _normalization_ref(text: str, basic: _BasicTokenizer) -> str:
    t = basic._clean_text(text)
    if basic.tokenize_chinese_chars:
        t = basic._tokenize_chinese_chars(t)
    return unicodedata.normalize("NFC", t)


def _align_basic_tokens(ref: str, basic_tokens: List[str]) -> List[Tuple[str, int, int]]:
    cursor = 0
    spans: List[Tuple[str, int, int]] = []
    for btok in basic_tokens:
        while cursor < len(ref) and ref[cursor].isspace():
            cursor += 1
        if ref[cursor : cursor + len(btok)] != btok:
            raise ValueError(
                f"WordPiece alignment failed at offset {cursor}: expected {btok!r} in {ref!r}"
            )
        spans.append((btok, cursor, cursor + len(btok)))
        cursor += len(btok)
    return spans


def load_tokenizer_config(model_dir: Path) -> dict:
    p = model_dir / "tokenizer_config.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def encode_bert_wordpiece(
    text: str,
    model_dir: Path,
) -> Tuple[List[int], List[str], List[Tuple[int, int]], str]:
    """
    Match HuggingFace ``BertTokenizer`` (this repo's export) for ``vocab.txt`` + config.

    Returns ``input_ids`` (with [CLS] and [SEP]), WordPiece token strings, character offsets into
    ``ref_text``, and ``ref_text`` (NFC string after the same cleaning / CJK spacing as HF basic
    tokenization). Slice surfaces as ``ref_text[s:e]``.
    """
    model_dir = Path(model_dir)
    vocab_path = model_dir / "vocab.txt"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Missing {vocab_path}")
    cfg = load_tokenizer_config(model_dir)
    vocab = _load_vocab(vocab_path)
    unk = str(cfg["unk_token"])
    cls_t = str(cfg["cls_token"])
    sep_t = str(cfg["sep_token"])

    basic = _BasicTokenizer(
        do_lower_case=bool(cfg.get("do_lower_case", True)),
        tokenize_chinese_chars=bool(cfg.get("tokenize_chinese_chars", True)),
        strip_accents=cfg.get("strip_accents"),
    )
    wp = _WordpieceTokenizer(vocab, unk_token=unk)

    ref = _normalization_ref(text, basic)
    basic_tokens = basic.tokenize(text)
    aligned = _align_basic_tokens(ref, basic_tokens)

    pieces: List[str] = []
    offsets: List[Tuple[int, int]] = []
    ids: List[int] = []

    for bstr, s0, e0 in aligned:
        wps = wp.tokenize(bstr)
        cur = s0
        for wpt in wps:
            # OOV whole-word fallback: one unk spans the full basic token (matches HF offset policy).
            if wpt == unk:
                pieces.append(wpt)
                offsets.append((s0, e0))
                ids.append(vocab.get(wpt, vocab[unk]))
                cur = e0
                break
            raw = wpt[2:] if wpt.startswith("##") else wpt
            ln = len(raw)
            if ref[cur : cur + ln] != raw:
                raise ValueError(f"WordPiece span mismatch for {wpt!r} at {cur} in {ref!r}")
            pieces.append(wpt)
            offsets.append((cur, cur + ln))
            ids.append(vocab.get(wpt, vocab[unk]))
            cur += ln

    cls_id = vocab[cls_t]
    sep_id = vocab[sep_t]
    all_ids = [cls_id] + ids + [sep_id]
    all_tokens = [cls_t] + pieces + [sep_t]
    all_offsets = [(0, 0)] + offsets + [(0, 0)]
    return all_ids, all_tokens, all_offsets, ref
