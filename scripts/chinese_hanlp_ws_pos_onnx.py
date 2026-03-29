#!/usr/bin/env python3
"""
Chinese **word segmentation** + **POS tagging** with **ONNX Runtime** only for
neural inference. Preprocessing and decoding are ported from HanLP (see
``ctb9_electra_onnx_pure.py``) and do **not** require ``hanlp`` or ``torch``.

Default model directory::

    models/zh_hans/hanlp_ctb9_electra_small/

Requires::

    pip install onnxruntime numpy tokenizers

    To use the pure-Python ``vocab.txt`` tokenizer instead of ``tokenizer.json``::

        python scripts/chinese_hanlp_ws_pos_onnx.py --tokenizer vocab_txt ...

The export must include ``char_normalize.json`` (from
``export_hanlp_ctb9_tok_pos_onnx.py``) and HuggingFace tokenizer files.

Optional verification against full HanLP PyTorch (first *n* non-empty wiki lines)::

    pip install hanlp torch
    python scripts/chinese_hanlp_ws_pos_onnx.py --verify --verify-lines 100
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ctb9_electra_onnx_pure import (
    cws_features,
    cws_logits_to_words,
    load_char_normalize_json,
    load_electra_tokenizer,
    load_electra_tokenizer_vocab_txt,
    pos_features,
    pos_logits_to_tags,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "zh_hans" / "hanlp_ctb9_electra_small"


@dataclass(frozen=True)
class ChineseToken:
    word: str
    pos: str


@dataclass(frozen=True)
class ChineseSentenceAnnotation:
    sentence: str
    tokens: tuple[ChineseToken, ...]

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "tokens": [{"word": t.word, "pos": t.pos} for t in self.tokens],
        }


class HanlpCtb9OnnxWsPos:
    """CTB9 ELECTRA-small WS + POS using ONNX Runtime + HF tokenizer (no HanLP at inference)."""

    def __init__(
        self,
        model_dir: Path,
        *,
        providers: list[str] | None = None,
        tokenizer_backend: str = "json",
    ) -> None:
        import onnxruntime as ort

        model_dir = Path(model_dir)
        meta_path = model_dir / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing {meta_path}; run scripts/export_hanlp_ctb9_tok_pos_onnx.py first.")
        tok_dir = model_dir / "tokenizer"
        if not (tok_dir / "tokenizer_config.json").is_file():
            raise FileNotFoundError(f"Missing exported HF tokenizer under {tok_dir}.")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._tag_vocab_tok: list[str] = meta["tag_vocab_tok"]
        self._tag_vocab_pos: list[str] = meta["tag_vocab_pos"]
        self._pos_span_inner: int = int(meta.get("pos_token_span_inner", 16))

        self._char_map = load_char_normalize_json(model_dir / "char_normalize.json")
        if tokenizer_backend == "json":
            self._tokenizer = load_electra_tokenizer(tok_dir)
        elif tokenizer_backend == "vocab_txt":
            self._tokenizer = load_electra_tokenizer_vocab_txt(tok_dir)
        else:
            raise ValueError("tokenizer_backend must be 'json' or 'vocab_txt'")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = providers if providers is not None else ort.get_available_providers()
        self._sess_tok = ort.InferenceSession(
            str(model_dir / "tok.onnx"), sess_options=so, providers=prov
        )
        self._sess_pos = ort.InferenceSession(
            str(model_dir / "pos.onnx"), sess_options=so, providers=prov
        )

    def segment_words(self, sentence: str) -> list[str]:
        feat = cws_features(sentence, self._tokenizer, self._char_map)
        input_ids = feat["input_ids"]
        attention_mask = (input_ids != 0).astype(np.int64)
        logits = self._sess_tok.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )[0]
        return cws_logits_to_words(logits, self._tag_vocab_tok, feat)

    def tag_pos(self, words: Sequence[str]) -> list[str]:
        if not words:
            return []
        feat = pos_features(
            list(words),
            self._tokenizer,
            self._char_map,
            span_inner_pad=self._pos_span_inner,
        )
        input_ids = feat["input_ids"]
        attention_mask = (input_ids != 0).astype(np.int64)
        span = feat["token_span"]
        logits = self._sess_pos.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask, "token_span": span},
        )[0]
        return pos_logits_to_tags(logits, self._tag_vocab_pos, feat["n_words"])

    def segment_and_tag(self, sentence: str) -> ChineseSentenceAnnotation:
        words = self.segment_words(sentence)
        pos_tags = self.tag_pos(words)
        if len(pos_tags) != len(words):
            raise RuntimeError("POS length mismatch.")
        toks = tuple(ChineseToken(w, p) for w, p in zip(words, pos_tags, strict=True))
        return ChineseSentenceAnnotation(sentence=sentence, tokens=toks)


def _iter_nonempty_lines(f: TextIO) -> Iterable[str]:
    for line in f:
        s = line.rstrip("\n\r")
        if s.strip():
            yield s


def _verify(model_dir: Path, wiki_path: Path, n_lines: int, *, tokenizer_backend: str = "json") -> None:
    try:
        import hanlp
        import hanlp.pretrained.pos as ppos
        import hanlp.pretrained.tok as ptok
    except ImportError as e:
        raise SystemExit(
            "Verification needs HanLP + PyTorch. Install with: pip install hanlp torch\n"
            f"Import error: {e}"
        ) from e

    tok = hanlp.load(ptok.CTB9_TOK_ELECTRA_SMALL, devices=-1)
    pos = hanlp.load(ppos.CTB9_POS_ELECTRA_SMALL, devices=-1)
    onnx_p = HanlpCtb9OnnxWsPos(model_dir, tokenizer_backend=tokenizer_backend)

    lines: list[str] = []
    with wiki_path.open(encoding="utf-8") as f:
        for s in _iter_nonempty_lines(f):
            lines.append(s)
            if len(lines) >= n_lines:
                break

    mismatches = 0
    for i, s in enumerate(lines):
        ref_w = tok([s])[0]
        ref_p = pos([ref_w])[0]
        ref_pairs = list(zip(ref_w, ref_p, strict=True))

        ann = onnx_p.segment_and_tag(s)
        got_pairs = [(t.word, t.pos) for t in ann.tokens]

        if ref_pairs != got_pairs:
            mismatches += 1
            if mismatches <= 5:
                print(f"--- mismatch line {i + 1} ---", file=sys.stderr)
                print(f"sentence: {s[:120]!r}…", file=sys.stderr)
                print(f"ref: {ref_pairs[:20]}…", file=sys.stderr)
                print(f"onnx: {got_pairs[:20]}…", file=sys.stderr)

    if mismatches:
        raise SystemExit(
            f"Verification failed: {mismatches}/{len(lines)} lines differ from HanLP PyTorch (showing up to 5 above)."
        )
    print(f"OK: ONNX matches HanLP on all {len(lines)} line(s).")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR)
    p.add_argument(
        "--verify",
        action="store_true",
        help="Compare to HanLP PyTorch on wiki lines (needs hanlp + torch).",
    )
    p.add_argument(
        "--verify-path",
        type=Path,
        default=_REPO_ROOT / "data" / "zh_hans" / "wiki-text.txt",
    )
    p.add_argument("--verify-lines", type=int, default=100)
    p.add_argument(
        "--tokenizer",
        choices=("json", "vocab_txt"),
        default="json",
        help="Load tokenizer.json (default) or from-scratch BERT on vocab.txt.",
    )
    p.add_argument("-i", "--input", type=Path, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args(argv)

    if args.verify:
        _verify(args.model_dir, args.verify_path, args.verify_lines, tokenizer_backend=args.tokenizer)
        return

    pipe = HanlpCtb9OnnxWsPos(args.model_dir, tokenizer_backend=args.tokenizer)

    in_f: TextIO = sys.stdin if args.input is None else args.input.open(encoding="utf-8")
    out_f: TextIO = sys.stdout if args.output is None else args.output.open("w", encoding="utf-8")
    try:
        for line in _iter_nonempty_lines(in_f):
            ann = pipe.segment_and_tag(line)
            out_f.write(json.dumps(ann.to_json_obj(), ensure_ascii=False) + "\n")
    finally:
        if args.input is not None:
            in_f.close()
        if args.output is not None:
            out_f.close()


if __name__ == "__main__":
    main()
