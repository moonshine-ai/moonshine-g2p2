#!/usr/bin/env python3
"""
Simplified-Chinese **word segmentation** and **part-of-speech tagging** with
`HanLP <https://github.com/hankcs/HanLP>`_ using a **small** pretrained pair:
ELECTRA-small tokenization (CTB9) + matching POS tagger.

Default checkpoints::

    hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL
    hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL

They are trained on Chinese Treebank–style data and work well on everyday
Simplified Chinese prose (HanLP also documents larger / PKU / multilingual
models in ``hanlp.pretrained``).

Output: JSON Lines with ``sentence`` and ``tokens`` as ``{ "word", "pos" }``
(CTB-style tags such as ``VV``, ``NN``, ``PU`` for punctuation).

Install::

    pip install -U hanlp

**License:** HanLP **code** is Apache-2.0; bundled **pretrained models** are
typically **CC BY-NC-SA 4.0** unless a model page says otherwise—check the
`HanLP README <https://github.com/hankcs/HanLP>`_ before commercial use.

Examples::

    echo '他叫汤姆去拿外衣。' | python scripts/chinese_hanlp_ws_pos.py

    python scripts/chinese_hanlp_ws_pos.py -i data/zh_hans/wiki-text.txt -o tagged.jsonl \\
        --device cuda:0 --chunk-lines 32
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO


@dataclass(frozen=True)
class ChineseToken:
    word: str
    pos: str


@dataclass(frozen=True)
class ChineseSentenceAnnotation:
    sentence: str
    tokens: tuple[ChineseToken, ...]

    def to_json_obj(self) -> dict[str, object]:
        return {
            "sentence": self.sentence,
            "tokens": [asdict(t) for t in self.tokens],
        }


def _resolve_preset(name: str) -> tuple[str, str]:
    import hanlp.pretrained.tok as ptok
    import hanlp.pretrained.pos as ppos

    if name == "electra-small":
        return ptok.CTB9_TOK_ELECTRA_SMALL, ppos.CTB9_POS_ELECTRA_SMALL
    raise ValueError(f"Unknown preset {name!r}. Use: electra-small")


def _parse_devices(s: str | None) -> Any:
    """HanLP ``devices``: ``None`` = default (GPU if available), ``-1`` = CPU."""
    if s is None or s.lower() in ("auto", "default"):
        return None
    if s.lower() == "cpu":
        return -1
    m = re.fullmatch(r"cuda(?::(\d+))?", s.lower())
    if m or s.lower() in ("gpu", "cuda"):
        idx = 0 if s.lower() in ("gpu", "cuda") else int(m.group(1))
        return idx
    raise argparse.ArgumentTypeError(f"Expected cpu, cuda, cuda:N, auto; got {s!r}")


def load_tok_pos(
    tok_ref: str,
    pos_ref: str,
    *,
    devices: Any = None,
) -> tuple[Any, Any]:
    import hanlp

    tok = hanlp.load(tok_ref, devices=devices)
    pos = hanlp.load(pos_ref, devices=devices)
    return tok, pos


def segment_and_tag(sentences: list[str], tok: Any, pos: Any) -> list[ChineseSentenceAnnotation]:
    if not sentences:
        return []

    words_batch: list[list[str]] = tok(sentences)
    tags_batch: list[list[str]] = pos(words_batch)

    if len(words_batch) != len(sentences) or len(tags_batch) != len(sentences):
        raise RuntimeError("HanLP returned unexpected batch length.")

    out: list[ChineseSentenceAnnotation] = []
    for raw, words, tags in zip(sentences, words_batch, tags_batch, strict=True):
        if len(words) != len(tags):
            raise RuntimeError(
                f"Token/POS length mismatch ({len(words)} vs {len(tags)}) for: {raw[:80]!r}…"
            )
        toks = tuple(ChineseToken(word=w, pos=p) for w, p in zip(words, tags, strict=True))
        out.append(ChineseSentenceAnnotation(sentence=raw, tokens=toks))
    return out


def _iter_nonempty_lines(f: TextIO) -> Iterable[str]:
    for line in f:
        s = line.rstrip("\n\r")
        if s.strip():
            yield s


def _write_jsonl(rows: Iterable[ChineseSentenceAnnotation], out: TextIO) -> None:
    for ann in rows:
        out.write(json.dumps(ann.to_json_obj(), ensure_ascii=False) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="UTF-8 input; one sentence per line. Default: stdin.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="JSONL output. Default: stdout.",
    )
    p.add_argument(
        "--preset",
        default="electra-small",
        help='Model pair preset (default: electra-small → CTB9 ELECTRA small tok+pos).',
    )
    p.add_argument(
        "--tok",
        default=None,
        metavar="REF",
        help="Override tokenizer: HanLP pretrained key or URL/path (see hanlp.pretrained.tok).",
    )
    p.add_argument(
        "--pos",
        default=None,
        metavar="REF",
        help="Override POS tagger: HanLP pretrained key or URL/path (see hanlp.pretrained.pos).",
    )
    p.add_argument(
        "--device",
        type=_parse_devices,
        default=None,
        help='Torch device for HanLP: "cpu", "cuda", "cuda:0", or "auto" (default).',
    )
    p.add_argument(
        "--chunk-lines",
        type=int,
        default=32,
        metavar="N",
        help="Run inference in batches of N non-empty lines.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.tok is not None and args.pos is not None:
        tok_ref, pos_ref = args.tok, args.pos
    elif args.tok is None and args.pos is None:
        tok_ref, pos_ref = _resolve_preset(args.preset)
    else:
        raise SystemExit("Provide both --tok and --pos, or neither (to use --preset).")

    tok, pos = load_tok_pos(tok_ref, pos_ref, devices=args.device)

    in_f: TextIO
    if args.input is None:
        in_f = sys.stdin
    else:
        in_f = args.input.open(encoding="utf-8")

    out_f: TextIO
    if args.output is None:
        out_f = sys.stdout
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_f = args.output.open("w", encoding="utf-8")

    try:
        buf: list[str] = []
        for line in _iter_nonempty_lines(in_f):
            buf.append(line)
            if len(buf) >= args.chunk_lines:
                _write_jsonl(segment_and_tag(buf, tok, pos), out_f)
                buf.clear()
        if buf:
            _write_jsonl(segment_and_tag(buf, tok, pos), out_f)
    finally:
        if args.input is not None:
            in_f.close()
        if args.output is not None:
            out_f.close()


if __name__ == "__main__":
    main()
