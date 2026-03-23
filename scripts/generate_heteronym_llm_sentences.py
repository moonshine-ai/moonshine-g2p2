#!/usr/bin/env python3
"""
Generate Wikipedia-style example sentences for heteronym training via the Claude API.

For each ambiguous homograph taken from ``dict_filtered_heteronyms.tsv``-style TSV
(word, IPA; only orthographies with **multiple distinct IPA** rows) or optionally a
LibriG2P JSON / plain word list,
asks the model to emit one semi-formal encyclopedic sentence per item so downstream
pipelines (e.g. ``build_heteronym_espeak_dataset.py``) can mine heteronym spans.

Requires ``ANTHROPIC_API_KEY`` in the environment. Install::

    pip install anthropic

Example::

    python scripts/generate_heteronym_llm_sentences.py

    python scripts/generate_heteronym_llm_sentences.py \\
        --homograph-json data/en_us/heteronym-training/homograph_train.json

    python scripts/generate_heteronym_llm_sentences.py \\
        --homographs-file my_words.txt --out data/en_us/llm-text.txt --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_MODEL = "claude-sonnet-4-20250514"


def _unique_homographs_from_json(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        blob: dict[str, dict[str, object]] = json.load(f)
    seen: set[str] = set()
    out: list[str] = []
    for rec in blob.values():
        if not isinstance(rec, dict):
            continue
        h = rec.get("homograph")
        if not isinstance(h, str) or not h.strip():
            continue
        key = h.strip()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _homographs_from_dict_tsv(path: Path) -> list[str]:
    """Orthography tokens with at least two distinct IPA strings (heteronyms in-file).

    Assumes rows are sorted by orthography so all entries for one word are contiguous.
    """
    out: list[str] = []
    cur_word: str | None = None
    ipas: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            word = parts[0].strip()
            if not word:
                continue
            ipa = parts[1].strip() if len(parts) > 1 else ""

            if cur_word != word:
                if cur_word is not None and len(ipas) > 1:
                    out.append(cur_word)
                cur_word = word
                ipas = set()
            ipas.add(ipa)

        if cur_word is not None and len(ipas) > 1:
            out.append(cur_word)

    return out


def _homographs_from_lines(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _build_user_prompt(
    words: list[str],
    *,
    sentences_per_homograph: int,
) -> str:
    word_block = "\n".join(words)
    n = sentences_per_homograph
    per = (
        f"For each word in the list below, in list order, write exactly {n} distinct "
        f"sentences (each containing that word), one sentence per line, before moving "
        f"to the next word — {len(words) * n} lines total."
        if n > 1
        else "Write exactly one sentence per word below, in the same order as the list."
    )
    return f"""You are preparing plain-text training data for English TTS and G2P research.

{per}

Style requirements:
- Semi-formal, neutral, encyclopedia / Wikipedia-like tone (not chatty, not marketing).
- American English spelling and punctuation conventions.
- Each sentence must be a single standalone line (no bullets, no numbering prefix).
- The target homograph must appear as that exact word token (case may match natural use, e.g. sentence-initial capitalization is fine). Do not only hide it inside a longer unrelated word.
- Keep each sentence under about 220 characters when possible.
- Do not mention these instructions, JSON, or "homograph".

Words (in order):
{word_block}

Output: one sentence per line, nothing else."""


def _extract_text_blocks(content: object) -> str:
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(parts).strip()


def _parse_response_lines(text: str, *, expected: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^[\d]+[\).\s]+", "", s)
        s = s.lstrip("-*•").strip()
        if s.startswith("`") and s.endswith("`"):
            s = s[1:-1].strip()
        lines.append(s)
    if len(lines) != expected:
        logger.warning(
            "expected %d lines from model, got %d (continuing with what we have)",
            expected,
            len(lines),
        )
    return lines


def _call_claude(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit(
            "The 'anthropic' package is required. Install with: pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text_blocks(msg.content)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dict-path",
        type=Path,
        default=Path("data/en_us/dict_filtered_heteronyms.tsv"),
        help="word<TAB>ipa TSV sorted by word; only words with 2+ distinct IPA rows",
    )
    p.add_argument(
        "--homograph-json",
        type=Path,
        default=None,
        help="if set, use LibriG2P-style JSON instead of --dict-path",
    )
    p.add_argument(
        "--homographs-file",
        type=Path,
        default=None,
        help="if set, one homograph per line (# comments); overrides --dict-path and JSON",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/en_us/llm-text.txt"),
        help="output path (one sentence per line)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL),
        help="Claude model id (default: env ANTHROPIC_MODEL or %s)" % _DEFAULT_MODEL,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="homographs per API request",
    )
    p.add_argument(
        "--sentences-per-homograph",
        type=int,
        default=1,
        help="distinct sentences generated per homograph in each batch",
    )
    p.add_argument(
        "--max-homographs",
        type=int,
        default=0,
        help="if > 0, only process the first N homographs (after ordering)",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="pause between API calls (rate limiting)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="per-request max output tokens",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print batches and exit without calling the API",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="append to --out instead of overwriting",
    )
    return p.parse_args(argv)


def _batched(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)

    if args.homographs_file is not None:
        hf = args.homographs_file.resolve()
        if not hf.is_file():
            raise SystemExit(f"Missing --homographs-file: {hf}")
        homographs = _homographs_from_lines(hf)
    elif args.homograph_json is not None:
        jp = args.homograph_json.resolve()
        if not jp.is_file():
            raise SystemExit(f"Missing --homograph-json: {jp}")
        homographs = _unique_homographs_from_json(jp)
    else:
        dp = args.dict_path.resolve()
        if not dp.is_file():
            raise SystemExit(f"Missing --dict-path: {dp}")
        homographs = _homographs_from_dict_tsv(dp)

    homographs = sorted(homographs, key=str.lower)
    if args.max_homographs > 0:
        homographs = homographs[: args.max_homographs]

    if not homographs:
        raise SystemExit("No homographs to process.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY in the environment.")

    out_path: Path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    n_batches = (len(homographs) + args.batch_size - 1) // args.batch_size
    logger.info(
        "%d unique homographs → %d batches (batch_size=%d, sentences_per=%d)",
        len(homographs),
        n_batches,
        args.batch_size,
        args.sentences_per_homograph,
    )

    if args.dry_run:
        for bi, batch in enumerate(_batched(homographs, args.batch_size), start=1):
            logger.info("dry-run batch %d: %s", bi, ", ".join(batch[:5]) + ("..." if len(batch) > 5 else ""))
        return

    total_lines = 0
    with out_path.open(mode, encoding="utf-8") as out_f:
        for bi, batch in enumerate(_batched(homographs, args.batch_size), start=1):
            expected = len(batch) * args.sentences_per_homograph
            prompt = _build_user_prompt(
                batch, sentences_per_homograph=args.sentences_per_homograph
            )
            logger.info("API batch %d / %d (%d homographs)...", bi, n_batches, len(batch))
            text = _call_claude(
                api_key=api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
            lines = _parse_response_lines(text, expected=expected)
            for ln in lines:
                out_f.write(ln + "\n")
            total_lines += len(lines)
            if args.sleep_seconds > 0 and bi < n_batches:
                time.sleep(args.sleep_seconds)

    logger.info("wrote %d lines to %s", total_lines, out_path)


if __name__ == "__main__":
    main()
