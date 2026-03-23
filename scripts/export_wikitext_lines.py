#!/usr/bin/env python3
"""
Export Wikipedia- / Wikitext-style prose as one sentence per line (``wiki-text.txt``).

* **English (``en_us``)** — same source as before: HuggingFace ``wikitext`` /
  ``wikitext-103-raw-v1`` (streaming ``train``), with the same rough sentence
  chunking as ``scripts/build_heteronym_espeak_dataset.py``.
* **Other locales** — ``wikimedia/wikipedia`` dump ``20231101.<lang>`` (plain
  article ``text``), split into sentences with the same length cap. Spanish
  (``es_es`` / ``es_mx``), Portuguese (``pt_br`` / ``pt_pt``), and Chinese
  (``zh_hans`` / ``zh_hant``) share one Wikipedia language edition each, so one
  streamed pass writes identical lines to the paired folders.

Default output path per locale::

    data/<locale>/wiki-text.txt

Requires::

    pip install datasets

Examples::

    # English only (fast; matches older single-file workflows)
    python scripts/export_wikitext_lines.py --only en_us

    # All locales under data/ (slow; many large streams)
    python scripts/export_wikitext_lines.py --only all

    # Single custom path (English Wikitext-103 only)
    python scripts/export_wikitext_lines.py --out data/en_us/wiki-text.txt
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# Split after sentence punctuation; optional spaces (Latin) or none (CJK).
_SENT_BOUND = re.compile(r"(?<=[.!?。？！؟])\s*(?=\S)")

_WIKIPEDIA_DUMP = "20231101"


@dataclass(frozen=True)
class WikiExportJob:
    """One HF stream, written to one or more ``data/<folder>/wiki-text.txt`` files."""

    folders: tuple[str, ...]
    backend: Literal["wikitext", "wikipedia"]
    config: str


# Order: single-folder langs, then shared Wikipedia editions (see module docstring).
_EXPORT_JOBS: tuple[WikiExportJob, ...] = (
    WikiExportJob(("en_us",), "wikitext", "wikitext-103-raw-v1"),
    WikiExportJob(("ar",), "wikipedia", f"{_WIKIPEDIA_DUMP}.ar"),
    WikiExportJob(("de",), "wikipedia", f"{_WIKIPEDIA_DUMP}.de"),
    WikiExportJob(("fa",), "wikipedia", f"{_WIKIPEDIA_DUMP}.fa"),
    WikiExportJob(("fr",), "wikipedia", f"{_WIKIPEDIA_DUMP}.fr"),
    WikiExportJob(("id",), "wikipedia", f"{_WIKIPEDIA_DUMP}.id"),
    WikiExportJob(("it",), "wikipedia", f"{_WIKIPEDIA_DUMP}.it"),
    WikiExportJob(("ja",), "wikipedia", f"{_WIKIPEDIA_DUMP}.ja"),
    WikiExportJob(("ko",), "wikipedia", f"{_WIKIPEDIA_DUMP}.ko"),
    WikiExportJob(("nl",), "wikipedia", f"{_WIKIPEDIA_DUMP}.nl"),
    WikiExportJob(("ru",), "wikipedia", f"{_WIKIPEDIA_DUMP}.ru"),
    WikiExportJob(("tr",), "wikipedia", f"{_WIKIPEDIA_DUMP}.tr"),
    WikiExportJob(("uk",), "wikipedia", f"{_WIKIPEDIA_DUMP}.uk"),
    WikiExportJob(("vi",), "wikipedia", f"{_WIKIPEDIA_DUMP}.vi"),
    WikiExportJob(("es_es", "es_mx"), "wikipedia", f"{_WIKIPEDIA_DUMP}.es"),
    WikiExportJob(("pt_br", "pt_pt"), "wikipedia", f"{_WIKIPEDIA_DUMP}.pt"),
    WikiExportJob(("zh_hans", "zh_hant"), "wikipedia", f"{_WIKIPEDIA_DUMP}.zh"),
)

_ALL_FOLDERS: frozenset[str] = frozenset(f for job in _EXPORT_JOBS for f in job.folders)


def _rough_sentences(paragraph: str, *, max_chars: int) -> list[str]:
    paragraph = paragraph.strip()
    if not paragraph or paragraph.startswith("="):
        return []
    chunks = _SENT_BOUND.split(paragraph)
    out: list[str] = []
    for c in chunks:
        c = c.strip()
        if not c or len(c) > max_chars:
            continue
        out.append(c)
    if not out and len(paragraph) <= max_chars:
        out.append(paragraph)
    return out


def _article_paragraphs(article: str) -> Iterator[str]:
    """Turn Wikipedia plain text into paragraph-sized strings."""
    for block in re.split(r"\n\s*\n+", article):
        block = block.strip()
        if not block:
            continue
        lines = [ln for ln in block.split("\n") if not ln.strip().startswith("==")]
        if not lines:
            continue
        merged = " ".join(ln.strip() for ln in lines if ln.strip())
        if merged:
            yield merged


def _one_output_line(sentence: str) -> str:
    return " ".join(sentence.replace("\r\n", "\n").replace("\r", "\n").split())


def _load_dataset_streaming(dataset: str, name: str, *, split: str) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "This script needs the `datasets` package. "
            "Install with: pip install datasets\n"
            f"Original error: {e}"
        ) from e
    return iter(load_dataset(dataset, name, split=split, streaming=True))


def iter_wikitext103_sentences(
    *,
    max_sentences: int,
    max_chars: int,
    split: str = "train",
) -> Iterator[str]:
    ds = _load_dataset_streaming("wikitext", "wikitext-103-raw-v1", split=split)
    n = 0
    for row in ds:
        if n >= max_sentences:
            break
        for sent in _rough_sentences(row["text"], max_chars=max_chars):
            if n >= max_sentences:
                break
            yield sent
            n += 1


def iter_wikipedia_sentences(
    *,
    wiki_config: str,
    max_sentences: int,
    max_chars: int,
) -> Iterator[str]:
    ds = _load_dataset_streaming("wikimedia/wikipedia", wiki_config, split="train")
    n = 0
    for row in ds:
        if n >= max_sentences:
            break
        text = row.get("text") or ""
        if not text.strip():
            continue
        for para in _article_paragraphs(text):
            if n >= max_sentences:
                break
            for sent in _rough_sentences(para, max_chars=max_chars):
                if n >= max_sentences:
                    break
                yield sent
                n += 1


def _iter_job_sentences(
    job: WikiExportJob,
    *,
    max_sentences: int,
    max_chars: int,
    wikitext_split: str,
) -> Iterator[str]:
    if job.backend == "wikitext":
        yield from iter_wikitext103_sentences(
            max_sentences=max_sentences,
            max_chars=max_chars,
            split=wikitext_split,
        )
    else:
        yield from iter_wikipedia_sentences(
            wiki_config=job.config,
            max_sentences=max_sentences,
            max_chars=max_chars,
        )


def _parse_only(raw: str) -> frozenset[str]:
    raw = raw.strip()
    if raw.lower() == "all":
        return _ALL_FOLDERS
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    unknown = keys - _ALL_FOLDERS
    if unknown:
        raise SystemExit(
            f"Unknown --only folder(s): {sorted(unknown)}. "
            f"Valid names: all, or comma-separated from {sorted(_ALL_FOLDERS)}"
        )
    return frozenset(keys)


def _jobs_for_folders(folders: frozenset[str]) -> list[WikiExportJob]:
    return [job for job in _EXPORT_JOBS if any(f in folders for f in job.folders)]


def _export_job_to_files(
    job: WikiExportJob,
    out_root: Path,
    *,
    folders_filter: frozenset[str],
    max_sentences: int,
    max_chars: int,
    wikitext_split: str,
) -> None:
    target_folders = tuple(f for f in job.folders if f in folders_filter)
    if not target_folders:
        return

    paths = [out_root / folder / "wiki-text.txt" for folder in target_folders]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with ExitStack() as stack:
        files = [stack.enter_context(p.open("w", encoding="utf-8")) for p in paths]
        for sent in _iter_job_sentences(
            job,
            max_sentences=max_sentences,
            max_chars=max_chars,
            wikitext_split=wikitext_split,
        ):
            line = _one_output_line(sent)
            if not line:
                continue
            for fh in files:
                fh.write(line)
                fh.write("\n")
            n += 1
            if n % 10_000 == 0:
                logger.info(
                    "wrote %d lines -> %s",
                    n,
                    ", ".join(str(p) for p in paths),
                )

    logger.info(
        "wrote %d lines to %s",
        n,
        ", ".join(str(p) for p in paths),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-root",
        type=Path,
        default=_REPO_ROOT / "data",
        help="Repo data directory; each locale is <out-root>/<folder>/wiki-text.txt",
    )
    p.add_argument(
        "--only",
        type=str,
        default="en_us",
        help="Comma-separated data/ folder names (see multilingual lexicon keys), or 'all'.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "If set, ignore --only/--out-root: stream English Wikitext-103 only to this path "
            "(single-file mode)."
        ),
    )
    p.add_argument(
        "--wikitext-split",
        type=str,
        default="train",
        help="datasets split for wikitext-103-raw-v1 (English only)",
    )
    p.add_argument(
        "--max-sentences",
        type=int,
        default=10_000_000,
        help="cap emitted sentences per export job (same default as build_heteronym_espeak_dataset)",
    )
    p.add_argument(
        "--max-sentence-chars",
        type=int,
        default=384,
        help="skip longer sentences (same default as build_heteronym_espeak_dataset)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    out_root = args.out_root.resolve()

    if args.out is not None:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for sent in iter_wikitext103_sentences(
                max_sentences=args.max_sentences,
                max_chars=args.max_sentence_chars,
                split=args.wikitext_split,
            ):
                line = _one_output_line(sent)
                if not line:
                    continue
                f.write(line)
                f.write("\n")
                n += 1
                if n % 10_000 == 0:
                    logger.info("wrote %d lines", n)
        logger.info("wrote %d lines to %s", n, out_path)
        return

    folders = _parse_only(args.only)
    jobs = _jobs_for_folders(folders)
    if not jobs:
        raise SystemExit("No export jobs match --only (empty selection).")

    for job in jobs:
        logger.info(
            "export %s via %s %s",
            ",".join(job.folders),
            job.backend,
            job.config,
        )
        _export_job_to_files(
            job,
            out_root,
            folders_filter=folders,
            max_sentences=args.max_sentences,
            max_chars=args.max_sentence_chars,
            wikitext_split=args.wikitext_split,
        )


if __name__ == "__main__":
    main()
