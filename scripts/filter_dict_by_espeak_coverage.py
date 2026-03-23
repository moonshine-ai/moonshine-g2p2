#!/usr/bin/env python3
"""
Prune ``dict.tsv`` pronunciation rows using eSpeak NG coverage on a sentence corpus.

Scans ``data/en_us/input_text.txt`` (one sentence per line). For every token that has
multiple IPA entries in the dictionary, recovers eSpeak's IPA in context (same
remove-token diff as ``heteronym.espeak_heteronyms``) and maps it to a CMUdict
alternative via ``match_dictionary_alternative``. Any dictionary IPA row for a
multi-pronunciation word that never matched in the corpus is dropped.

If eSpeak never successfully aligned to *any* alternative for a multi-pron word
across the corpus, all of that word's rows are kept (insufficient signal to prune).

With ``--workers 1`` (default), while scanning lines in file order, any homograph key
that appears as CMU-ambiguous in at least ``--heteronym-mono-skip-after`` sentences
(default 100) while exactly one dictionary alternative has ever matched eSpeak is then
ignored for further heteronym detection and eSpeak extraction (dominant-reading
shortcut). Use ``--heteronym-mono-skip-after 0`` to disable. This shortcut is not
applied when ``--workers`` is greater than 1.

Homograph keys from ``--ignore-homographs-file`` (default path under
``heteronym-training/``) and ``--ignore-homograph-keys`` are **pinned** to the
**first** dictionary IPA for that key (same row order as ``dict.tsv``): that IPA is
inserted into the eSpeak “seen” map up front and the key is added to the same
dynamic-ignore set used after ``--heteronym-mono-skip-after`` fires, so heteronym
detection skips it and the writer keeps only that pronunciation. Keys not in the
dict or with a single pronunciation are unchanged. If the default file is missing,
no keys are loaded from it. Use a non-existent ``--ignore-homographs-file`` path to
disable file pins only.

Single-pronunciation words are always kept unchanged.

Requires ``espeak-phonemizer`` and system ``libespeak-ng`` (see
``scripts/build_heteronym_espeak_dataset.py``).

Example::

    python scripts/filter_dict_by_espeak_coverage.py \\
        --dict-path data/en_us/dict.tsv \\
        --input-text data/en_us/input_text.txt \\
        --out data/en_us/dict_filtered_heteronyms.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cmudict_ipa import CmudictIpa, normalize_word_for_lookup

from heteronym.espeak_heteronyms import (
    EspeakPhonemizer,
    extract_examples_for_sentence,
    iter_heteronym_spans_cmudict,
    sentence_has_ambiguous_heteronym,
)

logger = logging.getLogger(__name__)


def _parse_ignore_keys(s: str) -> frozenset[str]:
    if not s.strip():
        return frozenset()
    return frozenset(x.strip().lower() for x in s.split(",") if x.strip())


def _load_ignore_homographs_file(path: Path) -> frozenset[str]:
    """One CMUdict key per line (lowercased); empty lines and ``#`` comments skipped."""
    if not path.is_file():
        return frozenset()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys.add(line.lower())
    return frozenset(keys)


def _build_word_to_ipas(cmu: CmudictIpa) -> dict[str, list[str]]:
    d: dict[str, list[str]] = {}
    for w, ipa in cmu.iter_pronunciation_rows():
        d.setdefault(w, []).append(ipa)
    return d


def _manual_pin_seen_and_skip(
    word_ipas: dict[str, list[str]],
    manual_keys: frozenset[str],
) -> tuple[dict[str, set[str]], frozenset[str]]:
    """
    For each manual key with 2+ CMU readings, seed seen with the first IPA in file
    order and mark the key for immediate dynamic ignore (same effect as mono-skip).
    """
    seen_seed: dict[str, set[str]] = {}
    skip_keys: set[str] = set()
    for k in manual_keys:
        alts = word_ipas.get(k)
        if not alts or len(alts) < 2:
            continue
        seen_seed[k] = {alts[0]}
        skip_keys.add(k)
    return seen_seed, frozenset(skip_keys)


def _merge_seen(dst: dict[str, set[str]], src: dict[str, set[str]]) -> None:
    for k, v in src.items():
        dst[k].update(v)


def _collect_seen_for_lines(
    lines: list[str],
    *,
    cmudict: CmudictIpa,
    phonemizer: EspeakPhonemizer,
    voice: str,
    progress_desc: str | None = None,
    seen_seed: dict[str, set[str]] | None = None,
    dynamic_ignore_seed: frozenset[str] = frozenset(),
    heteronym_mono_skip_after: int = 0,
) -> tuple[dict[str, set[str]], frozenset[str]]:
    """
    Scan *lines* in order. *seen_seed* / *dynamic_ignore_seed* prepopulate the same
    structures used after eSpeak alignment (manual “pin to first pronunciation”).

    If *heteronym_mono_skip_after* > 0, homograph keys that appear in at least that
    many sentences (as CMU multi-pron tokens) while exactly one dictionary alternative
    has ever matched eSpeak are added to *dynamic_ignore* so later lines skip them.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    if seen_seed:
        for k, vs in seen_seed.items():
            seen[k].update(vs)
    dynamic_ignore: set[str] = set(dynamic_ignore_seed)
    ambiguous_hits: dict[str, int] = defaultdict(int)
    it: Iterable[str] = lines
    if progress_desc:
        it = tqdm(lines, desc=progress_desc, unit="line")
    for text in it:
        if not text.strip():
            continue
        ignore = frozenset(dynamic_ignore)
        if not sentence_has_ambiguous_heteronym(
            text,
            cmudict=cmudict,
            max_candidates=None,
            ignore_keys=ignore,
        ):
            continue
        keys_this_sentence: set[str] = set()
        for m, _alts in iter_heteronym_spans_cmudict(
            text,
            cmudict=cmudict,
            max_candidates=None,
            ignore_keys=ignore,
        ):
            k = normalize_word_for_lookup(m.group())
            if k:
                keys_this_sentence.add(k)
        for ex in extract_examples_for_sentence(
            text,
            cmudict=cmudict,
            phonemizer=phonemizer,
            voice=voice,
            max_candidates=None,
            ignore_keys=ignore,
        ):
            key = normalize_word_for_lookup(ex.homograph)
            if key:
                seen[key].add(ex.homograph_wordid)
        if heteronym_mono_skip_after > 0:
            for k in keys_this_sentence:
                ambiguous_hits[k] += 1
                if (
                    ambiguous_hits[k] >= heteronym_mono_skip_after
                    and len(seen[k]) == 1
                ):
                    dynamic_ignore.add(k)
    return dict(seen), frozenset(dynamic_ignore)


_worker_cmudict: CmudictIpa | None = None
_worker_phonemizer: EspeakPhonemizer | None = None
_worker_voice: str = "en-us"
_worker_dynamic_ignore_seed: frozenset[str] = frozenset()


def _mp_init(dict_path: str, voice: str, dynamic_ignore_seed_tuple: tuple[str, ...]) -> None:
    global _worker_cmudict, _worker_phonemizer, _worker_voice, _worker_dynamic_ignore_seed
    _worker_cmudict = CmudictIpa(dict_path)
    _worker_phonemizer = EspeakPhonemizer(default_voice=voice)
    _worker_voice = voice
    _worker_dynamic_ignore_seed = frozenset(dynamic_ignore_seed_tuple)


def _mp_worker(chunk: list[str]) -> dict[str, set[str]]:
    assert _worker_cmudict is not None and _worker_phonemizer is not None
    seen, _dynamic = _collect_seen_for_lines(
        chunk,
        cmudict=_worker_cmudict,
        phonemizer=_worker_phonemizer,
        voice=_worker_voice,
        seen_seed=None,
        dynamic_ignore_seed=_worker_dynamic_ignore_seed,
        heteronym_mono_skip_after=0,
    )
    return seen


def _write_filtered_dict(
    cmu: CmudictIpa,
    word_ipas: dict[str, list[str]],
    seen: dict[str, set[str]],
    out_path: Path,
    *,
    progress_desc: str | None = None,
) -> dict[str, int]:
    stats = {
        "rows_in": 0,
        "rows_out": 0,
        "rows_dropped_multi": 0,
        "multi_pron_words": 0,
        "multi_pron_words_pruned": 0,
    }
    multi_words = {w for w, alts in word_ipas.items() if len(alts) >= 2}
    stats["multi_pron_words"] = len(multi_words)

    out_lines: list[str] = []
    rows = cmu.iter_pronunciation_rows()
    if progress_desc:
        rows = tqdm(rows, desc=progress_desc, unit="row")
    for w, ipa in rows:
        stats["rows_in"] += 1
        alts = word_ipas[w]
        if len(alts) < 2:
            out_lines.append(f"{w}\t{ipa}")
            stats["rows_out"] += 1
            continue
        observed = seen.get(w, frozenset())
        if not observed:
            out_lines.append(f"{w}\t{ipa}")
            stats["rows_out"] += 1
            continue
        if ipa in observed:
            out_lines.append(f"{w}\t{ipa}")
            stats["rows_out"] += 1
        else:
            stats["rows_dropped_multi"] += 1

    pruned_words = {
        w
        for w in multi_words
        if seen.get(w) and any(ipa not in seen[w] for ipa in word_ipas[w])
    }
    stats["multi_pron_words_pruned"] = len(pruned_words)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
        if out_lines:
            f.write("\n")

    return stats


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dict-path",
        type=Path,
        default=Path("data/en_us/dict.tsv"),
        help="source TSV (word<TAB>ipa)",
    )
    p.add_argument(
        "--input-text",
        type=Path,
        default=Path("data/en_us/wiki-text.txt"),
        help="one sentence per line",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/en_us/dict_filtered_heteronyms.tsv"),
        help="filtered TSV output path",
    )
    p.add_argument("--voice", type=str, default="en-us", help="eSpeak voice, e.g. en-us")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel processes (each loads libespeak-ng)",
    )
    p.add_argument(
        "--ignore-homographs-file",
        type=Path,
        default=Path("data/en_us/heteronym-training/ignore_homographs.txt"),
        help=(
            "optional text file: one lowercase CMUdict key per line; each multi-pron "
            "key is pinned to its first dict IPA and skipped like mono-skip (missing "
            "file = no keys from file)"
        ),
    )
    p.add_argument(
        "--ignore-homograph-keys",
        type=str,
        default="",
        help=(
            "comma-separated lowercase CMUdict keys: same pinning as "
            "--ignore-homographs-file (merged with file keys)"
        ),
    )
    p.add_argument(
        "--heteronym-mono-skip-after",
        type=int,
        default=100,
        help=(
            "with --workers 1 only: skip further eSpeak work for a homograph key after "
            "it appears as CMU-ambiguous in at least this many sentences while exactly "
            "one alternative has matched eSpeak (0 disables)"
        ),
    )
    p.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="if > 0, only process the first N non-empty lines (debug)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    dict_path = args.dict_path.resolve()
    input_path = args.input_text.resolve()
    out_path = args.out.resolve()

    if not dict_path.is_file():
        raise SystemExit(f"Missing dictionary TSV: {dict_path}")
    if not input_path.is_file():
        raise SystemExit(
            f"Missing input text file: {input_path}\n"
            "Create it with one sentence per line (e.g. cache from wikitext or your corpus)."
        )

    lines = [
        ln.strip()
        for ln in input_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if args.max_lines > 0:
        lines = lines[: args.max_lines]

    logger.info("loaded %d sentences from %s", len(lines), input_path)

    ignore_file = args.ignore_homographs_file.resolve()
    from_file = _load_ignore_homographs_file(ignore_file)
    from_cli = _parse_ignore_keys(args.ignore_homograph_keys)
    manual_pin_keys = from_file | from_cli
    if from_file:
        logger.info(
            "loaded %d manual homograph key(s) from %s",
            len(from_file),
            ignore_file,
        )
    if from_cli:
        logger.info(
            "plus %d key(s) from --ignore-homograph-keys",
            len(from_cli),
        )
    mono_skip = args.heteronym_mono_skip_after
    if args.workers > 1 and mono_skip > 0:
        logger.warning(
            "--heteronym-mono-skip-after is ignored with multiple workers "
            "(requires sequential scan); using 0"
        )
        mono_skip = 0

    cmu = CmudictIpa(dict_path)
    word_ipas = _build_word_to_ipas(cmu)
    seen_seed, dynamic_ignore_seed = _manual_pin_seen_and_skip(
        word_ipas, manual_pin_keys
    )
    if seen_seed:
        logger.info(
            "pinned %d multi-pron key(s) to first dictionary IPA (skip heteronym scan)",
            len(seen_seed),
        )

    seen: dict[str, set[str]] = defaultdict(set)
    mono_skipped_keys: frozenset[str] = frozenset()

    if args.workers <= 1:
        try:
            phonemizer = EspeakPhonemizer(default_voice=args.voice)
        except OSError as e:
            raise SystemExit(
                "Could not load libespeak-ng (needed by espeak-phonemizer). "
                "Install the eSpeak NG library for your OS.\n"
                f"Original error: {e}"
            ) from e
        partial, mono_skipped_keys = _collect_seen_for_lines(
            lines,
            cmudict=cmu,
            phonemizer=phonemizer,
            voice=args.voice,
            progress_desc="Scanning sentences",
            seen_seed=dict(seen_seed),
            dynamic_ignore_seed=dynamic_ignore_seed,
            heteronym_mono_skip_after=mono_skip,
        )
        _merge_seen(seen, partial)
    else:
        n = len(lines)
        if n == 0:
            chunks: list[list[str]] = []
        else:
            w = args.workers
            step = max(1, (n + w - 1) // w)
            chunks = [lines[i : i + step] for i in range(0, n, step)]
        skip_tuple = tuple(sorted(dynamic_ignore_seed))
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_mp_init,
                initargs=(str(dict_path), args.voice, skip_tuple),
            ) as ex:
                futures = [ex.submit(_mp_worker, ch) for ch in chunks if ch]
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Worker chunks",
                    unit="chunk",
                ):
                    try:
                        part = fut.result()
                    except Exception as e:  # pragma: no cover
                        logger.warning("worker failed: %s", e)
                        continue
                    _merge_seen(seen, part)
        except OSError as e:
            raise SystemExit(
                "Could not start workers (libespeak-ng / espeak-phonemizer). "
                f"Original error: {e}"
            ) from e
        _merge_seen(seen, seen_seed)

    stats = _write_filtered_dict(
        cmu,
        word_ipas,
        dict(seen),
        out_path,
        progress_desc="Writing filtered dictionary",
    )
    stats["sentences_scanned"] = len(lines)
    stats["heteronym_keys_with_any_espeak_hit"] = sum(1 for k in seen if seen[k])
    stats["heteronym_mono_skipped_keys"] = len(mono_skipped_keys)

    logger.info(
        "wrote %s (%d rows, dropped %d multi-pron rows). "
        "multi-pron words touched=%d, words with at least one dropped alt=%d",
        out_path,
        stats["rows_out"],
        stats["rows_dropped_multi"],
        stats["multi_pron_words"],
        stats["multi_pron_words_pruned"],
    )
    for k, v in sorted(stats.items()):
        logger.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()
