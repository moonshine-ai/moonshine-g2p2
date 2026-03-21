#!/usr/bin/env python3
"""
Download open word→IPA lexicons for many languages, in the same spirit as
``download_cmudict_to_tsv.py`` (one pronunciation per line: ``word<TAB>ipa``).

Primary sources (see project notes / per-language caveats in the table below):

* **ipa-dict** (MIT): https://github.com/open-dict-data/ipa-dict — ``data/*.txt``
* **WikiPron** (Apache-2.0): https://github.com/CUNY-CL/wikipron — ``data/scrape/tsv/*.tsv``

ipa-dict does not ship Indonesian, Italian, Russian, Turkish, Ukrainian, or
European Portuguese; those are filled from WikiPron. Mandarin (simplified and
traditional) comes from ipa-dict only (``zh_hans.txt``, ``zh_hant.txt``); there
is no ``cmn_*.tsv`` in the current WikiPron tree.

Output layout: each language or country variant is a **direct child** of the
repo ``data/`` directory (by default). Folder names are the lexicon keys
(``ar``, ``de``, ``es_es``, ``es_mx``, ``pt_br``, ``zh_hant``, …)::

    data/<key>/dict.tsv
    data/<key>/source.txt

So Spanish and Portuguese variants sit beside e.g. ``data/en_us/`` rather than
under a shared ``lexicons`` directory. Override the parent with ``--out-root``.

Spanish uses separate Spain (``es_es``) and Latin-American (``es_mx``) folders.
Portuguese uses Brazilian (``pt_br``) and European (``pt_pt``).

These files are compatible with :class:`cmudict_ipa.CmudictIpa` TSV mode only if
lowercasing keys is acceptable (the class lowercases for English). For
case-sensitive orthographies (e.g. German nouns, Arabic) or scripts where
lowercasing is wrong, read the TSV directly or add a loader that preserves
surface forms.

+-------------------+-----------+------------------------------------------+
| Key               | Source    | Notes                                    |
+===================+===========+==========================================+
| ar                | ipa-dict  | MSA-oriented; best with voweled script   |
| de                | ipa-dict  |                                          |
| es_es, es_mx      | ipa-dict  | Castilian vs. Latin-American           |
| fa                | ipa-dict  | Experimental / incomplete (see ipa-dict) |
| fr                | ipa-dict  | ``fr_FR``                                |
| id                | WikiPron  | ``ind_latn_broad``                       |
| it                | WikiPron  | ``ita_latn_broad``                       |
| ja                | ipa-dict  |                                          |
| ko                | ipa-dict  |                                          |
| nl                | ipa-dict  | INT data, CC BY (see ipa-dict README)    |
| pt_br             | ipa-dict  |                                          |
| pt_pt             | WikiPron  | ``por_latn_po_broad``                    |
| ru                | WikiPron  | ``rus_cyrl_narrow`` (narrow in Wiktionary)|
| tr                | WikiPron  | ``tur_latn_broad``                       |
| uk                | WikiPron  | ``ukr_cyrl_narrow``                      |
| vi                | ipa-dict  | ``vi_N`` (Northern orthography)          |
| zh_hans, zh_hant  | ipa-dict  | Polyphony / script coverage caveats      |
+-------------------+-----------+------------------------------------------+
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

# Repo root (parent of ``scripts/``).
_ROOT = Path(__file__).resolve().parent.parent

IPA_DICT_RAW = (
    "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{file}"
)
WIKIPRON_SCRAPE_TSV = (
    "https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/{file}"
)

USER_AGENT = "moonshine-g2p-multilingual-lexicon-fetch/1.0"


@dataclass(frozen=True)
class LexiconSpec:
    key: str
    description: str
    source: Literal["ipa-dict", "wikipron"]
    upstream_file: str
    license_note: str


# Order matches the user’s language list; Spanish + Portuguese get split variants.
LEXICONS: tuple[LexiconSpec, ...] = (
    LexiconSpec(
        "ar",
        "Arabic (MSA-oriented; ipa-dict / Buckwalter pipeline)",
        "ipa-dict",
        "ar.txt",
        "MIT (ipa-dict). Arabic: voweled text strongly preferred.",
    ),
    LexiconSpec(
        "de",
        "German",
        "ipa-dict",
        "de.txt",
        "MIT (ipa-dict).",
    ),
    LexiconSpec(
        "es_es",
        "Spanish (Spain; ipa-dict es_ES)",
        "ipa-dict",
        "es_ES.txt",
        "MIT (ipa-dict).",
    ),
    LexiconSpec(
        "es_mx",
        "Spanish (Latin America; ipa-dict es_MX)",
        "ipa-dict",
        "es_MX.txt",
        "MIT (ipa-dict).",
    ),
    LexiconSpec(
        "fa",
        "Persian / Farsi (experimental in ipa-dict)",
        "ipa-dict",
        "fa.txt",
        "MIT (ipa-dict). Treat as low-confidence; short vowels often unwritten.",
    ),
    LexiconSpec(
        "fr",
        "French (fr_FR)",
        "ipa-dict",
        "fr_FR.txt",
        "MIT (ipa-dict).",
    ),
    LexiconSpec(
        "id",
        "Indonesian (WikiPron broad)",
        "wikipron",
        "ind_latn_broad.tsv",
        "Apache-2.0 (WikiPron scrape).",
    ),
    LexiconSpec(
        "it",
        "Italian (WikiPron broad)",
        "wikipron",
        "ita_latn_broad.tsv",
        "Apache-2.0 (WikiPron scrape).",
    ),
    LexiconSpec(
        "ja",
        "Japanese",
        "ipa-dict",
        "ja.txt",
        "MIT (ipa-dict). Typical pipeline: kanji → kana → IPA.",
    ),
    LexiconSpec(
        "ko",
        "Korean",
        "ipa-dict",
        "ko.txt",
        "MIT (ipa-dict).",
    ),
    LexiconSpec(
        "nl",
        "Dutch (INT / ipa-dict; CC BY — see ipa-dict README)",
        "ipa-dict",
        "nl.txt",
        "CC BY source via ipa-dict (see upstream README).",
    ),
    LexiconSpec(
        "pt_br",
        "Portuguese (Brazil; ipa-dict pt_BR)",
        "ipa-dict",
        "pt_BR.txt",
        "MIT (ipa-dict).",
    ),
    LexiconSpec(
        "pt_pt",
        "Portuguese (Portugal; WikiPron broad)",
        "wikipron",
        "por_latn_po_broad.tsv",
        "Apache-2.0 (WikiPron scrape).",
    ),
    LexiconSpec(
        "ru",
        "Russian (WikiPron narrow — Wiktionary-style for this language)",
        "wikipron",
        "rus_cyrl_narrow.tsv",
        "Apache-2.0 (WikiPron scrape).",
    ),
    LexiconSpec(
        "tr",
        "Turkish (WikiPron broad)",
        "wikipron",
        "tur_latn_broad.tsv",
        "Apache-2.0 (WikiPron scrape).",
    ),
    LexiconSpec(
        "uk",
        "Ukrainian (WikiPron narrow)",
        "wikipron",
        "ukr_cyrl_narrow.tsv",
        "Apache-2.0 (WikiPron scrape).",
    ),
    LexiconSpec(
        "vi",
        "Vietnamese (Northern; ipa-dict vi_N / vPhon-style pipeline)",
        "ipa-dict",
        "vi_N.txt",
        "MIT (ipa-dict). Preserve tone diacritics in orthography.",
    ),
    LexiconSpec(
        "zh_hans",
        "Mandarin (Simplified; ipa-dict)",
        "ipa-dict",
        "zh_hans.txt",
        "MIT (ipa-dict). Polyphony is the main difficulty.",
    ),
    LexiconSpec(
        "zh_hant",
        "Mandarin (Traditional; ipa-dict)",
        "ipa-dict",
        "zh_hant.txt",
        "MIT (ipa-dict). Polyphony is the main difficulty.",
    ),
)

_BY_KEY: dict[str, LexiconSpec] = {s.key: s for s in LEXICONS}


def _lexicon_url(spec: LexiconSpec) -> str:
    if spec.source == "ipa-dict":
        return IPA_DICT_RAW.format(file=spec.upstream_file)
    return WIKIPRON_SCRAPE_TSV.format(file=spec.upstream_file)


def _fetch_text(url: str, *, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _expand_ipa_dict_ipas(cell: str) -> list[str]:
    """ipa-dict uses slash-wrapped IPA; multiple readings are comma-separated."""
    cell = cell.strip()
    if not cell:
        return []
    found = re.findall(r"/([^/]*)/", cell)
    if found:
        return [p for p in found if p]
    return [cell]


def _collapse_wikipron_ipa(ipa: str) -> str:
    """WikiPron broad/narrow files use space-separated IPA tokens; join for one string."""
    return "".join(ipa.split())


def _iter_ipa_dict_rows(text: str) -> Iterator[tuple[str, str]]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            continue
        word, ipa_cell = line.split("\t", 1)
        word = word.strip()
        if not word:
            continue
        for ipa in _expand_ipa_dict_ipas(ipa_cell):
            if ipa:
                yield word, ipa


def _iter_wikipron_rows(text: str) -> Iterator[tuple[str, str]]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" not in line:
            continue
        word, ipa = line.split("\t", 1)
        word = word.strip()
        ipa = _collapse_wikipron_ipa(ipa.strip())
        if word and ipa:
            yield word, ipa


def _rows_for_spec(spec: LexiconSpec, text: str) -> list[tuple[str, str]]:
    if spec.source == "ipa-dict":
        return list(_iter_ipa_dict_rows(text))
    return list(_iter_wikipron_rows(text))


def _write_lexicon(
    out_dir: Path,
    spec: LexiconSpec,
    rows: Iterable[tuple[str, str]],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "dict.tsv"
    url = _lexicon_url(spec)
    n = 0
    with open(tsv_path, "w", encoding="utf-8") as f:
        for word, ipa in rows:
            f.write(f"{word}\t{ipa}\n")
            n += 1
    meta = out_dir / "source.txt"
    with open(meta, "w", encoding="utf-8") as mf:
        mf.write(f"url\t{url}\n")
        mf.write(f"source\t{spec.source}\n")
        mf.write(f"license_note\t{spec.license_note}\n")
        mf.write(f"description\t{spec.description}\n")
    return n


def _parse_only_keys(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    unknown = keys - _BY_KEY.keys()
    if unknown:
        raise SystemExit(
            f"Unknown --only keys: {sorted(unknown)}. Use --list for valid keys."
        )
    return keys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_ROOT / "data",
        help="Parent directory (default: repo data/). Each variant is <out-root>/<key>/dict.tsv.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of lexicon keys (default: all).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-request timeout in seconds (large Arabic/German files need time).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs and exit without downloading.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print catalog (key, description, URL) and exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for spec in LEXICONS:
            print(f"{spec.key}\t{spec.description}\t{_lexicon_url(spec)}")
        return

    only = _parse_only_keys(args.only)
    specs = [s for s in LEXICONS if only is None or s.key in only]

    if args.dry_run:
        for spec in specs:
            print(_lexicon_url(spec))
        return

    ok = 0
    for spec in specs:
        url = _lexicon_url(spec)
        print(f"fetch {spec.key} … {url}", file=sys.stderr)
        try:
            text = _fetch_text(url, timeout=args.timeout)
        except (urllib.error.URLError, OSError) as e:
            print(f"error {spec.key}: {e}", file=sys.stderr)
            continue
        rows = _rows_for_spec(spec, text)
        out_dir = args.out_root / spec.key
        n = _write_lexicon(out_dir, spec, rows)
        print(f"wrote {out_dir / 'dict.tsv'} ({n} rows)", file=sys.stderr)
        ok += 1

    print(f"done: {ok}/{len(specs)} lexicons", file=sys.stderr)
    if ok < len(specs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
