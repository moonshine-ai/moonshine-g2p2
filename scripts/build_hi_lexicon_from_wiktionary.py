#!/usr/bin/env python3
"""
Append Hindi ``word<TAB>ipa`` lines to ``data/hi/dict.tsv`` from English Wiktionary.

Source category (CC BY-SA): https://en.wiktionary.org/wiki/Category:Hindi_terms_with_IPA_pronunciation

Raw wikitext usually contains ``{{hi-IPA}}`` (expanded by Scribunto), not literal ``/…/`` in
``{{IPA|hi|…}}``. This script uses ``action=parse`` and reads the rendered HTML under the
``== Hindi ==`` section, taking the first phonemic IPA in a ``<span class="IPA …">`` (slashes).

Requires network. Be polite: default delay between parse requests, small ``User-Agent`` string.
Shows a ``tqdm`` bar on stderr when the ``tqdm`` package is installed (``pip install tqdm``).

Examples::

    python scripts/build_hi_lexicon_from_wiktionary.py --limit 50 --dry-run
    python scripts/build_hi_lexicon_from_wiktionary.py --out data/hi/dict_wiki.tsv
    python scripts/build_hi_lexicon_from_wiktionary.py --merge data/hi/dict.tsv

Frequency mode (OpenSubtitles-based list, CC BY-SA 4.0 — see ``--freq-url`` default)::

    python scripts/build_hi_lexicon_from_wiktionary.py --freq-url DEFAULT --limit 200 --dry-run
    python scripts/build_hi_lexicon_from_wiktionary.py --freq-file data/hi/hi_full.txt --limit 500 --out data/hi/dict_wiki.tsv
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

API = "https://en.wiktionary.org/w/api.php"
CATEGORY = "Category:Hindi_terms_with_IPA_pronunciation"
UA = "MoonshineG2P/1.0 (https://github.com/; lexicon build script) Python-urllib"

# OpenSubtitles-derived Hindi frequencies (hermitdave/FrequencyWords, CC BY-SA 4.0).
DEFAULT_FREQ_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/hi/hi_full.txt"
)

# Devanagari block (primary keys for this lexicon)
_DEVA_RE = re.compile(r"[\u0900-\u097F]")

# IPA spans in parsed HTML
_IPA_SPAN_RE = re.compile(
    r'<span[^>]*\bclass="[^"]*\bIPA\b[^"]*"[^>]*>([^<]*)</span>',
    re.I | re.DOTALL,
)


def _api_get(params: dict[str, str]) -> dict:
    q = urllib.parse.urlencode(params, safe="|")
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def iter_category_titles(cmcontinue: str | None = None):
    while True:
        p: dict[str, str] = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": CATEGORY,
            "cmlimit": "500",
        }
        if cmcontinue:
            p["cmcontinue"] = cmcontinue
        data = _api_get(p)
        q = data.get("query", {})
        for m in q.get("categorymembers", []):
            if m.get("ns") == 0:
                yield m["title"]
        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break


def parse_page_html(title: str) -> str | None:
    p = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "disabletoc": "1",
    }
    data = _api_get(p)
    err = data.get("error")
    if err:
        return None
    return data.get("parse", {}).get("text", {}).get("*")


def _hindi_section_html(full: str) -> str | None:
    anchor = full.find('id="Hindi"')
    if anchor < 0:
        return None
    start = full.rfind("<h2", 0, anchor + 1)
    if start < 0:
        start = anchor
    h2_end = full.find(">", start)
    if h2_end < 0:
        return None
    nxt = re.search(r"<h2\b", full[h2_end + 1 :])
    if not nxt:
        return full[start:]
    return full[start : h2_end + 1 + nxt.start()]


def first_phonemic_ipa(fragment: str) -> str | None:
    frag = html_lib.unescape(fragment)
    for m in _IPA_SPAN_RE.finditer(frag):
        inner = m.group(1).strip()
        if len(inner) >= 3 and inner.startswith("/") and inner.endswith("/"):
            return inner[1:-1].strip()
    return None


def extract_for_title(title: str) -> tuple[tuple[str, str] | None, str]:
    """Return ((word, ipa) or None, reason) with one parse per call."""
    if not _DEVA_RE.search(title):
        return None, "no_devanagari"
    page_html = parse_page_html(title)
    if not page_html:
        return None, "parse_error"
    block = _hindi_section_html(page_html)
    if not block:
        return None, "no_hindi_section"
    ipa = first_phonemic_ipa(block)
    if not ipa:
        return None, "no_ipa"
    return (title, ipa), "ok"


def fetch_url_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("utf-8")


def iter_top_freq_titles(text: str, max_lines: int) -> list[str]:
    """
    Parse ``word<SPACE>count`` lines (one token per line, highest frequency first).
    Return up to ``max_lines`` **titles** (the word column) in order; skip blanks and
    lines without Devanagari in the token.
    """
    titles: list[str] = []
    for raw in text.splitlines():
        if max_lines and len(titles) >= max_lines:
            break
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = parts[0]
        if not _DEVA_RE.search(word):
            continue
        try:
            int(parts[-1])
        except ValueError:
            continue
        titles.append(word)
    return titles


def _progress(it, *, desc: str, unit: str, total: int | None):
    """Wrap *it* with tqdm when installed; write to stderr so stdout stays clean for --dry-run TSV."""
    if tqdm is None:
        return it
    return tqdm(it, desc=desc, unit=unit, total=total, file=sys.stderr)


def load_existing_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.is_file():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "\t" not in s:
            continue
        keys.add(s.split("\t", 1)[0].strip())
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="Max pages to process (0 = all)")
    ap.add_argument("--delay", type=float, default=0.35, help="Seconds between parse requests")
    ap.add_argument("--out", type=Path, default=None, help="Write new lines only (TSV)")
    ap.add_argument("--merge", type=Path, default=None, help="Append missing keys to this file")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    ap.add_argument(
        "--freq-file",
        type=Path,
        default=None,
        help="Local frequency file: each line ``word count`` (OpenSubtitles-style); use with --limit for top N",
    )
    ap.add_argument(
        "--freq-url",
        type=str,
        default=None,
        metavar="URL",
        help=f"Download frequency list from URL (use literal DEFAULT for {DEFAULT_FREQ_URL})",
    )
    args = ap.parse_args()

    if not args.dry_run and not args.out and not args.merge:
        print("Specify --out and/or --merge, or use --dry-run.", file=sys.stderr)
        return 2

    use_freq = args.freq_file is not None or args.freq_url is not None
    if use_freq and not args.limit:
        print("Frequency mode requires --limit (top N tokens from the list).", file=sys.stderr)
        return 2

    freq_source_note = ""
    if use_freq:
        if args.freq_file and args.freq_url:
            print("Use only one of --freq-file or --freq-url.", file=sys.stderr)
            return 2
        if args.freq_file:
            text = args.freq_file.read_text(encoding="utf-8")
            freq_source_note = f"local file {args.freq_file}"
        else:
            url = DEFAULT_FREQ_URL if args.freq_url == "DEFAULT" else args.freq_url
            assert url is not None
            text = fetch_url_text(url)
            freq_source_note = url
        titles_iter = iter_top_freq_titles(text, args.limit)

    seen = load_existing_keys(args.merge) if args.merge else set()
    new_rows: list[tuple[str, str]] = []
    skip_existing = 0
    skip_no_deva = 0
    fail: dict[str, int] = {}

    processed = 0

    def handle_title(title: str) -> None:
        nonlocal skip_existing, skip_no_deva, processed
        processed += 1
        if not _DEVA_RE.search(title):
            skip_no_deva += 1
            return
        if title in seen:
            skip_existing += 1
            return

        time.sleep(max(0.0, args.delay))
        try:
            got, reason = extract_for_title(title)
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            print(f"error {title!r}: {e}", file=sys.stderr)
            fail["http_json"] = fail.get("http_json", 0) + 1
            return

        if got is None:
            fail[reason] = fail.get(reason, 0) + 1
            return

        w, ipa = got
        seen.add(w)
        new_rows.append((w, ipa))

    if use_freq:
        bar = _progress(
            titles_iter,
            desc="Wiktionary (freq list)",
            unit="word",
            total=len(titles_iter),
        )
        for title in bar:
            handle_title(title)
    else:

        def _category_titles_limited(n: int):
            g = iter_category_titles()
            for _ in range(n):
                try:
                    yield next(g)
                except StopIteration:
                    break

        if args.limit:
            bar = _progress(
                _category_titles_limited(args.limit),
                desc="Wiktionary (category)",
                unit="page",
                total=args.limit,
            )
            for title in bar:
                handle_title(title)
        else:
            bar = _progress(
                iter_category_titles(),
                desc="Wiktionary (category)",
                unit="page",
                total=None,
            )
            for title in bar:
                handle_title(title)

    print(
        f"processed={processed} new={len(new_rows)} skip_existing={skip_existing} "
        f"skip_no_devanagari_title={skip_no_deva} fail={fail}",
        file=sys.stderr,
    )

    if args.dry_run:
        for w, ipa in new_rows[:20]:
            print(f"{w}\t{ipa}")
        if len(new_rows) > 20:
            print(f"... and {len(new_rows) - 20} more", file=sys.stderr)
        return 0

    if use_freq:
        header = (
            f"# Wiktionary (en): IPA from parsed HTML; titles from frequency list ({freq_source_note}).\n"
            "# Wiktionary license: CC BY-SA ( https://en.wiktionary.org/wiki/Wiktionary:Copyrights )\n"
            "# Frequency list: hermitdave/FrequencyWords (OpenSubtitles), CC BY-SA 4.0\n"
        )
    else:
        header = (
            "# Wiktionary (en): Category:Hindi_terms_with_IPA_pronunciation — phonemic IPA from parsed HTML.\n"
            "# License: CC BY-SA (see https://en.wiktionary.org/wiki/Wiktionary:Copyrights )\n"
        )

    if args.out and new_rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            header + "".join(f"{w}\t{ipa}\n" for w, ipa in new_rows),
            encoding="utf-8",
        )
        print(f"wrote {len(new_rows)} lines to {args.out}", file=sys.stderr)

    if args.merge and new_rows:
        merge_path = args.merge.resolve()
        block = "".join(f"{w}\t{ipa}\n" for w, ipa in new_rows)
        need_leading_nl = merge_path.is_file() and merge_path.stat().st_size > 0
        with merge_path.open("a", encoding="utf-8") as f:
            if need_leading_nl:
                f.write("\n")
            f.write(header + block)
        print(f"appended {len(new_rows)} lines to {merge_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
