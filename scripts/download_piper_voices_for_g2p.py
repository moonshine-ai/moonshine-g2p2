#!/usr/bin/env python3
"""Download Piper ONNX voices from rhasspy/piper-voices into data/<lang>/piper-voices/.

Each voice file is stored directly under ``piper-voices/`` (basename only), e.g.::

    data/ar_msa/piper-voices/ar_JO-kareem-medium.onnx
    data/ar_msa/piper-voices/ar_JO-kareem-medium.onnx.json

Japanese and Korean have no voices in ``rhasspy/piper-voices`` (as of v1.0.0); those
directories are skipped with a message.

Requires: pip install huggingface_hub tqdm

Usage::

    python scripts/download_piper_voices_for_g2p.py
    python scripts/download_piper_voices_for_g2p.py --only de,fr
    python scripts/download_piper_voices_for_g2p.py --dry-run
    python scripts/download_piper_voices_for_g2p.py --flatten-only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ID = "rhasspy/piper-voices"
_REVISION = "v1.0.0"

# Moonshine G2P / repo ``data/<code>/`` folder name -> HF path prefixes under the repo root.
LANG_PREFIXES: dict[str, tuple[str, ...]] = {
    "ar_msa": ("ar/",),
    "de": ("de/",),
    "en_us": ("en/en_US/",),
    # UK English is used by Moonshine TTS; Piper stores it under en/en_GB/.
    "en_gb": ("en/en_GB/",),
    "es_es": ("es/es_ES/",),
    "es_mx": ("es/es_MX/",),
    # Piper also ships es_AR; map to a repo-style folder for Latin-American Spanish G2P.
    "es_ar": ("es/es_AR/",),
    "fr": ("fr/",),
    "hi": ("hi/",),
    "it": ("it/",),
    "nl": ("nl/",),
    "pt_br": ("pt/pt_BR/",),
    "pt_pt": ("pt/pt_PT/",),
    "ru": ("ru/",),
    "tr": ("tr/",),
    "uk": ("uk/",),
    "vi": ("vi/",),
    "zh_hans": ("zh/zh_CN/",),
}

# Rule-based G2P includes ja/ko but Piper HF repo has no ja/ko trees.
NO_PIPER_VOICES: frozenset[str] = frozenset({"ja", "ko"})


def _voice_file_filter(relpath: str) -> bool:
    return relpath.endswith(".onnx") or relpath.endswith(".onnx.json")


def flat_voice_path(dest_root: Path, repo_rel: str) -> Path:
    """Map HF repo path to ``dest_root / <basename>``."""
    name = Path(repo_rel).name
    if not name:
        raise ValueError(f"empty repo path: {repo_rel!r}")
    return dest_root / name


def _is_voice_asset(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix == ".onnx" and not p.name.endswith(".onnx.json"):
        return True
    return p.name.endswith(".onnx.json")


def flatten_piper_voices_dir(dest_root: Path, *, dry_run: bool) -> int:
    """Move nested trees or ``subdir/basename`` into ``piper-voices/<basename>``. Returns files moved."""
    if not dest_root.is_dir():
        return 0
    moved = 0
    # Collect files first so we do not walk into moved targets inconsistently.
    candidates: list[Path] = [p for p in dest_root.rglob("*") if _is_voice_asset(p) and ".cache" not in p.parts]
    # Deeper paths first (harmless either way; avoids odd edge cases with stale dirs).
    candidates.sort(key=lambda p: len(p.relative_to(dest_root).parts), reverse=True)
    for src in candidates:
        try:
            rel = src.relative_to(dest_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) <= 1:
            continue
        target = dest_root / parts[-1]
        if src.resolve() == target.resolve():
            continue
        if dry_run:
            print(f"  would move {src.relative_to(_REPO_ROOT)} -> {target.relative_to(_REPO_ROOT)}")
            moved += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if target.stat().st_size == src.stat().st_size:
                src.unlink()
            else:
                shutil.copy2(src, target)
                src.unlink()
        else:
            shutil.move(str(src), str(target))
        moved += 1

    if not dry_run and moved:
        _prune_empty_dirs(dest_root)
    return moved


def _prune_empty_dirs(root: Path) -> None:
    """Remove empty directories under root (excluding ``.cache``)."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_dir() or ".cache" in path.parts or path == root:
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            try:
                path.rmdir()
            except OSError:
                pass


def download_for_lang(
    moonshine_code: str,
    prefixes: tuple[str, ...],
    *,
    dry_run: bool,
    skip_flatten: bool,
) -> tuple[int, int]:
    """Returns (n_downloaded, n_skipped_existing)."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub tqdm", file=sys.stderr)
        raise SystemExit(1)

    dest_root = _REPO_ROOT / "data" / moonshine_code / "piper-voices"
    if not dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)
        if not skip_flatten:
            n_flat = flatten_piper_voices_dir(dest_root, dry_run=False)
            if n_flat:
                print(f"   flattened {n_flat} voice file(s) -> piper-voices/<basename>")

    from huggingface_hub import HfApi

    api = HfApi()
    all_files = api.list_repo_files(_REPO_ID, revision=_REVISION)
    want: list[str] = []
    for f in all_files:
        if not _voice_file_filter(f):
            continue
        if any(f.startswith(p) for p in prefixes):
            want.append(f)

    want.sort()
    downloaded = 0
    skipped = 0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    iterator = want
    if tqdm is not None:
        iterator = tqdm(want, desc=moonshine_code, unit="file")

    for rel in iterator:
        out_path = flat_voice_path(dest_root, rel)
        if out_path.is_file() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        if dry_run:
            print(f"  would fetch {rel} -> {out_path.relative_to(_REPO_ROOT)}")
            downloaded += 1
            continue
        try:
            cached = hf_hub_download(
                _REPO_ID,
                filename=rel,
                revision=_REVISION,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, out_path)
            downloaded += 1
        except HfHubHTTPError as e:
            print(f"Error downloading {rel}: {e}", file=sys.stderr)
            raise

    return downloaded, skipped


def main() -> int:
    p = argparse.ArgumentParser(description="Download Piper voices for Moonshine G2P languages.")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated moonshine data codes (e.g. de,fr,en_us). Default: all.",
    )
    p.add_argument("--dry-run", action="store_true", help="List files that would be downloaded.")
    p.add_argument(
        "--flatten-only",
        action="store_true",
        help="Only flatten existing nested piper-voices trees under data/*/piper-voices; no download.",
    )
    p.add_argument(
        "--no-flatten",
        action="store_true",
        help="Do not flatten nested layouts before download (default is to flatten).",
    )
    args = p.parse_args()

    if args.flatten_only:
        total = 0
        for pv in sorted(_REPO_ROOT.glob("data/*/piper-voices")):
            n = flatten_piper_voices_dir(pv, dry_run=args.dry_run)
            if n or args.dry_run:
                print(f"{pv.relative_to(_REPO_ROOT)}: {'would move' if args.dry_run else 'moved'} {n}")
            total += n
        print(f"Done. Total {'would move' if args.dry_run else 'moved'} {total}")
        return 0

    if args.only.strip():
        codes = [c.strip() for c in args.only.split(",") if c.strip()]
        for c in codes:
            if c in NO_PIPER_VOICES:
                print(f"Note: {c} — Piper does not publish voices for this locale on {_REPO_ID}.", file=sys.stderr)
            elif c not in LANG_PREFIXES:
                print(f"Unknown code {c!r}; known: {', '.join(sorted(LANG_PREFIXES))}", file=sys.stderr)
                return 1
        to_run = [(c, LANG_PREFIXES[c]) for c in codes if c in LANG_PREFIXES]
    else:
        to_run = list(LANG_PREFIXES.items())
        print("Languages with no Piper bundle on HF (skipped): ja, ko", file=sys.stderr)

    total_dl = 0
    total_skip = 0
    for code, prefixes in to_run:
        print(f"== {code} ({', '.join(prefixes)}) ==")
        d, s = download_for_lang(
            code,
            prefixes,
            dry_run=args.dry_run,
            skip_flatten=args.no_flatten,
        )
        total_dl += d
        total_skip += s
        print(f"   {'would download' if args.dry_run else 'downloaded'} {d}, skipped (already present) {s}")

    print(f"Done. Total {'would download' if args.dry_run else 'downloaded'} {total_dl}, skipped {total_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
