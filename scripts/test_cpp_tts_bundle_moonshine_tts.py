#!/usr/bin/env python3
"""
For each supported TTS language tag from ``cpp_tts_data_footprint``:

1. Copy the exact file set ``required_paths_for_lang`` would list into a fresh temp directory
   (same layout as ``--create-bundle``).
2. Run ``moonshine_tts`` with ``--model-root`` pointing at that directory so G2P + Kokoro/Piper
   resolve assets only from the bundle.

Requires a built ``moonshine-tts/build/moonshine_tts`` (see ``moonshine-tts/data/kokoro/README.md``).

Example::

    cmake --build moonshine-tts/build --target moonshine_tts
    python scripts/test_cpp_tts_bundle_moonshine_tts.py
    python scripts/test_cpp_tts_bundle_moonshine_tts.py --only en_us de ja
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import cpp_tts_data_footprint as ctf  # noqa: E402


def _tmp_tag_prefix(tag: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", tag)[:48]
    return f"moonshine_bundle_{safe}_"


def _sample_text(tag: str) -> str:
    """Short UTF-8 phrase appropriate for G2P + synthesis smoke (not linguistic coverage)."""
    k = ctf.normalize_lang_key(tag)
    t = ctf.normalize_rule_based_dialect_cli_key(tag)
    if k in ("ja", "jp") or t.startswith("ja"):
        return "こんにちは。"
    if k.startswith("zh") or "chinese" in k:
        return "你好。"
    if k in ("ar", "ar_msa") or t.startswith("ar"):
        return "مرحبا"
    if k in ("hi",) or "hindi" in k:
        return "नमस्ते"
    if k in ("ru",) or t.startswith("ru"):
        return "Привет"
    if k in ("uk",) or t.startswith("uk"):
        return "Привіт"
    if k in ("vi",) or t.startswith("vi"):
        return "Xin chào"
    if ctf.dialect_resolves_to_spanish_rules(t) or k.startswith("es") or t.startswith("es-"):
        return "Hola"
    if k in ("fr",) or t.startswith("fr"):
        return "Bonjour"
    if k in ("de",) or t.startswith("de"):
        return "Hallo"
    if k in ("it",) or t.startswith("it"):
        return "Ciao"
    if k in ("nl",) or t.startswith("nl"):
        return "Hallo"
    if "pt" in k or t.startswith("pt"):
        return "Olá"
    if k in ("tr",) or t.startswith("tr"):
        return "Merhaba"
    return "Hello world"


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test cpp_tts_data_footprint bundles with moonshine_tts.")
    ap.add_argument(
        "--moonshine-tts",
        type=Path,
        default=_REPO / "moonshine-tts" / "build" / "moonshine_tts",
        help="Path to moonshine_tts executable",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=_REPO / "moonshine-tts" / "data",
        help="Source tree (same as cpp_tts_data_footprint --data-root)",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="TAG",
        help="If set, only these language tags (otherwise all supported tags)",
    )
    ap.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Print temp directory paths and do not delete them after each tag",
    )
    args = ap.parse_args()

    exe = args.moonshine_tts.resolve()
    if not exe.is_file():
        raise SystemExit(f"missing moonshine_tts binary: {exe} (build with cmake --build moonshine-tts/build --target moonshine_tts)")

    cpp_data = args.data_root.resolve()
    tags = args.only if args.only is not None else ctf.all_supported_language_tags()

    failures: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    ok_n = 0

    for tag in tags:
        try:
            engine, paths = ctf.required_paths_for_lang(tag, cpp_data=cpp_data)
        except ValueError as e:
            skipped.append((tag, str(e)))
            print(f"skip {tag!r}: {e}")
            continue

        td_path = Path(tempfile.mkdtemp(prefix=_tmp_tag_prefix(tag)))
        if args.keep_tmp:
            print(f"  tmp {tag!r}: {td_path}")

        try:
            _n_copied, _n_bytes, missing = ctf.copy_cpp_tts_bundle(
                paths, cpp_data=cpp_data, dest_root=td_path
            )
            if missing:
                rels = []
                for p in missing:
                    try:
                        rels.append(str(Path(p).relative_to(cpp_data)))
                    except ValueError:
                        rels.append(str(p))
                failures.append((tag, "missing files: " + ", ".join(rels)))
                print(f"FAIL {tag!r}: missing {rels}")
                continue

            wav = td_path / "smoke.wav"
            text = _sample_text(tag)
            cmd = [
                str(exe),
                "--engine",
                engine,
                "--model-root",
                str(td_path),
                "--lang",
                tag,
                "-o",
                str(wav),
                "--text",
                text,
            ]
            r = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip() or f"exit {r.returncode}"
                failures.append((tag, err))
                print(f"FAIL {tag!r}: {err[:500]}")
                continue
            if not wav.is_file() or wav.stat().st_size < 100:
                failures.append((tag, f"bad wav {wav}"))
                print(f"FAIL {tag!r}: wav missing or tiny")
                continue
            ok_n += 1
            print(f"ok   {tag!r} ({engine}, {wav.stat().st_size} bytes)")
        finally:
            if not args.keep_tmp:
                shutil.rmtree(td_path, ignore_errors=True)

    print(f"\nSummary: {ok_n} ok, {len(skipped)} skipped, {len(failures)} failed")
    if failures:
        for tag, msg in failures:
            print(f"  {tag!r}: {msg[:300]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
