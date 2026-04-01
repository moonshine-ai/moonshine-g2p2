#!/usr/bin/env python3
"""
List cpp/data files needed for C++ TTS (MoonshineTTS/Kokoro vs Piper) per language.

Engine choice matches speak.py / product intent: Kokoro when the locale is in the C++
MoonshineTTS LangProfile table or resolves to Spanish rules; otherwise Piper with the
default ONNX basename from cpp/piper-tts.cpp (see PiperLangRow).

Spanish routing in resolve_piper_lang compares norm (lowercased) to "es-ES"/"es-AR";
that mirrors cpp/piper-tts.cpp exactly (may not distinguish es_es vs es_mx for all CLI forms).

Usage:
  python scripts/cpp_tts_data_footprint.py
  python scripts/cpp_tts_data_footprint.py --list-tags
  python scripts/cpp_tts_data_footprint.py en_us de ja es-MX
  python scripts/cpp_tts_data_footprint.py --sort-by-largest de fr
  python scripts/cpp_tts_data_footprint.py --create-bundle /tmp/moonshine-tts-data de fr
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from collections.abc import Iterable


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CPP_DATA = _REPO_ROOT / "cpp" / "data"

# cpp/moonshine-tts.cpp lookup_lang_profile keys -> default_voice (third field is G2P dialect tag).
_LANG_PROFILE: dict[str, tuple[str, str]] = {
    "en_us": ("af_heart", "en_us"),
    "en-us": ("af_heart", "en_us"),
    "en": ("af_heart", "en_us"),
    "en_gb": ("bf_emma", "en_us"),
    "en-gb": ("bf_emma", "en_us"),
    "es": ("ef_dora", "es-MX"),
    "fr": ("ff_siwis", "fr"),
    "hi": ("hf_alpha", "hi"),
    "it": ("if_sara", "it"),
    "pt_br": ("pf_dora", "pt_br"),
    "pt-br": ("pf_dora", "pt_br"),
    "pt": ("pf_dora", "pt_br"),
    "ja": ("jf_alpha", "ja"),
    "jp": ("jf_alpha", "ja"),
    "zh": ("zf_xiaobei", "zh"),
    "zh_hans": ("zf_xiaobei", "zh"),
}

# cpp/src/lang-specific/spanish.cpp spanish_dialect_cli_ids()
_SPANISH_CLI_IDS: frozenset[str] = frozenset(
    {
        "es-419",
        "es-AR",
        "es-BO",
        "es-CL",
        "es-CO",
        "es-CU",
        "es-DO",
        "es-EC",
        "es-ES",
        "es-ES-distincion",
        "es-GT",
        "es-MX",
        "es-PE",
        "es-PR",
        "es-PY",
        "es-UY",
        "es-VE",
    }
)

# cpp/piper-tts.cpp PiperLangRow (keys after normalize_lang_key); g2p_dialect, data_subdir, default_onnx
_PIPER_LANG: dict[str, tuple[str, str, str]] = {
    "en_us": ("en_us", "en_us", "en_US-lessac-medium.onnx"),
    "en-us": ("en_us", "en_us", "en_US-lessac-medium.onnx"),
    "en": ("en_us", "en_us", "en_US-lessac-medium.onnx"),
    "en_gb": ("en_us", "en_gb", "en_GB-cori-medium.onnx"),
    "en-gb": ("en_us", "en_gb", "en_GB-cori-medium.onnx"),
    "es": ("es-MX", "es_mx", "es_MX-ald-medium.onnx"),
    "es_mx": ("es-MX", "es_mx", "es_MX-ald-medium.onnx"),
    "es_es": ("es-ES", "es_es", "es_ES-davefx-medium.onnx"),
    "es_ar": ("es-AR", "es_ar", "es_AR-daniela-high.onnx"),
    "fr": ("fr-FR", "fr", "fr_FR-siwis-medium.onnx"),
    "hi": ("hi", "hi", "hi_IN-pratham-medium.onnx"),
    "it": ("it-IT", "it", "it_IT-paola-medium.onnx"),
    "pt_br": ("pt_br", "pt_br", "pt_BR-cadu-medium.onnx"),
    "pt-br": ("pt_br", "pt_br", "pt_BR-cadu-medium.onnx"),
    "pt": ("pt_br", "pt_br", "pt_BR-cadu-medium.onnx"),
    "pt_pt": ("pt_pt", "pt_pt", "pt_PT-tugão-medium.onnx"),
    "pt-pt": ("pt_pt", "pt_pt", "pt_PT-tugão-medium.onnx"),
    "zh": ("zh", "zh_hans", "zh_CN-huayan-medium.onnx"),
    "zh_hans": ("zh", "zh_hans", "zh_CN-huayan-medium.onnx"),
    "ar_msa": ("ar", "ar_msa", "ar_JO-kareem-medium.onnx"),
    "ar": ("ar", "ar_msa", "ar_JO-kareem-medium.onnx"),
    "de": ("de-DE", "de", "de_DE-thorsten-medium.onnx"),
    "nl": ("nl-NL", "nl", "nl_NL-mls-medium.onnx"),
    "ru": ("ru-RU", "ru", "ru_RU-denis-medium.onnx"),
    "tr": ("tr-TR", "tr", "tr_TR-dfki-medium.onnx"),
    "uk": ("uk-UA", "uk", "uk_UA-ukrainian_tts-medium.onnx"),
    "vi": ("vi-VN", "vi", "vi_VN-vais1000-medium.onnx"),
}


def normalize_lang_key(raw: str) -> str:
    s = raw.strip()
    out = []
    for c in s:
        if c == " ":
            out.append("_")
        elif "A" <= c <= "Z":
            out.append(chr(ord(c) - ord("A") + ord("a")))
        else:
            out.append(c)
    return "".join(out)


def normalize_rule_based_dialect_cli_key(raw: str) -> str:
    s = raw.strip()
    chs = []
    for c in s:
        if c == "_":
            chs.append("-")
        elif "A" <= c <= "Z":
            chs.append(chr(ord(c) - ord("A") + ord("a")))
        else:
            chs.append(c)
    return "".join(chs)


def normalize_spanish_dialect_cli_key(raw: str) -> str:
    """Mirror cpp/src/moonshine-g2p.cpp (anonymous) + rule-based-factory spanish normalizer."""
    s = normalize_rule_based_dialect_cli_key(raw)
    if len(s) >= 3 and s[0] == "e" and s[1] == "s" and s[2] == "-":
        chars = list(s)
        i = 3
        while i < len(chars) and chars[i] != "-":
            if chars[i].isalpha():
                chars[i] = chars[i].upper()
            i += 1
        if i < len(chars) and chars[i] == "-":
            i += 1
            while i < len(chars):
                if chars[i].isalpha():
                    chars[i] = chars[i].lower()
                i += 1
        s = "".join(chars)
    return s


def dialect_resolves_to_spanish_rules(dialect_id: str, spanish_narrow_obstruents: bool = True) -> bool:
    _ = spanish_narrow_obstruents
    spanish_key = normalize_spanish_dialect_cli_key(dialect_id)
    return spanish_key in _SPANISH_CLI_IDS


def uses_kokoro(lang: str) -> bool:
    k = normalize_lang_key(lang)
    if k in _LANG_PROFILE:
        return True
    norm = normalize_rule_based_dialect_cli_key(lang)
    return bool(norm) and dialect_resolves_to_spanish_rules(norm)


def resolve_piper_lang(lk: str) -> tuple[str, str, str]:
    """Returns (g2p_dialect, data_subdir, default_onnx). Mirrors cpp/piper-tts.cpp resolve_piper_lang."""
    k = normalize_lang_key(lk)
    if k in ("ja", "jp", "ko", "ko_kr", "korean"):
        raise ValueError(f"Piper does not bundle Japanese/Korean ONNX here; use Kokoro for {lk!r}.")
    norm = normalize_rule_based_dialect_cli_key(lk)
    if norm and dialect_resolves_to_spanish_rules(norm):
        g2p_dialect = norm
        # cpp compares against "es-ES" / "es-AR" on lowercased norm — see note in module docstring.
        if norm.rfind("es-ES", 0) == 0:
            data_subdir = "es_es"
            default_onnx = "es_ES-davefx-medium.onnx"
        elif norm == "es-AR":
            data_subdir = "es_ar"
            default_onnx = "es_AR-daniela-high.onnx"
        else:
            data_subdir = "es_mx"
            default_onnx = "es_MX-ald-medium.onnx"
        return g2p_dialect, data_subdir, default_onnx
    row = _PIPER_LANG.get(k)
    if row is None:
        raise ValueError(f"Unsupported --lang for Piper-style TTS: {lk!r}")
    return row


def _onnx_sidecar_json(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(onnx_path.suffix + ".json")


def _kokoro_bundle_files(kokoro_dir: Path, voice: str) -> list[Path]:
    return [
        kokoro_dir / "config.json",
        kokoro_dir / "model.onnx",
        kokoro_dir / "voices" / f"{voice}.kokorovoice",
    ]


def _onnx_dir_files(model_dir: Path) -> list[Path]:
    meta = model_dir / "meta.json"
    onnx_name = "model.onnx"
    if meta.is_file():
        with meta.open(encoding="utf-8") as f:
            onnx_name = json.load(f).get("onnx_model_file", "model.onnx")
    onnx_path = model_dir / onnx_name
    out: list[Path] = [
        model_dir / "vocab.txt",
        model_dir / "tokenizer_config.json",
        meta,
        onnx_path,
    ]
    sidecar = model_dir / (onnx_name + ".data")
    if sidecar.is_file():
        out.append(sidecar)
    return out


def _english_g2p_files(root: Path) -> list[Path]:
    base = root / "en_us"
    paths: list[Path] = []
    dict_main = base / "dict_filtered_heteronyms.tsv"
    paths.append(dict_main)
    cfg = base / "g2p-config.json"
    paths.append(cfg)
    if cfg.is_file():
        with cfg.open(encoding="utf-8") as f:
            j = json.load(f)
        if j.get("uses_heteronym_model"):
            het = base / "heteronym"
            paths.append(het / "model.onnx")
            paths.append(het / "homograph_index.json")
            paths.append(het / "onnx-config.json")
        if j.get("uses_oov_model"):
            oov = base / "oov"
            paths.append(oov / "model.onnx")
            paths.append(oov / "onnx-config.json")
    return paths


def _french_g2p_files(root: Path) -> list[Path]:
    fr = root / "fr"
    out = [fr / "dict.tsv"]
    if fr.is_dir():
        for p in sorted(fr.iterdir()):
            if p.is_file() and p.suffix.lower() == ".csv":
                out.append(p)
    return out


def g2p_cpp_data_files_for_moonshine_dialect(model_root: Path, g2p_dialect: str) -> list[Path]:
    """
    Files under cpp/data used by MoonshineG2P for the given dialect string
    (as passed from MoonshineTTS / Piper after resolution).
    """
    norm = normalize_rule_based_dialect_cli_key(g2p_dialect)

    # UK Kokoro uses the same ``en_us`` G2P dialect string and assets as US English (see C++ LangProfile).
    if norm in ("en-gb", "en_gb"):
        return _english_g2p_files(model_root)

    if norm in ("en-us", "english", "en", "en_us"):
        return _english_g2p_files(model_root)

    if dialect_resolves_to_spanish_rules(norm):
        return []

    if norm in ("de", "de-de", "de_de"):
        return [model_root / "de" / "dict.tsv"]

    if norm in ("fr", "fr-fr", "fr_fr", "french"):
        return _french_g2p_files(model_root)

    if norm in ("nl", "nl-nl", "nl_nl"):
        return [model_root / "nl" / "dict.tsv"]

    if norm in ("it", "it-it", "it_it"):
        return [model_root / "it" / "dict.tsv"]

    if norm in ("ru", "ru-ru", "ru_ru"):
        return [model_root / "ru" / "dict.tsv"]

    if norm in ("zh", "zh-hans", "zh-cn", "zh_cn", "zh_hans", "cmn", "chinese"):
        zh_onnx = model_root / "zh_hans" / "roberta_chinese_base_upos_onnx"
        return [model_root / "zh_hans" / "dict.tsv", *_onnx_dir_files(zh_onnx)]

    if norm in ("ko", "ko-kr", "ko_kr", "korean"):
        return [model_root / "ko" / "dict.tsv"]

    if norm in ("vi", "vi-vn", "vi_vn"):
        return [model_root / "vi" / "dict.tsv"]

    if norm in ("ja", "ja-jp", "ja_jp", "japanese"):
        ja_onnx = model_root / "ja" / "roberta_japanese_char_luw_upos_onnx"
        return [model_root / "ja" / "dict.tsv", *_onnx_dir_files(ja_onnx)]

    if norm in ("ar", "ar-msa", "ar_msa", "arabic", "msa"):
        ar_onnx = model_root / "ar_msa" / "arabertv02_tashkeel_fadel_onnx"
        pdict = model_root / "ar_msa" / "dict.tsv"
        return [*_onnx_dir_files(ar_onnx), pdict]

    if norm in ("pt-br", "pt_br", "brazil", "brazilian-portuguese", "brazilianportuguese", "portuguese-brazil"):
        return [model_root / "pt_br" / "dict.tsv"]

    if norm in ("pt-pt", "pt_pt", "portugal", "european-portuguese", "europeanportuguese"):
        return [model_root / "pt_pt" / "dict.tsv"]

    if norm in ("tr", "tr-tr", "tr_tr"):
        return []

    if norm in ("uk", "uk-ua", "uk_ua"):
        return []

    if norm in ("hi", "hi-in", "hi_in", "hindi"):
        return [model_root / "hi" / "dict.tsv"]

    # es-MX etc.: Spanish rules — no data files
    if norm.startswith("es-") or norm == "es":
        return []

    raise ValueError(f"Unknown G2P dialect for asset listing: {g2p_dialect!r} (norm={norm!r})")


def resolve_kokoro_g2p_dialect(lang: str) -> str:
    k = normalize_lang_key(lang)
    if k in _LANG_PROFILE:
        return _LANG_PROFILE[k][1]
    norm = normalize_rule_based_dialect_cli_key(lang)
    if norm and dialect_resolves_to_spanish_rules(norm):
        return normalize_spanish_dialect_cli_key(lang)
    raise ValueError(f"Not a Kokoro locale in C++ MoonshineTTS: {lang!r}")


def required_paths_for_lang(lang: str, *, cpp_data: Path) -> tuple[str, list[Path]]:
    """Returns (engine_label, sorted unique paths under cpp/data)."""
    model_root = cpp_data.resolve()
    kokoro_dir = model_root / "kokoro"
    if uses_kokoro(lang):
        voice = _LANG_PROFILE.get(normalize_lang_key(lang), (None, None))[0]
        if voice is None:
            # Spanish: ef_dora
            voice = "ef_dora"
        g2p_tag = resolve_kokoro_g2p_dialect(lang)
        g2p_files = g2p_cpp_data_files_for_moonshine_dialect(model_root, g2p_tag)
        tts_files = _kokoro_bundle_files(kokoro_dir, voice)
        merged = sorted({p.resolve() for p in g2p_files + tts_files}, key=lambda x: str(x))
        return "kokoro", merged

    g2p_dialect, data_subdir, default_onnx = resolve_piper_lang(lang)
    g2p_files = g2p_cpp_data_files_for_moonshine_dialect(model_root, g2p_dialect)
    voices_dir = model_root / data_subdir / "piper-voices"
    onnx_path = voices_dir / default_onnx
    tts_files = [onnx_path, _onnx_sidecar_json(onnx_path)]
    merged = sorted({p.resolve() for p in g2p_files + tts_files}, key=lambda x: str(x))
    return "piper", merged


def human_size(n: int) -> str:
    if n < 0:
        return "?"
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} B"


def all_supported_language_tags() -> list[str]:
    s: set[str] = set(_PIPER_LANG.keys()) | set(_LANG_PROFILE.keys()) | set(_SPANISH_CLI_IDS)
    # Common aliases not already keys
    s.update(["zh", "es", "pt", "ar", "zh_hans"])
    return sorted(s, key=lambda x: (x.lower(), x))


def collect_paths_for_tags(
    tags: list[str], *, cpp_data: Path
) -> tuple[dict[str, Path], list[tuple[str, str]]]:
    """Merge resolved paths for each tag. Returns (path_key -> path, [(tag, error), ...])."""
    merged: dict[str, Path] = {}
    errors: list[tuple[str, str]] = []
    for raw in tags:
        try:
            _, paths = required_paths_for_lang(raw, cpp_data=cpp_data)
        except ValueError as e:
            errors.append((raw, str(e)))
            continue
        for p in paths:
            r = p.resolve()
            merged[str(r)] = r
    return merged, errors


def _paths_sort_order(paths: list[Path], *, sort_by_largest: bool) -> list[Path]:
    if not sort_by_largest:
        return sorted(paths, key=lambda p: str(p))

    def key_fn(p: Path) -> tuple:
        if p.is_file():
            return (0, -p.stat().st_size, str(p))
        return (1, 0, str(p))

    return sorted(paths, key=key_fn)


def copy_cpp_tts_bundle(
    paths: Iterable[Path], *, cpp_data: Path, dest_root: Path
) -> tuple[int, int, list[Path]]:
    """
    Copy files under *dest_root* preserving paths relative to *cpp_data*.
    Returns (n_copied, n_bytes_copied, missing_paths).
    """
    cpp_data = cpp_data.resolve()
    dest_root = dest_root.resolve()
    missing: list[Path] = []
    n_copied = 0
    n_bytes = 0
    for p in paths:
        pr = p.resolve()
        if not pr.is_file():
            missing.append(pr)
            continue
        try:
            rel = os.path.relpath(pr, cpp_data)
        except ValueError:
            raise SystemExit(
                f"Refusing to bundle path outside --data-root:\n  {pr}\n  data-root={cpp_data}"
            ) from None
        target = dest_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pr, target)
        n_copied += 1
        n_bytes += pr.stat().st_size
    return n_copied, n_bytes, missing


def print_path_set_lines(
    paths_by_key: dict[str, Path], cpp_data: Path, *, sort_by_largest: bool = False
) -> tuple[int, int]:
    """Print rel + size per path. Returns (total_bytes_of_existing_files, n_paths)."""
    if sort_by_largest:
        ordered = _paths_sort_order(list(paths_by_key.values()), sort_by_largest=True)
        key_order = [str(p.resolve()) for p in ordered]
    else:
        key_order = sorted(paths_by_key.keys())

    total = 0
    for key in key_order:
        p = paths_by_key[key]
        try:
            rel = os.path.relpath(p, cpp_data)
        except ValueError:
            rel = str(p)
        if p.is_file():
            sz = p.stat().st_size
            total += sz
            print(f"    {rel}\t{human_size(sz)}")
        else:
            print(f"    {rel}\t(missing)")
    return total, len(paths_by_key)


def main() -> None:
    ap = argparse.ArgumentParser(description="cpp/data footprint for C++ TTS (Kokoro vs Piper).")
    ap.add_argument(
        "langs",
        nargs="*",
        help="Language/dialect CLI tags (e.g. en_us de ja). If omitted, print combined footprint for all supported tags.",
    )
    ap.add_argument(
        "--list-tags",
        action="store_true",
        help="Print supported language/dialect tags and exit.",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_CPP_DATA,
        help=f"Override cpp data root (default: {_DEFAULT_CPP_DATA})",
    )
    ap.add_argument(
        "--sort-by-largest",
        action="store_true",
        help="List files in decreasing order of size (missing paths last).",
    )
    ap.add_argument(
        "--create-bundle",
        type=Path,
        default=None,
        metavar="DEST",
        help=(
            "After listing, copy all resolved files for the language set into DEST, "
            "preserving paths relative to --data-root. Exits with status 1 if any file is missing."
        ),
    )
    args = ap.parse_args()
    cpp_data = args.data_root.resolve()

    if args.list_tags and args.create_bundle is not None:
        raise SystemExit("--list-tags cannot be used with --create-bundle")

    if args.list_tags:
        print("Supported language/dialect tags (union of C++ Piper + Moonshine/Kokoro + Spanish CLI ids):")
        for tag in all_supported_language_tags():
            print(f"  {tag}")
        print(
            "\nNote: moonshine-tts-cli infers ko from Hangul, but MoonshineTTS::resolve_lang_for_tts "
            "has no Korean LangProfile yet, so full Kokoro TTS for ko is not wired in C++."
        )
        return

    if not args.langs:
        tags = all_supported_language_tags()
        path_map, errs = collect_paths_for_tags(tags, cpp_data=cpp_data)
        print(
            f"Combined cpp/data footprint for all {len(tags)} supported language/dialect tags "
            "(union of C++ Piper + Moonshine/Kokoro + Spanish CLI ids)."
        )
        print(f"\n=== {len(path_map)} unique files ===")
        total, _ = print_path_set_lines(path_map, cpp_data, sort_by_largest=args.sort_by_largest)
        print(f"\n  total size (existing files): {human_size(total)}")
        if errs:
            print("\n  Skipped tags (errors):")
            for tag, msg in errs:
                print(f"    {tag!r}: {msg}")
        print(
            "\nNote: moonshine-tts-cli infers ko from Hangul, but MoonshineTTS::resolve_lang_for_tts "
            "has no Korean LangProfile yet, so full Kokoro TTS for ko is not wired in C++."
        )
        if args.create_bundle is not None:
            print(f"\n=== bundle → {args.create_bundle} ===")
            n_copied, n_bytes, missing = copy_cpp_tts_bundle(
                path_map.values(), cpp_data=cpp_data, dest_root=args.create_bundle
            )
            print(
                f"  copied {n_copied} files ({human_size(n_bytes)}) under {args.create_bundle.resolve()}"
            )
            if missing:
                print("  missing (not copied):", file=sys.stderr)
                for p in missing:
                    try:
                        r = os.path.relpath(p, cpp_data)
                    except ValueError:
                        r = str(p)
                    print(f"    {r}", file=sys.stderr)
                raise SystemExit(1)
        print("\nUse --list-tags to print tag names; pass tags as arguments for a per-language breakdown.")
        return

    path_map_union: dict[str, Path] | None = None
    if args.create_bundle is not None:
        path_map_union, errs_union = collect_paths_for_tags(args.langs, cpp_data=cpp_data)
        if errs_union:
            print("Bundle: skipped tags (errors):", file=sys.stderr)
            for tag, msg in errs_union:
                print(f"  {tag!r}: {msg}", file=sys.stderr)

    grand: dict[str, tuple[int, str]] = {}
    for raw in args.langs:
        print(f"\n=== {raw!r} ===")
        try:
            engine, paths = required_paths_for_lang(raw, cpp_data=cpp_data)
        except ValueError as e:
            print(f"  Error: {e}")
            continue
        print(f"  engine: {engine}")
        subtotal = 0
        for p in _paths_sort_order(list(paths), sort_by_largest=args.sort_by_largest):
            try:
                rel = os.path.relpath(p, cpp_data)
            except ValueError:
                rel = str(p)
            if p.is_file():
                sz = p.stat().st_size
                subtotal += sz
                h = human_size(sz)
                print(f"    {rel}\t{h}")
                grand[str(p)] = (sz, rel)
            else:
                print(f"    {rel}\t(missing)")
        print(f"  subtotal (existing files): {human_size(subtotal)}")

    if len(args.langs) > 1:
        total = sum(sz for sz, _ in grand.values())
        print(f"\n=== Combined (deduped paths) ===")
        print(f"  {len(grand)} unique files")
        print(f"  total size: {human_size(total)}")

    if args.create_bundle is not None and path_map_union is not None:
        print(f"\n=== bundle → {args.create_bundle} ===")
        n_copied, n_bytes, missing = copy_cpp_tts_bundle(
            path_map_union.values(), cpp_data=cpp_data, dest_root=args.create_bundle
        )
        print(
            f"  copied {n_copied} files ({human_size(n_bytes)}) under {args.create_bundle.resolve()}"
        )
        if missing:
            print("  missing (not copied):", file=sys.stderr)
            for p in missing:
                try:
                    r = os.path.relpath(p, cpp_data)
                except ValueError:
                    r = str(p)
                print(f"    {r}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
