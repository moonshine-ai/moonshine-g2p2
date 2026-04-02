#!/usr/bin/env python3
"""
Pack large FP32 initializers in Piper voice ONNX files using onnx-shrink-ray (int8 storage +
Cast/Mul/Add dequant), matching ``export_arabic_msa_diacritizer_onnx.py`` / ``download_kokoro_onnx.py``.

Default targets: ``moonshine-tts/data/**/piper-voices/*.onnx`` and ``data/**/piper-voices/*.onnx`` (two copies
in this repo). Files that already look int8-packed are skipped.

Requires: ``pip install onnx onnx-shrink-ray onnx-graphsurgeon``

Example::

    python scripts/shrink_piper_voice_onnx_weights.py
    python scripts/shrink_piper_voice_onnx_weights.py --root moonshine-tts/data --dry-run
    python scripts/shrink_piper_voice_onnx_weights.py --backup
    python scripts/shrink_piper_voice_onnx_weights.py --name-contains melotts
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent


def _patch_onnx_for_onnx_graphsurgeon() -> None:
    import onnx.helper as h

    if not hasattr(h, "float32_to_bfloat16"):

        def float32_to_bfloat16(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            ui = x.view(np.uint32)
            return ((ui + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)

        h.float32_to_bfloat16 = float32_to_bfloat16  # type: ignore[method-assign]

    if not hasattr(h, "float32_to_float8e4m3"):

        def float32_to_float8e4m3(x, fn=True, uz=False):
            raise RuntimeError("float32_to_float8e4m3 compat stub: not used")

        h.float32_to_float8e4m3 = float32_to_float8e4m3  # type: ignore[method-assign]


def _initializer_bytes_by_dtype(onnx_path: Path) -> dict[str, int]:
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    out: dict[str, int] = {}
    for init in m.graph.initializer:
        name = onnx.TensorProto.DataType.Name(init.data_type)
        if init.raw_data:
            n = len(init.raw_data)
        elif init.float_data:
            n = len(init.float_data) * 4
        elif init.int32_data:
            n = len(init.int32_data) * 4
        elif init.int64_data:
            n = len(init.int64_data) * 8
        elif init.double_data:
            n = len(init.double_data) * 8
        else:
            n = 0
        out[name] = out.get(name, 0) + n
    return out


def _looks_int8_packed(onnx_path: Path) -> bool:
    b = _initializer_bytes_by_dtype(onnx_path)
    total = sum(b.values()) or 1
    int8ish = b.get("INT8", 0) + b.get("UINT8", 0)
    return int8ish > 0.25 * total


def _shrink_onnx_weights(
    onnx_path: Path,
    *,
    min_elements: int,
    verbose: bool,
) -> None:
    _patch_onnx_for_onnx_graphsurgeon()
    try:
        from onnx_shrink_ray.shrink import quantize_weights
    except ImportError as e:
        raise SystemExit(
            "ONNX shrink requires onnx-shrink-ray. Install with "
            "`pip install onnx-shrink-ray onnx-graphsurgeon`."
        ) from e

    import onnx

    model = onnx.load(str(onnx_path))
    new_model = quantize_weights(
        model,
        min_elements=min_elements,
        float_quantization=False,
        verbose=verbose,
    )
    sidecar = onnx_path.parent / (onnx_path.name + ".data")
    if sidecar.is_file():
        sidecar.unlink()
    onnx.save(new_model, str(onnx_path))


def _clamp_onnx_ir_version(onnx_path: Path, *, max_ir: int) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    if model.ir_version <= max_ir:
        return
    model.ir_version = max_ir
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def _collect_onnx_paths(roots: list[Path], *, name_contains: list[str] | None) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        out.extend(sorted(root.glob("**/piper-voices/*.onnx")))
    if name_contains:
        needles = [n.lower() for n in name_contains if n]
        out = [p for p in out if any(n in p.name.lower() for n in needles)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Shrink Piper piper-voices ONNX weights (onnx-shrink-ray).")
    ap.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="Directory to scan (repeatable). Default: moonshine-tts/data and data under repo root.",
    )
    ap.add_argument("--dry-run", action="store_true", help="List files only; do not modify.")
    ap.add_argument(
        "--backup",
        action="store_true",
        help="Copy each file to <name>.onnx.fp32.bak before overwriting.",
    )
    ap.add_argument("--shrink-min-elements", type=int, default=16 * 1024)
    ap.add_argument("--shrink-verbose", action="store_true")
    ap.add_argument("--max-onnx-ir", type=int, default=11, help="Clamp model IR version for older ORT.")
    ap.add_argument(
        "--name-contains",
        action="append",
        default=None,
        metavar="SUBSTR",
        help="Only process ONNX files whose basename contains SUBSTR (case-insensitive). Repeatable (OR).",
    )
    args = ap.parse_args()

    roots = args.root
    if not roots:
        roots = [_REPO / "moonshine-tts" / "data", _REPO / "data"]
    roots = [r.resolve() for r in roots]

    paths = _collect_onnx_paths(roots, name_contains=args.name_contains)
    if not paths:
        print("No piper-voices/*.onnx found under given roots.", file=sys.stderr)
        raise SystemExit(1)

    skipped = 0
    done = 0
    for p in paths:
        rel = p.relative_to(_REPO) if p.is_relative_to(_REPO) else p
        if _looks_int8_packed(p):
            print(f"skip (already int8-packed): {rel}")
            skipped += 1
            continue
        before = p.stat().st_size
        if args.dry_run:
            print(f"would shrink: {rel} ({before / 1e6:.2f} MB)")
            done += 1
            continue
        if args.backup:
            bak = p.parent / (p.name + ".fp32.bak")
            shutil.copy2(p, bak)
        print(f"shrinking: {rel} ({before / 1e6:.2f} MB)…", file=sys.stderr)
        _shrink_onnx_weights(
            p,
            min_elements=args.shrink_min_elements,
            verbose=args.shrink_verbose,
        )
        _clamp_onnx_ir_version(p, max_ir=args.max_onnx_ir)
        after = p.stat().st_size
        print(f"  -> {after / 1e6:.2f} MB")
        done += 1

    print(
        f"Finished: {done} processed, {skipped} skipped.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
