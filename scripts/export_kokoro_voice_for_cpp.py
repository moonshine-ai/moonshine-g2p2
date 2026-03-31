#!/usr/bin/env python3
"""Export a Kokoro ``voices/*.pt`` tensor pack to ``*.kokorovoice`` for C++ ``MoonshineTTS``.

The C++ runtime does not load PyTorch pickles. After downloading the Kokoro bundle::

    python scripts/download_kokoro_onnx.py --out-dir models/kokoro --voices af_heart
    python scripts/export_kokoro_voice_for_cpp.py models/kokoro/voices/af_heart.pt \\
        models/kokoro/voices/af_heart.kokorovoice

Convert every voice in a directory (same layout as Hugging Face / download script)::

    python scripts/export_kokoro_voice_for_cpp.py --voices-dir models/kokoro/voices
    python scripts/export_kokoro_voice_for_cpp.py --voices-dir cpp/data/kokoro/voices
"""

from __future__ import annotations

import argparse
import struct
import sys
import warnings
from pathlib import Path


def export_pt_to_kokorovoice(pt: Path, out: Path, torch) -> int:
    if not pt.is_file():
        print(f"Missing {pt}", file=sys.stderr)
        return 1
    pack = torch.load(pt, map_location="cpu", weights_only=True)
    if not hasattr(pack, "numpy"):
        print(f"Voice .pt must contain a tensor: {pt}", file=sys.stderr)
        return 1
    t = pack.detach().cpu().to(torch.float32)
    while t.dim() > 2:
        squeezed = False
        for d in range(t.dim()):
            if int(t.shape[d]) == 1:
                t = t.squeeze(d)
                squeezed = True
                break
        if not squeezed:
            t = t.reshape(int(t.shape[0]), -1)
            break
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        print(f"{pt}: expected voice tensor to become 2D, got shape {tuple(t.shape)}", file=sys.stderr)
        return 1
    rows, cols = int(t.shape[0]), int(t.shape[1])
    flat = t.numpy().astype("<f4").tobytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"KVO1")
        f.write(struct.pack("<II", rows, cols))
        f.write(flat)
    print(f"Wrote {out} ({rows}x{cols} float32)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_pt", type=Path, nargs="?", help="Kokoro voice .pt path")
    p.add_argument("output_kokorovoice", type=Path, nargs="?", help="Output .kokorovoice path")
    p.add_argument(
        "--voices-dir",
        type=Path,
        metavar="DIR",
        help="Export every DIR/*.pt to DIR/<name>.kokorovoice (omit positional args)",
    )
    args = p.parse_args()
    try:
        import torch
    except ImportError as e:
        print("export_kokoro_voice_for_cpp: need PyTorch (`pip install torch`)", file=sys.stderr)
        raise SystemExit(1) from e

    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

    if args.voices_dir is not None:
        if args.input_pt is not None or args.output_kokorovoice is not None:
            print("Use either --voices-dir or two positional paths, not both.", file=sys.stderr)
            return 2
        d = args.voices_dir
        if not d.is_dir():
            print(f"Not a directory: {d}", file=sys.stderr)
            return 1
        pts = sorted(d.glob("*.pt"))
        if not pts:
            print(f"No .pt files in {d}", file=sys.stderr)
            return 1
        code = 0
        for pt in pts:
            out = pt.with_suffix(".kokorovoice")
            r = export_pt_to_kokorovoice(pt, out, torch)
            if r != 0:
                code = r
        return code

    if args.input_pt is None or args.output_kokorovoice is None:
        p.print_help()
        print("\nError: provide input.pt and output.kokorovoice, or use --voices-dir DIR.", file=sys.stderr)
        return 2
    return export_pt_to_kokorovoice(args.input_pt, args.output_kokorovoice, torch)


if __name__ == "__main__":
    raise SystemExit(main())
