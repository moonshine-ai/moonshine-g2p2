#!/usr/bin/env python3
"""
Download ``onnx/model_quantized.onnx`` from `onnx-community/Kokoro-82M-ONNX` and patch it for C++
``MoonshineTTS``.

The community ONNX uses input name ``style`` and ``speed`` as float32 ``[1]``; this repo expects
``ref_s`` and (depending on export) ``speed`` as double scalar or float ``[1]``. This script:

1. Renames the graph input ``style`` → ``ref_s`` (and the single consumer edge).
2. Writes to ``--out`` (default: ``cpp/data/kokoro/model.onnx``).

C++ ``moonshine-tts.cpp`` detects ``speed`` element type at runtime (float vs double).

Source: https://huggingface.co/onnx-community/Kokoro-82M-ONNX
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO / "cpp" / "data" / "kokoro" / "model.onnx"
_HF_REPO = "onnx-community/Kokoro-82M-ONNX"
_HF_FILE = "onnx/model_quantized.onnx"


def patch_style_to_ref_s(model) -> None:
    import onnx

    for inp in model.graph.input:
        if inp.name == "style":
            inp.name = "ref_s"
    for node in model.graph.node:
        for i, x in enumerate(node.input):
            if x == "style":
                node.input[i] = "ref_s"
    onnx.checker.check_model(model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="Output model.onnx path")
    ap.add_argument(
        "--hf-repo",
        default=_HF_REPO,
        help="Hugging Face repo id",
    )
    ap.add_argument(
        "--hf-file",
        default=_HF_FILE,
        help="Path within repo (default: onnx/model_quantized.onnx)",
    )
    ap.add_argument(
        "--backup",
        action="store_true",
        help="If --out exists, copy it to model.onnx.fp32.bak next to --out first",
    )
    args = ap.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    import onnx

    print("Downloading", args.hf_repo, args.hf_file, file=sys.stderr)
    src = Path(hf_hub_download(repo_id=args.hf_repo, filename=args.hf_file))
    model = onnx.load(str(src))
    patch_style_to_ref_s(model)

    if args.backup and out.is_file():
        bak = out.with_suffix(out.suffix + ".fp32.bak")
        shutil.copy2(out, bak)
        print("Backed up existing model to", bak, file=sys.stderr)

    onnx.save(model, str(out))
    sz_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({sz_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
