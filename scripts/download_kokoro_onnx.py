#!/usr/bin/env python3
"""
Download Kokoro-82M PyTorch weights and voice packs from Hugging Face, export the
acoustic model to FP32 ONNX (voice ``.pt`` tensors stay as downloaded). After download,
all ``voices/*.pt`` are also exported to ``*.kokorovoice`` for C++ ``moonshine_tts`` unless
``--skip-kokorovoice-export`` is set.

ONNX export uses ``disable_complex=True`` (real-valued CustomSTFT path in ``kokoro``),
because the default complex STFT graph is not ONNX-exportable. Use the same flag for
PyTorch when comparing to ONNX (``speak.py`` does this automatically for local ``.pth``).

**INT8:** ONNX Runtime dynamic weight quantization (MatMul/Gemm) was tested on this graph
and breaks duration prediction, producing unrelated audio. The optional flag
``--experimental-int8`` still writes ``model.int8.onnx`` for research; default builds omit it.

Example::

    python scripts/download_kokoro_onnx.py --out-dir models/kokoro --verify

To match the C++ default bundle path, copy (or symlink) into ``cpp/data/kokoro``::

    cp -a models/kokoro/config.json models/kokoro/model.onnx models/kokoro/voices \\
        cpp/data/kokoro/
    # voices/ should contain ``*.kokorovoice`` (see ``export_kokoro_voice_for_cpp.py``).

Dependencies::

    pip install kokoro torch onnx onnxruntime onnxruntime-extensions
    # plus huggingface_hub; quantization uses onnxruntime.quantization
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "models" / "kokoro"
_REPO_ID = "hexgrad/Kokoro-82M"
_WEIGHTS_NAME = "kokoro-v1_0.pth"
_ONNX_NAME = "model.onnx"
_EXPORT_META = "onnx_export_meta.json"


def _download_assets(
    out_dir: Path,
    repo_id: str,
    *,
    voices_glob: str | None,
    voice_names: list[str] | None,
) -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    out_dir.mkdir(parents=True, exist_ok=True)
    voices_dir = out_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("config.json", _WEIGHTS_NAME):
        src = hf_hub_download(repo_id=repo_id, filename=fname)
        shutil.copy2(src, out_dir / fname)

    files = list_repo_files(repo_id)
    voice_files = sorted(f for f in files if f.startswith("voices/") and f.endswith(".pt"))
    if voice_names:
        want = {f"{n}.pt" if not n.endswith(".pt") else n for n in voice_names}
        voice_files = [f for f in voice_files if Path(f).name in want]
    if voices_glob:
        import fnmatch

        voice_files = [f for f in voice_files if fnmatch.fnmatch(f, voices_glob)]
    for vf in voice_files:
        name = Path(vf).name
        dst = voices_dir / name
        if dst.is_file():
            continue
        src = hf_hub_download(repo_id=repo_id, filename=vf)
        shutil.copy2(src, dst)
    return out_dir / _WEIGHTS_NAME


def _export_onnx(weights_path: Path, config_path: Path, onnx_path: Path, opset: int) -> None:
    import torch
    from kokoro.model import KModel, KModelForONNX

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    m = KModel(repo_id=_REPO_ID, config=config, model=str(weights_path), disable_complex=True)
    m.eval()
    for mod in m.modules():
        if mod.__class__.__name__ == "InstanceNorm1d":
            mod.eval()
    wrap = KModelForONNX(m)
    seq = min(64, m.context_length - 2)
    dummy_ids = torch.ones(1, seq, dtype=torch.long)
    dummy_ref = torch.randn(1, 256) * 0.05
    dummy_speed = 1.0
    torch.onnx.export(
        wrap,
        (dummy_ids, dummy_ref, dummy_speed),
        str(onnx_path),
        input_names=["input_ids", "ref_s", "speed"],
        output_names=["waveform", "pred_dur"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "ref_s": {0: "batch"},
            "waveform": {0: "audio_len"},
            "pred_dur": {0: "dur_len"},
        },
        opset_version=opset,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
    )
    meta = {
        "repo_id": _REPO_ID,
        "weights_file": _WEIGHTS_NAME,
        "disable_complex": True,
        "opset": opset,
        "notes": (
            "Exported with CustomSTFT (disable_complex=True). "
            "PyTorch reference for parity checks should use the same flag."
        ),
    }
    with open(onnx_path.parent / _EXPORT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _audio_metrics(a: object, b: object) -> dict[str, float]:
    import numpy as np

    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    err = x - y
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = float(np.sqrt(np.mean(x**2)) + 1e-12)
    rel_rmse = rmse / denom
    cor = float(np.corrcoef(x, y)[0, 1]) if n > 1 and np.std(x) > 0 and np.std(y) > 0 else 0.0
    snr = float(10.0 * np.log10((np.mean(x**2) + 1e-20) / (np.mean(err**2) + 1e-20)))
    return {"n": float(n), "rmse": rmse, "rel_rmse": rel_rmse, "correlation": cor, "snr_db": snr}


def _try_dynamic_int8(fp32_path: Path, int8_path: Path) -> bool:
    """
    ONNX Runtime dynamic INT8 for MatMul/Gemm was evaluated on Kokoro: it breaks
    duration prediction and yields uncorrelated audio. This helper is kept for
    experiments; it returns True only if the quantized graph runs without error.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    try:
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
    except Exception:
        return False
    return int8_path.is_file()


def verify_onnx(out_dir: Path, *, experimental_int8: Path | None = None) -> int:
    import numpy as np
    import onnxruntime as ort
    import torch
    from kokoro.model import KModel, KModelForONNX

    config_path = out_dir / "config.json"
    weights_path = out_dir / _WEIGHTS_NAME
    fp32_onnx = out_dir / _ONNX_NAME
    voice_path = out_dir / "voices" / "af_heart.pt"

    if not all(p.is_file() for p in (config_path, weights_path, fp32_onnx, voice_path)):
        print("verify: missing config, weights, model.onnx, or voices/af_heart.pt", file=sys.stderr)
        return 1

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    vocab: dict[str, int] = config["vocab"]
    inv = list(vocab.keys())
    chars = [c for c in inv if len(c) == 1 and not c.isspace()][:40]
    ps = "".join(chars[:24])
    input_ids_list = [0] + [vocab[c] for c in ps if c in vocab] + [0]
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    idx = min(len(ps) - 1, pack.shape[0] - 1) if len(ps) > 0 else 0
    ref_s = pack[idx].squeeze(0).clone()
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    speed = 1.0

    m = KModel(repo_id=_REPO_ID, config=config, model=str(weights_path), disable_complex=True)
    m.eval()
    wrap = KModelForONNX(m)
    with torch.no_grad():
        w_pt, _ = wrap(input_ids, ref_s, speed)
    w_pt_np = w_pt.cpu().numpy().ravel()

    def run_ort(path: Path) -> np.ndarray:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        w, _ = sess.run(
            None,
            {
                "input_ids": input_ids.numpy().astype(np.int64),
                "ref_s": ref_s.numpy().astype(np.float32),
                "speed": np.array(speed, dtype=np.float64),
            },
        )
        return np.asarray(w).ravel()

    w32 = run_ort(fp32_onnx)
    m32 = _audio_metrics(w_pt_np, w32)
    print("PyTorch (disable_complex=True) vs ONNX fp32:", m32)

    code = 0
    if m32["correlation"] < 0.98 or m32["rel_rmse"] > 0.12:
        print(
            "verify: fp32 ONNX deviates from PyTorch more than expected — export may be broken.",
            file=sys.stderr,
        )
        code = 1

    if experimental_int8 is not None and experimental_int8.is_file():
        w8 = run_ort(experimental_int8)
        m8 = _audio_metrics(w_pt_np, w8)
        print("PyTorch vs ONNX int8 (experimental):", m8)
        if m8["correlation"] < 0.9:
            print(
                "verify: INT8 audio diverges (known issue with ORT dynamic quant on this graph).",
                file=sys.stderr,
            )
            code = max(code, 2)
    return code


def _export_kokorovoice_sidecars(out_dir: Path) -> int:
    """Run ``export_kokoro_voice_for_cpp.py --voices-dir`` for C++ MoonshineTTS."""
    voices_dir = out_dir / "voices"
    if not voices_dir.is_dir():
        return 0
    if not any(voices_dir.glob("*.pt")):
        return 0
    script = Path(__file__).resolve().parent / "export_kokoro_voice_for_cpp.py"
    if not script.is_file():
        print(f"Missing {script}; skipping .kokorovoice export.", file=sys.stderr)
        return 0
    print("Exporting voices/*.kokorovoice for C++ (moonshine_tts) …")
    r = subprocess.run(
        [sys.executable, str(script), "--voices-dir", str(voices_dir)],
        check=False,
    )
    return int(r.returncode)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT, help="Output directory")
    ap.add_argument("--repo-id", default=_REPO_ID)
    ap.add_argument(
        "--voices-glob",
        default=None,
        help="Optional fnmatch under voices/ (default: all *.pt)",
    )
    ap.add_argument(
        "--voices",
        default=None,
        help="Comma-separated voice ids to fetch (e.g. af_heart,ef_dora). Default: all.",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-download", action="store_true", help="Reuse files in out-dir")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument(
        "--experimental-int8",
        action="store_true",
        help=(
            "Also write model.int8.onnx via ORT dynamic quantization (MatMul/Gemm). "
            "This currently breaks prosody; for research only."
        ),
    )
    ap.add_argument("--verify", action="store_true", help="Run numeric parity check after build")
    ap.add_argument(
        "--skip-kokorovoice-export",
        action="store_true",
        help="Do not run export_kokoro_voice_for_cpp.py on voices/ (C++ needs .kokorovoice separately).",
    )
    args = ap.parse_args(argv)

    out_dir = args.out_dir.resolve()
    if not args.skip_download:
        vnames = [x.strip() for x in args.voices.split(",")] if args.voices else None
        _download_assets(
            out_dir,
            args.repo_id,
            voices_glob=args.voices_glob,
            voice_names=vnames,
        )

    weights_path = out_dir / _WEIGHTS_NAME
    config_path = out_dir / "config.json"
    onnx_path = out_dir / _ONNX_NAME
    int8_path = out_dir / "model.int8.onnx"

    if not args.skip_export:
        print("Exporting ONNX to", onnx_path)
        _export_onnx(weights_path, config_path, onnx_path, args.opset)

    if args.experimental_int8:
        print("Writing experimental INT8 model to", int8_path)
        _try_dynamic_int8(onnx_path, int8_path)

    if args.verify:
        raise SystemExit(
            verify_onnx(out_dir, experimental_int8=int8_path if args.experimental_int8 else None)
        )

    if not args.skip_kokorovoice_export:
        kc = _export_kokorovoice_sidecars(out_dir)
        if kc != 0:
            print(
                "Warning: .kokorovoice export failed; run manually:\n  python scripts/export_kokoro_voice_for_cpp.py "
                f"--voices-dir {out_dir / 'voices'}",
                file=sys.stderr,
            )

    print("Done. Assets in:", out_dir)


if __name__ == "__main__":
    main()
