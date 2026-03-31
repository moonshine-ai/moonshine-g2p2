# Kokoro — bundled TTS for `moonshine_tts`

This directory is the **default Kokoro ONNX bundle** for the C++ CLI and `MoonshineTTS` (`builtin_kokoro_bundle_dir()` / `cpp/data/kokoro`). It must contain at least:

| Path | Role |
|------|------|
| `config.json` | Model config (includes phoneme `vocab` for ONNX). |
| `model.onnx` | Kokoro-82M acoustic model, FP32, exported with `disable_complex=True`. |
| `voices/*.kokorovoice` | Style tensors for ONNX inference (C++ cannot load Hugging Face `voices/*.pt` pickles). |

Optional in a **build tree** (not required under `cpp/data` if you only ship C++): `kokoro-v1_0.pth` (PyTorch weights, for re-exporting ONNX), `voices/*.pt` (source for `.kokorovoice`), `onnx_export_meta.json` (written by the download/export script).

## Provenance

| Asset | Source |
|--------|--------|
| **Weights, config, native voices (`voices/*.pt`)** | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on Hugging Face (see upstream **VOICES.md** for voice IDs and locales). |
| **`model.onnx`** | Produced locally by `scripts/download_kokoro_onnx.py`, which uses the **`kokoro`** Python package (`KModel`, `KModelForONNX`) and `torch.onnx.export`. Export uses **`disable_complex=True`** so the graph matches ONNX Runtime (real-valued STFT path). |
| **`*.kokorovoice`** | Produced by `scripts/export_kokoro_voice_for_cpp.py` from each `voices/*.pt` tensor pack. Format: magic `KVO1`, little-endian `uint32` rows/cols, row-major `float32` data (after squeezing singleton dims to shape `[N, 256]`). |

Python TTS in this repo (`speak.py`) can use the same HF bundle or PyTorch weights; the C++ path is **ONNX + `.kokorovoice` only**.

## Dependencies (rebuild)

```bash
pip install kokoro torch onnx onnxruntime onnxruntime-extensions huggingface_hub
```

Versions drift over time; if export fails, align with the `kokoro` release compatible with the checkpoint (see HF model card).

## Rebuild everything (recommended flow)

From the **repository root**:

1. **Download weights, config, voice `.pt` files, and export ONNX + `.kokorovoice`** into a staging directory (example: `models/kokoro`):

   ```bash
   python scripts/download_kokoro_onnx.py --out-dir models/kokoro --verify
   ```

   `--verify` runs a numeric parity check (PyTorch vs ONNX) and is **optional** but recommended after toolchain upgrades.

   To fetch only some voices (faster):

   ```bash
   python scripts/download_kokoro_onnx.py --out-dir models/kokoro --voices af_heart,jf_alpha
   ```

   To write **directly** into this bundle (overwrites same paths):

   ```bash
   python scripts/download_kokoro_onnx.py --out-dir cpp/data/kokoro --verify
   ```

   Use `--skip-kokorovoice-export` if you only want ONNX/config/`.pt` without regenerating `.kokorovoice`.

2. **Install into `cpp/data/kokoro`** if you built under `models/kokoro`:

   ```bash
   mkdir -p cpp/data/kokoro/voices
   cp -a models/kokoro/config.json models/kokoro/model.onnx cpp/data/kokoro/
   cp -a models/kokoro/voices/*.kokorovoice cpp/data/kokoro/voices/
   ```

## Rebuild only `.kokorovoice` files

If you already have `voices/*.pt` (from HF or a previous download) but need to refresh C++ sidecars:

```bash
python scripts/export_kokoro_voice_for_cpp.py --voices-dir cpp/data/kokoro/voices
# or
python scripts/export_kokoro_voice_for_cpp.py --voices-dir models/kokoro/voices
```

Single file:

```bash
python scripts/export_kokoro_voice_for_cpp.py \
  models/kokoro/voices/af_heart.pt \
  cpp/data/kokoro/voices/af_heart.kokorovoice
```

## Rebuild only `model.onnx` (weights unchanged)

```bash
python scripts/download_kokoro_onnx.py --out-dir models/kokoro --skip-download
```

Then copy `model.onnx` (and updated `onnx_export_meta.json` if present) into `cpp/data/kokoro/` as needed.

## Sanity checks after a rebuild

1. **Voice export (needs PyTorch + one `*.pt`):**

   ```bash
   python scripts/export_kokoro_voice_for_cpp.py \
     models/kokoro/voices/af_heart.pt /tmp/af_heart.kokorovoice
   python3 -c "import struct; d=open('/tmp/af_heart.kokorovoice','rb').read(12); assert d[:4]==b'KVO1'; print('ok', struct.unpack('<II', d[4:12]))"
   ```

   Expect: `ok (510, 256)` (rows/cols may match upstream; second dimension is style size).

2. **End-to-end TTS (needs built `moonshine_tts`, ONNX, voices, and `cpp/data` G2P assets):**

   ```bash
   cmake --build cpp/build --target moonshine_tts
   cpp/build/moonshine_tts --lang en_us -o /tmp/kokoro_smoke.wav --text "Hello"
   ```

   Expect: success message with non-zero sample count at 24000 Hz.

These checks were executed successfully against this repository (export from `models/kokoro/voices/af_heart.pt`, header `KVO1` + shape `(510, 256)`, and `moonshine_tts` WAV output).
