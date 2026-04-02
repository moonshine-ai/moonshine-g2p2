#!/usr/bin/env bash
# Export a Zeroth-prepared Lightning .ckpt to ONNX + voice JSON (default: work dir, not moonshine-tts/data).
# The bundled repo voice under moonshine-tts/data is MeloTTS (`ko_KR-melotts-medium`); use
# export_melotts_checkpoint_to_cpp.sh to refresh that. This script remains for Zeroth/LJS pipelines.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
CKPT="${1:-$ROOT/work/prepared_22050/lightning_logs/version_0/checkpoints/epoch=499-step=22000.ckpt}"
CFG="${2:-$ROOT/work/prepared_22050/config.json}"
OUT_DIR="${OUT_DIR:-$ROOT/work/piper_export_zeroth}"
OUT_ONNX="$OUT_DIR/ko_KR-zeroth-medium.onnx"
OUT_JSON="$OUT_DIR/ko_KR-zeroth-medium.onnx.json"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT_DIR"
"$PYTHON" -m piper_train.export_onnx "$CKPT" "$OUT_ONNX"
"$PYTHON" << PY
import json
from pathlib import Path
cfg = json.loads(Path("$CFG").read_text(encoding="utf-8"))
cfg["audio"]["quality"] = "medium"
cfg["dataset"] = cfg.get("dataset", "zeroth_korean")
cfg["language"] = {
    "code": "ko_KR",
    "family": "ko",
    "region": "KR",
    "name_native": "한국어",
    "name_english": "Korean",
    "country_english": "Korea",
}
cfg.setdefault("espeak", {})["voice"] = "ko"
Path("$OUT_JSON").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Wrote", "$OUT_JSON")
PY
echo "Done: $OUT_ONNX"
echo "Copy into moonshine-tts/data/ko/piper-voices/ only if you intend to ship this checkpoint."
