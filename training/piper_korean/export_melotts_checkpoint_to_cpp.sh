#!/usr/bin/env bash
# Export a MeloTTS-trained Korean Lightning .ckpt to cpp/data/ko/piper-voices (ONNX + JSON)
# and symlink into data/ko/piper-voices for Python/tests.
#
# Usage:
#   ./export_melotts_checkpoint_to_cpp.sh [CKPT_PATH] [CONFIG_JSON]
# Defaults: latest checkpoint under prepared_melotts_22050/lightning_logs/version_*/checkpoints/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
PREP="$ROOT/work/prepared_melotts_22050"
CKPT="${1:-}"
CFG="${2:-$PREP/config.json}"
OUT_ONNX="$REPO/cpp/data/ko/piper-voices/ko_KR-melotts-medium.onnx"
OUT_JSON="$REPO/cpp/data/ko/piper-voices/ko_KR-melotts-medium.onnx.json"
DATA_VOICES="$REPO/data/ko/piper-voices"
PYTHON="${PYTHON:-python3}"

pick_latest_ckpt() {
  local best="" line t p
  while IFS= read -r line; do
    t="${line%% *}"
    p="${line#* }"
    best="$p"
  done < <(
    find "$PREP/lightning_logs" -type f \( -name 'last.ckpt' -o -name 'epoch=*.ckpt' \) \
      -printf '%T@ %p\n' 2>/dev/null | sort -n
  )
  printf '%s' "$best"
}

if [[ -z "$CKPT" ]]; then
  CKPT="$(pick_latest_ckpt)"
fi
if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
  echo "ERROR: No checkpoint found under $PREP/lightning_logs/*/checkpoints/" >&2
  echo "Training saves every --checkpoint-epochs (default 25) and, after restart, also last.ckpt each epoch." >&2
  echo "Pass CKPT path as first argument once a .ckpt exists." >&2
  exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: Missing config: $CFG" >&2
  exit 1
fi

echo "Using CKPT=$CKPT"
mkdir -p "$(dirname "$OUT_ONNX")" "$DATA_VOICES"
"$PYTHON" -m piper_train.export_onnx "$CKPT" "$OUT_ONNX"
"$PYTHON" << PY
import json
from pathlib import Path
cfg = json.loads(Path("$CFG").read_text(encoding="utf-8"))
cfg["audio"]["quality"] = "medium"
cfg["dataset"] = cfg.get("dataset", "melotts_synthetic_ko")
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

ln -sf "../../../cpp/data/ko/piper-voices/ko_KR-melotts-medium.onnx" \
  "$DATA_VOICES/ko_KR-melotts-medium.onnx"
ln -sf "../../../cpp/data/ko/piper-voices/ko_KR-melotts-medium.onnx.json" \
  "$DATA_VOICES/ko_KR-melotts-medium.onnx.json"
echo "Symlinked into $DATA_VOICES"
echo "Done: $OUT_ONNX"
