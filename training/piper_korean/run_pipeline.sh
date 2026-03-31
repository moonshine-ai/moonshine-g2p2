#!/usr/bin/env bash
# Korean Piper: Zeroth (CC BY 4.0) -> LJSpeech -> preprocess @ 22,050 Hz (Piper "medium" rate)
# -> fine-tune from a Chinese medium checkpoint + per-epoch WAV of a fixed Korean phrase.
#
# Prerequisites (Debian/Ubuntu-style):
#   sudo apt-get install -y espeak-ng ffmpeg libsndfile1
#
# Python: piper_train + pytorch-lightning 1.7.x (see TRAINING.md).
#   pip install 'pip==24.0' 'torchmetrics==0.11.4'
#   pip install cython 'piper-phonemize~=1.1.0'
#   (cd <piper>/src/python && bash build_monotonic_align.sh && pip install -e .)
#
# Base checkpoint (download once):
#   hf download rhasspy/piper-checkpoints --repo-type dataset \
#     --include 'zh/zh_CN/xiao_ya/medium/*.ckpt' \
#     --local-dir training/piper_korean/work/piper-checkpoints
#
# Reference: https://github.com/rhasspy/piper/blob/master/TRAINING.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="${REPO_ID:-kresnik/zeroth_korean}"
SPEAKER_ID="${SPEAKER_ID:-201}"
SAMPLE_RATE="${SAMPLE_RATE:-22050}"
MAX_UTTERANCES="${MAX_UTTERANCES:-0}"

LJSPEECH_DIR="${LJSPEECH_DIR:-$ROOT/data/ljspeech_ko_speaker_${SPEAKER_ID}}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/work/prepared_${SAMPLE_RATE}}"
MAX_EPOCHS="${MAX_EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-25}"
BASE_CKPT="${BASE_CKPT:-$ROOT/work/piper-checkpoints/zh/zh_CN/xiao_ya/medium/epoch=2803-step=437424.ckpt}"
EPOCH_AUDIO_DIR="${EPOCH_AUDIO_DIR:-$ROOT/work/fixed_phrase_epoch_audio}"
FIXED_KOREAN_TEXT="${FIXED_KOREAN_TEXT:-안녕하세요}"
PYTHON="${PYTHON:-python3}"

export ESPEAK_DATA_PATH="${ESPEAK_DATA_PATH:-}"

echo "==> 1) Export LJSpeech layout from Zeroth (HF: $REPO_ID, speaker $SPEAKER_ID)"
prep_args=( "$ROOT/prepare_ljspeech_from_zeroth.py" --repo-id "$REPO_ID" --out-dir "$LJSPEECH_DIR" --speaker-id "$SPEAKER_ID" )
if [[ "$MAX_UTTERANCES" != "0" ]]; then
  prep_args+=( --max-utterances "$MAX_UTTERANCES" )
fi
"$PYTHON" "${prep_args[@]}"

echo "==> 2) Piper preprocess (eSpeak voice: ko, ${SAMPLE_RATE} Hz)"
rm -rf "$PREPARED_DIR"
"$PYTHON" -m piper_train.preprocess \
  --language ko \
  --sample-rate "$SAMPLE_RATE" \
  --input-dir "$LJSPEECH_DIR" \
  --output-dir "$PREPARED_DIR" \
  --dataset-format ljspeech \
  --single-speaker \
  --dataset-name zeroth_korean \
  --audio-quality "medium_${SAMPLE_RATE}"

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "ERROR: Base checkpoint not found: $BASE_CKPT"
  echo "Download with huggingface-cli (see comments at top of this script)."
  exit 1
fi

echo "==> 3) Fine-tune from Chinese medium weights (weights-only init) + epoch phrase audio -> $EPOCH_AUDIO_DIR"
export PYTHONUNBUFFERED=1
cd "$ROOT"
"$PYTHON" "$ROOT/train_korean_finetune.py" \
  --dataset-dir "$PREPARED_DIR" \
  --accelerator gpu \
  --devices 1 \
  --batch-size "$BATCH_SIZE" \
  --validation-split 0.05 \
  --num-test-examples 10 \
  --max_epochs "$MAX_EPOCHS" \
  --checkpoint-epochs "$CHECKPOINT_EPOCHS" \
  --precision 32 \
  --quality medium \
  --max-phoneme-ids 520 \
  --base-checkpoint "$BASE_CKPT" \
  --epoch-audio-dir "$EPOCH_AUDIO_DIR" \
  --fixed-korean-text "$FIXED_KOREAN_TEXT"

echo "Done."
echo "  Checkpoints: $PREPARED_DIR/lightning_logs/version_0/checkpoints/"
echo "  Phrase WAVs: $EPOCH_AUDIO_DIR/epoch_NNNN.wav"
echo "  TensorBoard: tensorboard --logdir $PREPARED_DIR/lightning_logs"
