#!/usr/bin/env python3
"""
Prototype: batch Korean WAV synthesis with MeloTTS-Korean for synthetic data experiments.

Uses MyShell ``MeloTTS`` with ``language='KR'`` (weights: https://huggingface.co/myshell-ai/MeloTTS-Korean).
Model / library are MIT per the HF card; confirm before redistribution.

**Install** (recommended: **separate venv** — MeloTTS pins older ``torch`` / ``transformers`` than ``piper_train``):

  git clone https://github.com/myshell-ai/MeloTTS.git
  cd MeloTTS && pip install -e .
  python -m unidic download

The PyPI package ``melotts`` is currently broken (missing ``requirements.txt`` in the sdist).

**Output**: LJSpeech-like layout under ``--out-dir``::

  wav/melok_000001.wav ...
  metadata.csv   (utterance_id|text)

You can point ``piper_train.preprocess`` at this tree with ``--dataset-format ljspeech`` if you want
to phonemize the text with eSpeak (MeloTTS phonology ≠ Piper/eSpeak, so treat as experimental).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _sanitize_id(raw: str) -> str:
    s = raw.strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:200] if len(s) > 200 else s


def _default_lines() -> list[str]:
    return [
        "안녕하세요.",
        "오늘 날씨가 참 좋습니다.",
        "합성 음성 데이터 프로토타입 테스트입니다.",
        "Project Gutenberg은 한국어 오디오북이 거의 없습니다.",
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="MeloTTS-Korean batch synth → LJSpeech-like folder")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("work/melotts_korean_proto"),
        help="Output root (wav/ + metadata.csv)",
    )
    p.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="UTF-8 text file, one sentence per line (empty lines skipped)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="cpu | cuda:0 | cuda | auto (MeloTTS passes through to PyTorch)",
    )
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--max-utterances", type=int, default=0, help="If >0, cap count after loading lines")
    p.add_argument("--prefix", default="melok", help="Utterance id prefix")
    args = p.parse_args()

    try:
        from melo.api import TTS
    except ImportError as e:
        print(
            "MeloTTS is not installed. Clone and install from GitHub:\n"
            "  git clone https://github.com/myshell-ai/MeloTTS.git && cd MeloTTS && pip install -e .\n"
            "  python -m unidic download\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 1

    if args.text_file:
        lines = [
            ln.strip()
            for ln in args.text_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    else:
        lines = _default_lines()

    if args.max_utterances and args.max_utterances > 0:
        lines = lines[: args.max_utterances]

    out: Path = args.out_dir.resolve()
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MeloTTS KR …", flush=True)
    model = TTS(language="KR", device=args.device)
    spk2id = model.hps.data.spk2id
    if "KR" not in spk2id:
        print(f"Unexpected speakers: {list(spk2id.keys())}", file=sys.stderr)
        return 1
    spk_id = spk2id["KR"]

    meta_path = out / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8") as meta:
        for i, text in enumerate(lines, start=1):
            utt = _sanitize_id(f"{args.prefix}_{i:06d}")
            wav_path = wav_dir / f"{utt}.wav"
            model.tts_to_file(text, spk_id, str(wav_path), speed=args.speed)
            meta.write(f"{utt}|{text}\n")
            print(f"  [{i}/{len(lines)}] {utt}.wav", flush=True)

    print(f"Wrote {len(lines)} utterances under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
