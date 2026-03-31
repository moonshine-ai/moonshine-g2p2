#!/usr/bin/env python3
"""
Export a single-speaker LJSpeech-style layout from Zeroth Korean (CC BY 4.0).

Primary source: OpenSLR SLR40 — https://openslr.org/40/
HF mirror used here: kresnik/zeroth_korean (includes speaker_id for filtering).

Output:
  <out_dir>/wav/<utterance_id>.wav
  <out_dir>/metadata.csv   (id|text per Piper ljspeech + --single-speaker)
"""
from __future__ import annotations

import argparse
import io
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files


def _sanitize_id(raw: str) -> str:
    s = raw.strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:200] if len(s) > 200 else s


def pick_top_speaker(repo_id: str) -> int:
    files = sorted(
        f for f in list_repo_files(repo_id, repo_type="dataset") if f.startswith("data/train-")
    )
    counts: Counter[int] = Counter()
    for f in files:
        path = hf_hub_download(repo_id, f, repo_type="dataset")
        col = pq.read_table(path, columns=["speaker_id"]).column(0)
        counts.update(col.to_pylist())
    speaker, n = counts.most_common(1)[0]
    return int(speaker)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-id",
        default="kresnik/zeroth_korean",
        help="Hugging Face dataset id (parquet shards with speaker_id).",
    )
    p.add_argument(
        "--speaker-id",
        type=int,
        default=None,
        help="Zeroth reader/speaker id. Default: speaker with the most train utterances.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="LJSpeech root (will contain wav/ and metadata.csv).",
    )
    p.add_argument(
        "--max-utterances",
        type=int,
        default=0,
        help="If > 0, export at most this many utterances (after shuffle by shard order).",
    )
    args = p.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    speaker_id = args.speaker_id
    if speaker_id is None:
        print("Counting utterances per speaker (train shards)...")
        speaker_id = pick_top_speaker(args.repo_id)
        print(f"Using speaker_id={speaker_id} (most train utterances).")

    files = sorted(
        f for f in list_repo_files(args.repo_id, repo_type="dataset") if f.startswith("data/train-")
    )
    meta_path = out_dir / "metadata.csv"
    n_written = 0
    with open(meta_path, "w", encoding="utf-8") as meta:
        for shard in files:
            if args.max_utterances and n_written >= args.max_utterances:
                break
            path = hf_hub_download(args.repo_id, shard, repo_type="dataset")
            tbl = pq.read_table(path)
            ids = tbl.column("id").to_pylist()
            spk = tbl.column("speaker_id").to_pylist()
            texts = tbl.column("text").to_pylist()
            audio_col = tbl.column("audio")
            for i in range(tbl.num_rows):
                if args.max_utterances and n_written >= args.max_utterances:
                    break
                if int(spk[i]) != int(speaker_id):
                    continue
                utt_id = _sanitize_id(str(ids[i]))
                text = str(texts[i]).strip()
                if not text:
                    continue
                struct = audio_col[i]
                blob = struct["bytes"].as_py()
                if not blob:
                    continue
                buf = io.BytesIO(blob)
                data, sr = sf.read(buf, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                out_wav = wav_dir / f"{utt_id}.wav"
                sf.write(out_wav, data, sr, subtype="PCM_16")
                meta.write(f"{utt_id}|{text}\n")
                n_written += 1

    license_path = out_dir / "LICENSE_ZEROTH.txt"
    license_path.write_text(
        "Zeroth Korean (OpenSLR SLR40) — CC BY 4.0\n"
        "https://openslr.org/40/\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        "Attribution: Zeroth project / Atlas Guide; see OpenSLR page for citation details.\n",
        encoding="utf-8",
    )
    print(f"Wrote {n_written} utterances under {out_dir}")


if __name__ == "__main__":
    main()
