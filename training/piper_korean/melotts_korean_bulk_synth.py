#!/usr/bin/env python3
"""
Generate many hours of Korean speech with MeloTTS-Korean from streamed wiki text + casual lines.

**Sources**
  - Korean Wikipedia (streaming): ``wikimedia/wikipedia``, config ``20231101.ko`` (override with ``--wiki-config``).
  - Fixed + lightly templated **casual conversation** lines (informal tone, everyday situations).

**Target duration** is controlled by ``--target-hours`` (default **25**). Synthesis stops when the sum of
written WAV durations reaches that goal (measured with ``soundfile`` after each file).

**Output** (under ``--out-dir``)::

  wav/melob_000001.wav ...
  metadata.csv       # utterance_id|text  (Piper / LJSpeech style)
  manifest.jsonl     # one JSON object per line: id, text, duration_sec, source

**Resume**: if ``manifest.jsonl`` and ``wav/`` already exist, the script loads cumulative duration and the
highest utterance index and continues appending until the target is met.

**Install** (separate venv strongly recommended; see ``requirements-melotts-korean.txt``)::

  git clone https://github.com/myshell-ai/MeloTTS.git && cd MeloTTS && pip install -e .
  python -m unidic download
  pip install datasets soundfile

Synthetic MeloTTS audio is **not** a substitute for human recordings for speaker-specific vocoders;
use for experiments, augmentation, or tooling only. Wikipedia text is CC BY-SA; comply with license
when redistributing **text**; MeloTTS output is separate (check MeloTTS / model license).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Casual / conversational Korean lines (single-speaker friendly, varied length)
# ---------------------------------------------------------------------------

_CAUSAL_LINES: list[str] = [
    "어, 진짜?",
    "아 그래? 몰랐네.",
    "음… 그건 좀 애매한데.",
    "오늘 너무 피곤해.",
    "점심 뭐 먹을까?",
    "나 커피 한 잔만 마시고 갈게.",
    "지하철이 또 지연됐대.",
    "이번 주말에 시간 있어?",
    "주말에 영화 보러 갈래?",
    "그 얘기 들었어. 완전 웃겼대.",
    "잠깐만, 문자 하나 볼게.",
    "배고파. 근처에 맛집 없어?",
    "에어컨 좀 줄여줄래? 너무 추워.",
    "내일 몇 시에 만날까?",
    "늦어서 미안, 길이 막혔어.",
    "괜찮아, 천천히 와.",
    "이거 어디서 샀어? 예쁘다.",
    "가격이 생각보다 괜찮네.",
    "환불 가능한지 물어봐야겠다.",
    "오늘 회의 길어질 것 같아.",
    "퇴근하고 한강 갈까?",
    "비 온다던데 우산 챙겼어?",
    "덥다. 아이스 아메리카노로 할래.",
    "이번 달에 운동 세 번밖에 못 했어.",
    "헬스장 등록했는데 잘 안 가게 되네.",
    "부모님이 건강 챙기라고 하셔.",
    "고양이가 키보드 위에 앉아서 못 쳐.",
    "주말에 청소해야지. 집이 엉망이야.",
    "새 이어폰 샀는데 음질 진짜 좋다.",
    "배터리 거의 없다. 충전기 좀 빌려줘.",
    "와이파이 끊겼어. 재부팅 해볼게.",
    "그 드라마 끝났어? 스포일러 하지 마.",
    "한 episode만 더 보고 잘게.",
    "알람 여섯 개 맞춰놓고도 늦잠 잤어.",
    "아침에 커피 없으면 안 깨.",
    "택배 왔는지 확인해봐야겠다.",
    "문 앞에 놓아달라고 했는데 벨 눌렀대.",
    "이번에 이사 가면 동네 추천 좀 해줘.",
    "전세 대출 금리가 좀 내렸다더라.",
    "친구 결혼식 축의금 얼마가 적당하지?",
    "옷장 정리하다가 옛날 사진 나왔어.",
    "헤어진 지 벌써 반년이네.",
    "연락 안 한 지 오래됐지?",
    "그냥 솔직하게 말하는 게 나을 것 같아.",
    "기분 상하게 했다면 미안해.",
    "내가 너무 예민했나 봐.",
    "그건 좀 선 넘은 것 같은데.",
    "장난이었어, 진지하게 받아들이지 마.",
    "오늘 기분이 별로야. 말 걸지 마.",
    "같이 산책할래? 공기 좀 쐬고 싶어.",
    "약국 어디 있어? 감기 기운 있어.",
    "열은 안 나는데 목이 따끔거려.",
    "주말에 등산 가자고 했는데 비 예보야.",
    "예약 취소해야겠다. 수수료 나오려나.",
    "리뷰 보고 왔는데 기대 이하였어.",
    "직원 분이 친절해서 기분은 좋았어.",
    "주차 자리 없어서 한 바퀴 돌았어.",
    "네비가 이상한 길로 안내했어.",
    "길 물어볼까? 지도가 헷갈려.",
    "여기 분위기 조용해서 공부하기 좋다.",
    "소음 때문에 집중이 안 돼.",
    "이어플러그 끼고 자야겠다.",
    "룸메가 새벽에 들어와서 깼어.",
    "청소 돌리고 나갈게. 빨래 넣어뒀어.",
    "냉장고에 우유 없어. 마트 들를까?",
    "레시피 대로 했는데 좀 짜.",
    "불 조절이 어렵네. 타버릴 뻔.",
    "설거지는 내가 할게. 너는 쉬어.",
    "오늘은 그냥 배달 시킬래. 요리하기 싫어.",
    "쿠폰 있어? 할인 되면 좋겠다.",
    "포인트 적립됐어? 카드 찍었어.",
    "영수증은 필요 없어. 간단히 주세요.",
    "포장해 주세요. 여기서 안 먹어요.",
    "매운맛 단계 몇으로 할까?",
    "맵기 조절 가능한가요?",
    "알레르기 있는데 견과류 빼주실 수 있어요?",
    "채식 옵션 있어요?",
    "물 좀 더 주시겠어요?",
    "계산 분할할게요. 카드 둘이 나눠서요.",
    "팁 포함이에요, 별도예요?",
    "예약 이름으로 박민준이요.",
    "두 명이요. 창가 자리 가능할까요?",
    "조금만 기다려 주세요. 바로 준비할게요.",
    "다 됐어요. 천천히 오세요.",
    "문 잠갔는지 다시 확인해봐.",
    "가스 불 껐지? 불안하네.",
    "에어컨 끄고 나왔나?",
    "열쇠 어디 뒀지? 가방 안에 있나?",
    "지갑 두고 나온 줄 알고 식겁했어.",
    "핸드폰 충전기 집에 뒀다.",
    "회의 링크 보내줄게. 슬랙 봐봐.",
    "화면 공유 안 보여. 권한 다시 줄게.",
    "목소리 잘 안 들려. 마이크 확인해봐.",
    "오늘 안건은 이거 세 가지면 될 것 같아.",
    "다음 주까지 초안만 보내주면 돼.",
    "피드백 반영해서 수정할게.",
    "이 부분은 좀 더 구체적으로 써야겠다.",
    "레퍼런스 링크 첨부해뒀어.",
    "깃허브 이슈 열어놨어. 번호 알려줄게.",
    "빌드 깨졌대. 로그 같이 볼래?",
    "테스트 로컬에서는 통과했어.",
    "환경 변수 이름 바뀐 거 맞지?",
    "문서 업데이트는 내가 할게.",
    "번역은 톤 맞추는 게 제일 어렵다.",
    "이 표현은 구어체로 바꾸는 게 자연스러워.",
    "존댓말로 쓸지 반말로 쓸지 통일하자.",
    "농담이야, 기분 나빴어?",
    "진심으로 말한 거야.",
    "그건 좀 오해의 소지가 있을 것 같아.",
    "다시 한번만 설명해줄게.",
    "내 말은 그 뜻이 아니었어.",
    "알겠어. 그렇게 할게.",
    "좋아, 그럼 그날로 하자.",
    "약속 취소해야 할 것 같아. 미안.",
    "갑자기 일이 생겨서 못 갈 것 같아.",
    "다음에 밥이라도 살게.",
    "오랜만이다. 잘 지냈어?",
    "요즘 뭐하고 지내?",
    "회사는 어때? 적응했어?",
    "이직 생각 중이야. 좀 망설여져.",
    "연차 쓰고 쉬고 싶다.",
    "휴가 때 어디 갈지 아직 못 정했어.",
    "비행기 표 값이 너무 올랐어.",
    "호텔은 그냥 깔끔한 데로 잡았어.",
    "짐 줄이려고 캐리어 작은 거 가져갈래.",
    "공항 가는 길 막히면 늦겠다.",
    "체크인 시간 맞춰서 가자.",
    "짐 찾는 데 한참 걸렸어.",
    "입국 심사 금방 끝났어.",
    "환전은 현지에서 할까, 미리 할까?",
    "카드 수수료 나오는지 확인해봐.",
    "환율 보고 결정하자.",
    "날씨 앱 보니까 내일은 맑대.",
    "우산은 그냥 챙겨. 변덕스러워.",
    "선크림 발랐어? 자외선 강하대.",
    "모자 쓰고 나가. 머리 탄다.",
    "신발 불편하면 바꿔 신어.",
    "발 아프면 잠깐 앉아 쉬자.",
    "저기 벤치 있네. 잠깐 앉을래?",
    "사진 여기서 찍자. 배경 예쁘다.",
    "필터 없이 그냥 찍어도 나온다.",
    "동영상으로 찍을까? 추억 남기게.",
    "배터리 얼마 없다. 절약 모드 켤게.",
    "충전은 카페 가서 할 수 있을 것 같아.",
    "와이파이 비번이 뭐였더라?",
    "데이터 켜야겠다. 길 찾아야 해.",
    "지도상으로는 여기서 좌회전이래.",
    "길 잃었나? 위치 공유할게.",
    "택시 탈까? 걸으면 이십 분 걸려.",
    "버스가 더 빠를 수도 있어.",
    "환승 한 번이면 돼.",
    "막차 시간 확인해봐.",
    "늦으면 그냥 택시 타고 가.",
    "집에 도착하면 문자 줘.",
    "들어가서 연락해.",
    "오늘 재밌었어. 또 보자.",
    "다음에 시간 날 때 연락해.",
    "조심히 들어가.",
    "너도 잘 자.",
    "내일 봐.",
    "수고했어. 고생 많았어.",
]


def _templated_casual(rng: random.Random) -> list[str]:
    """Extra variety from light templates (still natural-ish Korean)."""
    names = ["민수", "지영", "현우", "서연", "준호", "유진"]
    places = ["강남", "홍대", "이태원", "부산", "제주", "성수"]
    foods = ["김치찌개", "비빔밥", "삼겹살", "초밥", "파스타", "치킨"]
    times = ["아침", "점심", "저녁", "주말", "내일", "모레"]
    out = []
    for _ in range(40):
        out.append(f"{rng.choice(names)}야, {rng.choice(times)}에 {rng.choice(places)}에서 만날래?")
        out.append(f"오늘은 {rng.choice(foods)}가 먹고 싶네.")
        out.append(f"{rng.choice(places)} 가본 적 있어? 거기 카페 괜찮대.")
    return out


def _clean_wiki_paragraph(p: str) -> str:
    p = re.sub(r"\{\{[^}]+\}\}", " ", p)
    p = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", p)
    p = re.sub(r"https?://\S+", " ", p)
    p = re.sub(r"={2,}.+?={2,}", " ", p)
    p = re.sub(r"'''?", " ", p)
    p = re.sub(r"<[^>]+>", " ", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。…!?])\s+", text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        for sub in re.split(r"(?<=[,，、])\s*", p):
            t = sub.strip()
            if t:
                out.append(t)
    return out


def _chunk_line(s: str, max_chars: int) -> list[str]:
    s = s.strip()
    if len(s) <= max_chars:
        return [s] if s else []
    chunks: list[str] = []
    start = 0
    while start < len(s):
        end = min(start + max_chars, len(s))
        piece = s[start:end]
        if end < len(s):
            cut = max(piece.rfind(" "), piece.rfind("，"), piece.rfind(","), piece.rfind("、"))
            if cut > max_chars // 3:
                piece = piece[:cut]
                end = start + cut
        piece = piece.strip()
        if piece:
            chunks.append(piece)
        start = end if end > start else start + max_chars
    return chunks


def _wiki_sentence_stream(
    *,
    wiki_config: str,
    min_chars: int,
    max_chars: int,
    skip_title_re: re.Pattern[str] | None,
) -> Iterator[tuple[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "Install datasets: pip install datasets\n" + str(e)
        ) from e

    ds = load_dataset(
        "wikimedia/wikipedia",
        wiki_config,
        split="train",
        streaming=True,
    )
    for row in ds:
        title = str(row.get("title", "") or "")
        if skip_title_re and skip_title_re.search(title):
            continue
        body = str(row.get("text", "") or "")
        for para in body.split("\n\n"):
            para = _clean_wiki_paragraph(para)
            if len(para) < min_chars:
                continue
            for sent in _split_sentences(para):
                if len(sent) < min_chars:
                    continue
                for chunk in _chunk_line(sent, max_chars):
                    if len(chunk) >= min_chars:
                        yield ("wiki", chunk)


def _interleave_casual(
    wiki_iter: Iterator[tuple[str, str]],
    casual_pool: list[str],
    rng: random.Random,
    inject_every: int,
) -> Iterator[tuple[str, str]]:
    """Every `inject_every` wiki lines, yield one shuffled casual line."""
    rng.shuffle(casual_pool)
    ci = 0
    n = 0
    for item in wiki_iter:
        if inject_every > 0 and n > 0 and n % inject_every == 0:
            yield ("casual", casual_pool[ci % len(casual_pool)])
            ci += 1
        yield item
        n += 1


def _load_progress(out_dir: Path) -> tuple[int, float, Path, Path]:
    """Return (next_index, total_seconds_so_far, meta_path, manifest_path)."""
    wav_dir = out_dir / "wav"
    meta_path = out_dir / "metadata.csv"
    manifest_path = out_dir / "manifest.jsonl"
    wav_dir.mkdir(parents=True, exist_ok=True)
    next_idx = 1
    total_sec = 0.0
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_sec += float(o.get("duration_sec", 0.0))
                m = re.match(r"melob_(\d+)$", str(o.get("id", "")))
                if m:
                    next_idx = max(next_idx, int(m.group(1)) + 1)
    return next_idx, total_sec, meta_path, manifest_path


def main() -> int:
    ap = argparse.ArgumentParser(description="MeloTTS bulk Korean synth: wiki + casual → target hours")
    ap.add_argument("--out-dir", type=Path, default=Path("work/melotts_bulk_25h"))
    ap.add_argument("--target-hours", type=float, default=25.0)
    ap.add_argument("--wiki-config", default="20231101.ko", help="wikimedia/wikipedia config name")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=260)
    ap.add_argument("--inject-casual-every", type=int, default=12, help="Insert a casual line every N wiki lines")
    ap.add_argument("--dry-run", action="store_true", help="Stream wiki only; print counts, no TTS")
    ap.add_argument(
        "--dry-run-max-lines",
        type=int,
        default=10_000,
        help="Stop dry run after this many text lines (avoids huge scans)",
    )
    ap.add_argument(
        "--skip-title-regex",
        default="",
        help="If set, skip articles whose title matches this regex (e.g. '(?i)위키|틀:')",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    casual_pool = list(_CAUSAL_LINES) + _templated_casual(rng)
    rng.shuffle(casual_pool)

    skip_re = re.compile(args.skip_title_regex) if args.skip_title_regex.strip() else None

    wiki_iter = _wiki_sentence_stream(
        wiki_config=args.wiki_config,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        skip_title_re=skip_re,
    )
    text_iter = _interleave_casual(wiki_iter, casual_pool, rng, args.inject_casual_every)

    if args.dry_run:
        n = 0
        w = c = 0
        cap = max(1, args.dry_run_max_lines)
        for src, _ in text_iter:
            n += 1
            if src == "wiki":
                w += 1
            else:
                c += 1
            if n >= cap:
                break
        print(f"Dry run: scanned {n} lines (wiki={w}, casual={c}); cap={cap}.")
        # PyArrow parquet streaming can abort() on interpreter shutdown; exit immediately.
        os._exit(0)

    try:
        from melo.api import TTS
    except ImportError as e:
        print(
            "MeloTTS not installed. See melotts_korean_synth_prototype.py docstring.\n" f"{e}",
            file=sys.stderr,
        )
        return 1

    try:
        import soundfile as sf
    except ImportError as e:
        print("pip install soundfile\n" + str(e), file=sys.stderr)
        return 1

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    next_idx, total_sec, meta_path, manifest_path = _load_progress(out_dir)
    target_sec = float(args.target_hours) * 3600.0

    if total_sec >= target_sec:
        print(f"Already at {total_sec/3600:.2f} h >= target {args.target_hours} h — nothing to do.")
        return 0

    print(
        f"Resuming: index {next_idx}, have {total_sec/3600:.3f} h / target {args.target_hours} h",
        flush=True,
    )

    model = TTS(language="KR", device=args.device)
    spk2id = model.hps.data.spk2id
    spk_id = spk2id["KR"]

    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_mode = "a" if meta_path.is_file() else "w"
    with open(meta_path, meta_mode, encoding="utf-8") as meta_f, open(
        manifest_path, "a", encoding="utf-8"
    ) as man_f:
        if meta_mode == "w":
            pass  # no header for LJSpeech
        t0 = time.perf_counter()
        synth_count = 0
        for source, text in text_iter:
            if total_sec >= target_sec:
                break
            utt = f"melob_{next_idx:08d}"
            wav_path = wav_dir / f"{utt}.wav"
            try:
                model.tts_to_file(text, spk_id, str(wav_path), speed=args.speed)
            except Exception as e:
                print(f"SKIP {utt}: {e}", file=sys.stderr, flush=True)
                next_idx += 1
                continue
            info = sf.info(wav_path)
            dur = float(info.duration)
            total_sec += dur
            synth_count += 1
            meta_f.write(f"{utt}|{text}\n")
            meta_f.flush()
            man_f.write(
                json.dumps(
                    {"id": utt, "text": text, "duration_sec": dur, "source": source},
                    ensure_ascii=False,
                )
                + "\n"
            )
            man_f.flush()
            if synth_count % 25 == 0 or total_sec >= target_sec:
                elapsed = time.perf_counter() - t0
                rate = synth_count / elapsed if elapsed > 0 else 0
                print(
                    f"[{synth_count}] {utt} +{dur:.1f}s → total {total_sec/3600:.3f} h "
                    f"({rate:.2f} utt/s, ~{total_sec/max(synth_count,1):.1f}s avg/utt)",
                    flush=True,
                )
            next_idx += 1

    print(f"Done. Total audio ≈ {total_sec/3600:.3f} h in {out_dir}")
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
