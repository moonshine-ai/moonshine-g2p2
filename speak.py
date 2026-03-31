#!/usr/bin/env python3
"""
Synthesize speech with the Kokoro-82M vocoder (https://huggingface.co/hexgrad/Kokoro-82M)
using this repository's multilingual G2P, not Misaki text-to-phoneme.

Local ONNX bundle (from ``scripts/download_kokoro_onnx.py``)::

    python scripts/download_kokoro_onnx.py --out-dir models/kokoro --voices af_heart --verify
    python speak.py --kokoro-dir models/kokoro --backend onnx --lang en_us --text "Hello" -o hi.wav

Dependencies (install as needed)::

    pip install kokoro>=0.9.2 soundfile torch transformers huggingface_hub phonemizer espeakng_loader
    # ONNX inference: pip install onnxruntime onnx

Japanese and Chinese pipelines in ``kokoro`` pull in heavy optional stacks; this script only
needs a *quiet* pipeline for voice loading and :meth:`KPipeline.generate_from_tokens`, so it
uses a lightweight Spanish ``KPipeline`` shell while still loading any Kokoro voice (``jf_*``,
``zf_*``, etc.). The ONNX path avoids initializing ``KModel`` in PyTorch entirely.

Example::

    python speak.py --lang es --text "Hola mundo" --output hola.wav
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_KOKORO_DIR = _REPO_ROOT / "models" / "kokoro"

# (Kokoro pipeline lang code, default voice id, Moonshine G2P label).
# Default voices follow https://huggingface.co/hexgrad/Kokoro-82M (VOICES.md): one strong
# pick per locale (e.g. US → af_heart, UK → bf_emma, SIWIS for French).
_LANG_PROFILES: dict[str, tuple[str, str, str]] = {
    "en_us": ("a", "af_heart", "en_us"),
    "en-us": ("a", "af_heart", "en_us"),
    "en": ("a", "af_heart", "en_us"),
    "en_gb": ("b", "bf_emma", "en_gb"),
    "en-gb": ("b", "bf_emma", "en_gb"),
    "es": ("e", "ef_dora", "es"),
    "fr": ("f", "ff_siwis", "fr"),
    "hi": ("h", "hf_alpha", "hi"),
    "it": ("i", "if_sara", "it"),
    "pt_br": ("p", "pf_dora", "pt_br"),
    "pt-br": ("p", "pf_dora", "pt_br"),
    "pt": ("p", "pf_dora", "pt_br"),
    "ja": ("j", "jf_alpha", "ja"),
    "jp": ("j", "jf_alpha", "ja"),
    "zh": ("z", "zf_xiaobei", "zh"),
    "zh_hans": ("z", "zf_xiaobei", "zh"),
}

# Voice file name prefixes that match Kokoro’s training locale (see VOICES.md).
_VOICE_PREFIX_BY_KOKORO_LANG: dict[str, tuple[str, ...]] = {
    "a": ("af_", "am_"),
    "b": ("bf_", "bm_"),
    "e": ("ef_", "em_"),
    "f": ("ff_",),
    "h": ("hf_", "hm_"),
    "i": ("if_", "im_"),
    "p": ("pf_", "pm_"),
    "j": ("jf_", "jm_"),
    "z": ("zf_", "zm_"),
}


def _select_voice(
    requested: str | None,
    *,
    kokoro_lang: str,
    default_voice: str,
    kokoro_dir: Path | None,
) -> str:
    """Use *default_voice* when omitted; fix locale mismatch; fall back if voice file missing."""
    v = (requested or "").strip() or default_voice
    prefixes = _VOICE_PREFIX_BY_KOKORO_LANG.get(kokoro_lang, ())
    if prefixes and not any(v.startswith(p) for p in prefixes):
        print(
            f"Note: voice {v!r} does not match --lang locale ({kokoro_lang!r}); "
            f"using {default_voice!r} instead.",
            file=sys.stderr,
        )
        v = default_voice
    if kokoro_dir is not None:
        vf = kokoro_dir / "voices" / f"{v}.pt"
        if not vf.is_file():
            print(
                f"Note: missing {vf.name}; using {default_voice!r}.",
                file=sys.stderr,
            )
            v = default_voice
    return v

# Misaki / Kokoro espeak-style mappings for IPA that our G2P emits (no ^ tie bars).
_DIPHTHONGS_AFFRICATES: list[tuple[str, str]] = [
    ("t͡ʃ", "ʧ"),
    ("d͡ʒ", "ʤ"),
    ("tʃ", "ʧ"),
    ("dʒ", "ʤ"),
    ("eɪ", "A"),
    ("aɪ", "I"),
    ("aʊ", "W"),
    ("oʊ", "O"),
    ("əʊ", "Q"),
    ("ɔɪ", "Y"),
    ("ɝ", "ɜɹ"),
    ("ɚ", "əɹ"),
]

_REPO_ID = "hexgrad/Kokoro-82M"
# Any Kokoro pipeline whose __init__ does not require spaCy/MeCab; used only for load_voice + generate_from_tokens.
_INFER_PIPELINE_LANG = "e"


def _load_kokoro_config(*, kokoro_dir: Path | None) -> dict:
    if kokoro_dir is not None:
        p = Path(kokoro_dir) / "config.json"
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(repo_id=_REPO_ID, filename="config.json")
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _load_kokoro_vocab(*, kokoro_dir: Path | None) -> frozenset[str]:
    return frozenset(_load_kokoro_config(kokoro_dir=kokoro_dir)["vocab"])


def _phoneme_str_to_input_ids(phonemes: str, vocab: dict[str, int]) -> list[int]:
    return [0] + [i for i in (vocab.get(c) for c in phonemes) if i is not None] + [0]


def _load_voice_pack(kokoro_dir: Path, voice: str) -> "object":
    import torch

    vpath = kokoro_dir / "voices" / f"{voice}.pt"
    if not vpath.is_file():
        raise FileNotFoundError(f"Voice not found: {vpath}")
    return torch.load(vpath, map_location="cpu", weights_only=True)


def _synthesize_chunks_onnx(
    phoneme_chunks: list[str],
    *,
    onnx_path: Path,
    kokoro_dir: Path,
    voice: str,
    speed: float,
    vocab: dict[str, int],
    ort_providers: list[str] | None,
) -> list[float]:
    import numpy as np
    import onnxruntime as ort

    providers = ort_providers
    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    pack = _load_voice_pack(kokoro_dir, voice)
    wave_parts: list[float] = []
    for piece in phoneme_chunks:
        if not piece.strip():
            continue
        ids = _phoneme_str_to_input_ids(piece, vocab)
        if len(ids) > 512:
            raise ValueError("phoneme token sequence too long for Kokoro")
        input_ids = np.array([ids], dtype=np.int64)
        idx = min(len(piece) - 1, pack.shape[0] - 1) if len(piece) > 0 else 0
        ref = pack[idx]
        while ref.dim() > 1:
            ref = ref.squeeze(0)
        ref_s = ref.unsqueeze(0).numpy().astype(np.float32)
        w, _ = sess.run(
            None,
            {
                "input_ids": input_ids,
                "ref_s": ref_s,
                "speed": np.array(speed, dtype=np.float64),
            },
        )
        wave_parts.extend(np.asarray(w).astype(np.float64).ravel().tolist())
    return wave_parts


def _normalize_ipa_to_kokoro(ipa: str, *, kokoro_lang: str, vocab: frozenset[str]) -> str:
    s = unicodedata.normalize("NFC", ipa.strip())
    if kokoro_lang in "ab":
        for old, new in _DIPHTHONGS_AFFRICATES:
            s = s.replace(old, new)
    else:
        for old, new in _DIPHTHONGS_AFFRICATES:
            if old in ("ɝ", "ɚ"):
                continue
            s = s.replace(old, new)
    if kokoro_lang == "h":
        s = s.replace(".", "")
        s = s.replace("t̪", "t").replace("d̪", "d")
    s = "".join(ch for ch in s if ch in vocab or ch.isspace())
    return " ".join(s.split())


def _chunk_phonemes(ps: str, max_len: int = 510) -> list[str]:
    if len(ps) <= max_len:
        return [ps] if ps else []
    chunks: list[str] = []
    rest = ps
    while rest:
        if len(rest) <= max_len:
            chunks.append(rest.strip())
            break
        window = rest[: max_len + 1]
        cut = window.rfind(" ")
        if cut <= 0:
            cut = max_len
        piece = rest[:cut].strip()
        if piece:
            chunks.append(piece)
        rest = rest[cut:].lstrip()
    return [c for c in chunks if c]


def _english_text_to_ipa(text: str) -> str:
    from cmudict_ipa import normalize_word_for_lookup
    from english_rule_g2p import EnglishLexiconRuleG2p

    g2p = EnglishLexiconRuleG2p.from_default_paths()
    parts: list[str] = []
    for m in re.finditer(r"\S+", text):
        if not normalize_word_for_lookup(m.group(0)):
            continue
        ipa = g2p.g2p_span(text, m.start(), m.end())
        if ipa:
            parts.append(ipa)
    return " ".join(parts)


def _g2p_dispatch(moonshine_lang: str) -> Callable[[str], str]:
    if moonshine_lang == "en_us":
        return _english_text_to_ipa
    if moonshine_lang == "en_gb":
        # Lexicon is US-oriented; same engine, Kokoro side uses British voices.
        return _english_text_to_ipa
    if moonshine_lang == "es":
        import spanish_rule_g2p as m

        return m.text_to_ipa
    if moonshine_lang == "fr":
        import french_g2p as m

        return m.text_to_ipa
    if moonshine_lang == "hi":
        import hindi_rule_g2p as m

        return m.text_to_ipa
    if moonshine_lang == "it":
        import italian_rule_g2p as m

        return m.text_to_ipa
    if moonshine_lang == "pt_br":
        import portuguese_rule_g2p as m

        return lambda t: m.text_to_ipa(t, variant="pt_br")
    if moonshine_lang == "ja":
        from japanese_onnx_g2p import text_to_ipa

        return text_to_ipa
    if moonshine_lang == "zh":
        from chinese_rule_g2p import ChineseOnnxLexiconG2p

        g = ChineseOnnxLexiconG2p()
        return g.sentence_to_ipa
    raise KeyError(moonshine_lang)


def _resolve_lang(user: str) -> tuple[str, str, str]:
    k = user.strip().lower().replace(" ", "_")
    if k in _LANG_PROFILES:
        kokoro_lang, voice, moonshine = _LANG_PROFILES[k]
        return kokoro_lang, voice, moonshine
    raise SystemExit(
        f"Unknown --lang {user!r}. Supported: {', '.join(sorted(set(_LANG_PROFILES.keys())))}"
    )


def _pick_onnx_path(kokoro_dir: Path) -> Path | None:
    """Prefer FP32 ONNX only; INT8 builds are experimental and harm quality."""
    fp32 = kokoro_dir / "model.onnx"
    return fp32 if fp32.is_file() else None


def synthesize_wav(
    text: str,
    *,
    lang: str,
    output: Path,
    voice: str | None = None,
    speed: float = 1.0,
    device: str | None = None,
    repo_id: str = _REPO_ID,
    kokoro_dir: Path | None = None,
    backend: str = "auto",
    ort_providers: list[str] | None = None,
) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KModel, KPipeline

    env_dir = os.environ.get("KOKORO_DIR", "").strip()
    if kokoro_dir is not None:
        kdir = Path(kokoro_dir)
    elif env_dir:
        kdir = Path(env_dir)
    else:
        kdir = _DEFAULT_KOKORO_DIR
    if not kdir.is_dir():
        kdir = None

    kokoro_lang, default_voice, moonshine_lang = _resolve_lang(lang)
    voice = _select_voice(
        voice, kokoro_lang=kokoro_lang, default_voice=default_voice, kokoro_dir=kdir
    )
    g2p_fn = _g2p_dispatch(moonshine_lang)
    ipa = g2p_fn(text)
    if not ipa.strip():
        raise SystemExit("G2P returned empty IPA; check input text and language.")

    cfg = _load_kokoro_config(kokoro_dir=kdir)
    vocab_chars = frozenset(cfg["vocab"])
    vocab_map: dict[str, int] = cfg["vocab"]
    phonemes = _normalize_ipa_to_kokoro(ipa, kokoro_lang=kokoro_lang, vocab=vocab_chars)
    if not phonemes:
        raise SystemExit("After Kokoro vocabulary filtering, phoneme string is empty.")

    chunks = _chunk_phonemes(phonemes)
    if not chunks:
        raise SystemExit("No phoneme chunks to synthesize.")

    onnx_file: Path | None = None
    if backend == "onnx":
        if kdir is None:
            raise SystemExit("--backend onnx requires --kokoro-dir (or KOKORO_DIR) with model.onnx")
        onnx_file = kdir / "model.onnx"
        if not onnx_file.is_file():
            raise SystemExit(f"Missing {onnx_file} (run scripts/download_kokoro_onnx.py)")
    elif backend == "auto":
        if kdir is not None:
            onnx_file = _pick_onnx_path(kdir)

    sample_rate = 24_000
    wave_parts: list[float]

    if onnx_file is not None:
        assert kdir is not None
        wave_parts = _synthesize_chunks_onnx(
            chunks,
            onnx_path=onnx_file,
            kokoro_dir=kdir,
            voice=voice,
            speed=speed,
            vocab=vocab_map,
            ort_providers=ort_providers,
        )
    else:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        weights_path = None
        if kdir is not None:
            wp = kdir / "kokoro-v1_0.pth"
            if wp.is_file():
                weights_path = str(wp)
        use_local_bundle = weights_path is not None
        model = KModel(
            repo_id=repo_id,
            config=cfg if use_local_bundle else None,
            model=weights_path,
            disable_complex=use_local_bundle,
        ).to(device).eval()
        pipeline = KPipeline(
            lang_code=_INFER_PIPELINE_LANG,
            model=model,
            repo_id=repo_id,
            device=device,
        )
        if kdir is not None and (kdir / "voices").is_dir():
            import torch as _torch

            vfile = kdir / "voices" / f"{voice}.pt"
            if vfile.is_file():
                pack = _torch.load(vfile, map_location=device, weights_only=True)
                pipeline.voices[voice] = pack

        wave_parts = []
        for piece in chunks:
            for result in pipeline.generate_from_tokens(piece, voice=voice, speed=speed):
                if result.audio is None:
                    continue
                wave_parts.extend(result.audio.numpy().tolist())

    if not wave_parts:
        raise SystemExit("Kokoro produced no audio.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(wave_parts, dtype=np.float32)
    sf.write(str(output), audio, sample_rate)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", required=True, help="Language tag (e.g. en_us, es, ja, zh)")
    p.add_argument("--text", default="", help="UTF-8 text to speak")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("out.wav"),
        help="Output .wav path (default: out.wav)",
    )
    p.add_argument(
        "--voice",
        default=None,
        help=(
            "Kokoro voice id (default: picked from --lang, e.g. es→ef_dora, ja→jf_alpha). "
            "Must match locale prefix (ef_/em_ for Spanish, …)."
        ),
    )
    p.add_argument("--speed", type=float, default=1.0, help="Speed factor (default: 1)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--repo-id", default=_REPO_ID, help="Hugging Face repo for Kokoro weights")
    p.add_argument(
        "--kokoro-dir",
        type=Path,
        default=None,
        help=f"Local bundle (config, weights, model.onnx, voices/). Default: {_DEFAULT_KOKORO_DIR} or KOKORO_DIR",
    )
    p.add_argument(
        "--backend",
        choices=("auto", "pytorch", "onnx"),
        default="auto",
        help="auto: ONNX if model.onnx exists under kokoro-dir, else PyTorch",
    )
    p.add_argument(
        "--ort-provider",
        action="append",
        dest="ort_providers",
        default=None,
        help="ONNX Runtime provider (repeatable), e.g. --ort-provider CUDAExecutionProvider",
    )
    p.add_argument("text_positional", nargs="*", help="Text if --text omitted")
    args = p.parse_args(argv)
    text = args.text.strip() or " ".join(args.text_positional).strip()
    if not text:
        p.print_help()
        print("\nError: provide --text or positional text.", file=sys.stderr)
        raise SystemExit(2)
    try:
        synthesize_wav(
            text,
            lang=args.lang,
            output=args.output,
            voice=args.voice,
            speed=args.speed,
            device=args.device,
            repo_id=args.repo_id,
            kokoro_dir=args.kokoro_dir,
            backend=args.backend,
            ort_providers=args.ort_providers,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"Missing data file or model: {e}") from e


if __name__ == "__main__":
    main()
