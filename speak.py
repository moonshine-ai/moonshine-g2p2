#!/usr/bin/env python3
"""
Synthesize speech with **Kokoro-82M** or **Piper** (ONNX), optionally from local bundles.

**Kokoro** uses this repository's multilingual G2P → IPA → Kokoro phoneme ids (not Misaki).

**Piper** uses each model's built-in eSpeak phonemization (``phoneme_type`` in the ``.onnx.json``).
Download voices into ``data/<lang>/piper-voices/*.onnx`` via ``scripts/download_piper_voices_for_g2p.py``.
Use ``--piper-inference-backend onnxruntime`` to run synthesis with **ONNX Runtime** inside this
script (same phoneme pipeline as ``piper``; inference matches ``piper-tts`` on CPU—see
``tests/test_speak_piper_ort_parity.py``).
Both engines write **24 kHz** mono WAV; Piper native rates (often 22050 Hz) are linearly resampled
with NumPy.

**Default ``--engine auto``** picks Kokoro when the language has Kokoro voices and your Kokoro bundle
under ``--kokoro-dir`` / ``KOKORO_DIR`` / ``models/kokoro`` contains the default voice ``.pt`` plus
``model.onnx`` or ``kokoro-v1_0.pth``; if that bundle is incomplete but Piper ONNX files exist for
the same ``--lang``, Piper is used. With no local Kokoro directory, auto uses Kokoro via Hugging Face.
Languages without Kokoro (e.g. German, Dutch) use Piper when ``data/<lang>/piper-voices`` has models.

Local Kokoro bundle (from ``scripts/download_kokoro_onnx.py``)::

    python scripts/download_kokoro_onnx.py --out-dir models/kokoro --voices af_heart --verify
    python speak.py --engine kokoro --kokoro-dir models/kokoro --backend onnx --lang en_us --text "Hello" -o hi.wav

Piper example::

    python speak.py --engine piper --lang en_us --text "Hello world" -o hi.wav

Dependencies (install as needed)::

    pip install kokoro>=0.9.2 soundfile torch numpy transformers huggingface_hub phonemizer espeakng_loader
    pip install onnxruntime onnx
    # Piper engine: pip install piper-tts onnxruntime

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
_OUTPUT_SAMPLE_RATE = 24_000

# ``--engine piper``: normalized --lang key -> (data/<subdir>/piper-voices, default *.onnx basename).
_PIPER_LANG: dict[str, tuple[str, str]] = {
    "en_us": ("en_us", "en_US-lessac-medium.onnx"),
    "en-us": ("en_us", "en_US-lessac-medium.onnx"),
    "en": ("en_us", "en_US-lessac-medium.onnx"),
    "en_gb": ("en_gb", "en_GB-cori-medium.onnx"),
    "en-gb": ("en_gb", "en_GB-cori-medium.onnx"),
    "es": ("es_mx", "es_MX-ald-medium.onnx"),
    "es_mx": ("es_mx", "es_MX-ald-medium.onnx"),
    "es_es": ("es_es", "es_ES-davefx-medium.onnx"),
    "es_ar": ("es_ar", "es_AR-daniela-high.onnx"),
    "fr": ("fr", "fr_FR-siwis-medium.onnx"),
    "hi": ("hi", "hi_IN-pratham-medium.onnx"),
    "it": ("it", "it_IT-paola-medium.onnx"),
    "pt_br": ("pt_br", "pt_BR-cadu-medium.onnx"),
    "pt-br": ("pt_br", "pt_BR-cadu-medium.onnx"),
    "pt": ("pt_br", "pt_BR-cadu-medium.onnx"),
    "pt_pt": ("pt_pt", "pt_PT-tugão-medium.onnx"),
    "pt-pt": ("pt_pt", "pt_PT-tugão-medium.onnx"),
    "zh": ("zh_hans", "zh_CN-huayan-medium.onnx"),
    "zh_hans": ("zh_hans", "zh_CN-huayan-medium.onnx"),
    "ar_msa": ("ar_msa", "ar_JO-kareem-medium.onnx"),
    "ar": ("ar_msa", "ar_JO-kareem-medium.onnx"),
    "de": ("de", "de_DE-thorsten-medium.onnx"),
    "nl": ("nl", "nl_NL-mls-medium.onnx"),
    "ru": ("ru", "ru_RU-denis-medium.onnx"),
    "tr": ("tr", "tr_TR-dfki-medium.onnx"),
    "uk": ("uk", "uk_UA-ukrainian_tts-medium.onnx"),
    "vi": ("vi", "vi_VN-vais1000-medium.onnx"),
    "ko": ("ko", "ko_KR-melotts-medium.onnx"),
    "ko_kr": ("ko", "ko_KR-melotts-medium.onnx"),
    "korean": ("ko", "ko_KR-melotts-medium.onnx"),
}

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


def _normalize_lang_key(user: str) -> str:
    return user.strip().lower().replace(" ", "_")


def _resolve_lang(user: str) -> tuple[str, str, str]:
    k = _normalize_lang_key(user)
    if k in _LANG_PROFILES:
        kokoro_lang, voice, moonshine = _LANG_PROFILES[k]
        return kokoro_lang, voice, moonshine
    raise SystemExit(
        f"Unknown --lang {user!r}. Supported: {', '.join(sorted(set(_LANG_PROFILES.keys())))}"
    )


def _resolve_kokoro_bundle_dir(kokoro_dir: Path | None) -> Path | None:
    env_dir = os.environ.get("KOKORO_DIR", "").strip()
    if kokoro_dir is not None:
        kdir = Path(kokoro_dir)
    elif env_dir:
        kdir = Path(env_dir)
    else:
        kdir = _DEFAULT_KOKORO_DIR
    if not kdir.is_dir():
        return None
    return kdir


def _kokoro_local_bundle_usable(lang_key: str, kdir: Path) -> bool:
    """True if *kdir* has weights/onnx plus the default ``voices/*.pt`` for *lang_key*."""
    if lang_key not in _LANG_PROFILES:
        return False
    _, default_voice, _ = _LANG_PROFILES[lang_key]
    vpt = kdir / "voices" / f"{default_voice}.pt"
    if not vpt.is_file():
        return False
    return (kdir / "model.onnx").is_file() or (kdir / "kokoro-v1_0.pth").is_file()


def _piper_has_any_model(lang_key: str, piper_voices_dir: Path | None) -> bool:
    if lang_key not in _PIPER_LANG:
        return False
    data_subdir, _ = _PIPER_LANG[lang_key]
    if piper_voices_dir is not None:
        pdir = Path(piper_voices_dir)
    else:
        pdir = _REPO_ROOT / "data" / data_subdir / "piper-voices"
    return pdir.is_dir() and any(pdir.glob("*.onnx"))


def _resolve_auto_engine(lang: str, kokoro_dir: Path | None, piper_voices_dir: Path | None) -> str:
    """Prefer Kokoro when the locale has Kokoro voices and the bundle is usable (or use HF when no local dir)."""
    k = _normalize_lang_key(lang)
    in_kokoro = k in _LANG_PROFILES
    in_piper = k in _PIPER_LANG
    if not in_kokoro and not in_piper:
        all_tags = sorted(set(_LANG_PROFILES.keys()) | set(_PIPER_LANG.keys()))
        raise SystemExit(
            f"Unknown --lang {lang!r} for --engine auto. "
            f"Try one of: {', '.join(all_tags)}"
        )
    kdir = _resolve_kokoro_bundle_dir(kokoro_dir)

    if in_kokoro:
        if kdir is None:
            return "kokoro"
        if _kokoro_local_bundle_usable(k, kdir):
            return "kokoro"
        if in_piper and _piper_has_any_model(k, piper_voices_dir):
            return "piper"
        return "kokoro"

    return "piper"


def _resolve_piper_lang(user: str) -> tuple[str, str]:
    k = _normalize_lang_key(user)
    if k in ("ja", "jp"):
        raise SystemExit(
            "Piper: Japanese ONNX voices are not bundled here; use --engine kokoro for Japanese."
        )
    if k not in _PIPER_LANG:
        opts = ", ".join(sorted(set(_PIPER_LANG.keys())))
        raise SystemExit(f"Piper: unknown --lang {user!r}. Supported: {opts}")
    return _PIPER_LANG[k]


def _pick_piper_onnx(voices_dir: Path, voice: str | None, default_basename: str) -> Path:
    """Resolve ``*.onnx`` under *voices_dir*; *voice* may be a stem or full basename."""
    if not voices_dir.is_dir():
        raise FileNotFoundError(f"Piper voices directory not found: {voices_dir}")
    if voice and voice.strip():
        name = voice.strip()
        if not name.endswith(".onnx"):
            name = f"{name}.onnx"
        cand = voices_dir / name
        if cand.is_file():
            return cand
        print(
            f"Note: Piper model {cand.name!r} not found; using default {default_basename!r}.",
            file=sys.stderr,
        )
    d = voices_dir / default_basename
    if d.is_file():
        return d
    models = sorted(voices_dir.glob("*.onnx"))
    if not models:
        raise FileNotFoundError(f"No *.onnx models in {voices_dir} (run scripts/download_piper_voices_for_g2p.py)")
    print(f"Note: default {default_basename!r} missing; using {models[0].name!r}.", file=sys.stderr)
    return models[0]


def _resample_linear_1d(samples: "object", orig_sr: int, target_sr: int):
    """Resample mono float audio with NumPy linear interpolation only."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if orig_sr == target_sr or x.size == 0:
        return x.astype(np.float32)
    duration = (x.size - 1) / float(orig_sr)
    n_out = max(2, int(round(duration * target_sr)) + 1)
    t_old = np.linspace(0.0, duration, num=x.size, endpoint=True, dtype=np.float64)
    t_new = np.linspace(0.0, duration, num=n_out, endpoint=True, dtype=np.float64)
    y = np.interp(t_new, t_old, x)
    return y.astype(np.float32)


def _piper_use_cuda(device: str | None) -> bool:
    if device == "cuda":
        return True
    if device == "cpu":
        return False
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _piper_make_ort_session(onnx_path: Path, *, use_cuda: bool):
    """Create an ONNX Runtime session with the same provider choice as ``PiperVoice.load``."""
    import onnxruntime as ort

    if use_cuda:
        providers: list = [
            (
                "CUDAExecutionProvider",
                {"cudnn_conv_algo_search": "HEURISTIC"},
            )
        ]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), sess_options=ort.SessionOptions(), providers=providers)


def _piper_infer_phoneme_ids_to_audio(
    session: "object",
    piper_cfg: "object",
    phoneme_ids: list[int],
    *,
    length_scale: float | None,
    noise_scale: float | None,
    noise_w_scale: float | None,
    speaker_id: int | None = None,
):
    """Run Piper ONNX inference only (no normalize/volume); mirrors ``PiperVoice.phoneme_ids_to_audio`` ORT call."""
    import numpy as np

    ls = piper_cfg.length_scale if length_scale is None else length_scale
    ns = piper_cfg.noise_scale if noise_scale is None else noise_scale
    nw = piper_cfg.noise_w_scale if noise_w_scale is None else noise_w_scale

    phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
    phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
    scales = np.array([ns, ls, nw], dtype=np.float32)

    args: dict[str, object] = {
        "input": phoneme_ids_array,
        "input_lengths": phoneme_ids_lengths,
        "scales": scales,
    }

    sid = speaker_id
    if piper_cfg.num_speakers <= 1:
        sid = None
    if piper_cfg.num_speakers > 1 and sid is None:
        sid = 0
    if sid is not None:
        args["sid"] = np.array([sid], dtype=np.int64)

    result = session.run(None, args)
    audio = result[0].squeeze()
    return np.asarray(audio, dtype=np.float32)


def _piper_apply_synthesis_output_effects(audio: "object", syn: "object"):
    """Same post-processing as ``PiperVoice.synthesize`` after ``phoneme_ids_to_audio`` (normalize, volume, clip)."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if syn.normalize_audio:
        max_val = float(np.max(np.abs(audio)))
        if max_val < 1e-8:
            audio = np.zeros_like(audio)
        else:
            audio = (audio / max_val).astype(np.float32)
    if syn.volume != 1.0:
        audio = (audio * float(syn.volume)).astype(np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _piper_synthesis_config_for_speed(speed: float):
    """Match ``_synthesize_piper_audio`` / ``SynthesisConfig(length_scale=...)`` speed mapping."""
    from piper.config import SynthesisConfig

    sp = float(speed)
    if sp <= 0:
        sp = 1.0
    length_scale = 1.0 / max(0.25, min(4.0, sp))
    return SynthesisConfig(length_scale=length_scale)


def _synthesize_piper_audio(
    text: str,
    *,
    onnx_path: Path,
    speed: float,
    use_cuda: bool,
    inference_backend: str = "piper",
):
    """Return (mono float32 samples, native sample rate).

    ``inference_backend``:
      - ``piper`` — ``PiperVoice.synthesize`` (default).
      - ``onnxruntime`` — same phonemization + id mapping, ONNX Runtime ``session.run`` in this module.
    """
    import numpy as np
    from piper import PiperVoice

    if inference_backend == "piper":
        from piper.config import SynthesisConfig

        voice = PiperVoice.load(onnx_path, use_cuda=use_cuda)
        syn = _piper_synthesis_config_for_speed(speed)
        parts: list = []
        native_sr = voice.config.sample_rate
        for chunk in voice.synthesize(text, syn_config=syn):
            parts.append(np.asarray(chunk.audio_float_array, dtype=np.float32))
            native_sr = chunk.sample_rate
        if not parts:
            raise SystemExit("Piper produced no audio.")
        return np.concatenate(parts), int(native_sr)

    if inference_backend != "onnxruntime":
        raise ValueError(f"Unknown Piper inference backend: {inference_backend!r}")

    voice = PiperVoice.load(onnx_path, use_cuda=use_cuda)
    session = _piper_make_ort_session(onnx_path, use_cuda=use_cuda)
    syn = _piper_synthesis_config_for_speed(speed)
    ns = syn.noise_scale if syn.noise_scale is not None else voice.config.noise_scale
    ls = syn.length_scale if syn.length_scale is not None else voice.config.length_scale
    nw = syn.noise_w_scale if syn.noise_w_scale is not None else voice.config.noise_w_scale

    parts: list = []
    native_sr = int(voice.config.sample_rate)
    for phonemes in voice.phonemize(text):
        if not phonemes:
            continue
        phoneme_ids = voice.phonemes_to_ids(phonemes)
        audio = _piper_infer_phoneme_ids_to_audio(
            session,
            voice.config,
            phoneme_ids,
            length_scale=ls,
            noise_scale=ns,
            noise_w_scale=nw,
            speaker_id=syn.speaker_id,
        )
        audio = _piper_apply_synthesis_output_effects(audio, syn)
        parts.append(audio)
    if not parts:
        raise SystemExit("Piper produced no audio.")
    return np.concatenate(parts), native_sr


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
    engine: str = "auto",
    piper_voices_dir: Path | None = None,
    piper_inference_backend: str = "piper",
) -> None:
    import numpy as np
    import soundfile as sf

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    eff_engine = _resolve_auto_engine(lang, kokoro_dir, piper_voices_dir) if engine == "auto" else engine
    if eff_engine == "piper":
        data_subdir, default_onnx = _resolve_piper_lang(lang)
        if piper_voices_dir is not None:
            vdir = Path(piper_voices_dir)
        else:
            vdir = _REPO_ROOT / "data" / data_subdir / "piper-voices"
        onnx_p = _pick_piper_onnx(vdir, voice, default_onnx)
        use_cuda = _piper_use_cuda(device)
        audio, native_sr = _synthesize_piper_audio(
            text,
            onnx_path=onnx_p,
            speed=speed,
            use_cuda=use_cuda,
            inference_backend=piper_inference_backend,
        )
        if native_sr != _OUTPUT_SAMPLE_RATE:
            audio = _resample_linear_1d(audio, native_sr, _OUTPUT_SAMPLE_RATE)
        sf.write(str(output), audio, _OUTPUT_SAMPLE_RATE)
        return

    import torch
    from kokoro import KModel, KPipeline

    kdir = _resolve_kokoro_bundle_dir(kokoro_dir)

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

    sample_rate = _OUTPUT_SAMPLE_RATE
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

    audio = np.asarray(wave_parts, dtype=np.float32)
    sf.write(str(output), audio, sample_rate)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", required=True, help="Language tag (e.g. en_us, es, ja, zh)")
    p.add_argument(
        "--engine",
        choices=("auto", "kokoro", "piper"),
        default="auto",
        help=(
            "auto (default): Kokoro when the language has Kokoro voices and the local bundle is complete "
            "(or no local dir → Hugging Face); else Piper if models exist under data/<lang>/piper-voices. "
            "kokoro / piper force one backend."
        ),
    )
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
            "Kokoro: voice id (e.g. af_heart). Piper: ONNX basename or stem under --piper-voices-dir "
            "(e.g. en_US-ryan-high)."
        ),
    )
    p.add_argument("--speed", type=float, default=1.0, help="Speed factor (default: 1)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.add_argument("--repo-id", default=_REPO_ID, help="Hugging Face repo for Kokoro weights")
    p.add_argument(
        "--kokoro-dir",
        type=Path,
        default=None,
        help=f"Kokoro: local bundle (config, weights, model.onnx, voices/). Default: {_DEFAULT_KOKORO_DIR} or KOKORO_DIR",
    )
    p.add_argument(
        "--piper-voices-dir",
        type=Path,
        default=None,
        help="Piper: folder with *.onnx models (default: data/<lang>/piper-voices from --lang).",
    )
    p.add_argument(
        "--piper-inference-backend",
        choices=("piper", "onnxruntime"),
        default="piper",
        help=(
            "Piper only: piper = PiperVoice.synthesize (default); onnxruntime = ONNX Runtime session.run "
            "in speak.py (same eSpeak phonemization). CPU parity is tested in tests/test_speak_piper_ort_parity.py."
        ),
    )
    p.add_argument(
        "--backend",
        choices=("auto", "pytorch", "onnx"),
        default="auto",
        help="Kokoro only: auto uses ONNX if model.onnx exists under --kokoro-dir, else PyTorch.",
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
            engine=args.engine,
            piper_voices_dir=args.piper_voices_dir,
            piper_inference_backend=args.piper_inference_backend,
        )
    except FileNotFoundError as e:
        raise SystemExit(f"Missing data file or model: {e}") from e


if __name__ == "__main__":
    main()
