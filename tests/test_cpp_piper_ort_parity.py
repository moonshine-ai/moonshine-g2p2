"""C++ ``PiperTTS`` / ``piper_phoneme_infer`` vs ``speak.py`` ONNX Runtime path (same phoneme ids).

Requires a built ``moonshine-tts/build/piper_phoneme_infer``, ``piper-tts``, ``onnxruntime``, ``soundfile``,
and ``data/en_us/piper-voices/en_US-lessac-medium.onnx``.

End-to-end text differs (Moonshine G2P vs eSpeak); this test locks parity on the shared ORT +
normalization + 24 kHz resample path.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_CPP_BIN = _REPO / "moonshine-tts" / "build" / "piper_phoneme_infer"
_ONNX = _REPO / "data" / "en_us" / "piper-voices" / "en_US-lessac-medium.onnx"


def _try_speak():
    try:
        import speak

        return speak
    except ImportError as e:
        raise unittest.SkipTest(f"speak module: {e}") from e


def _try_piper():
    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        return PiperVoice, SynthesisConfig
    except ImportError as e:
        raise unittest.SkipTest(f"piper-tts: {e}") from e


def _try_sf():
    try:
        import soundfile as sf

        return sf
    except ImportError as e:
        raise unittest.SkipTest(f"soundfile: {e}") from e


def _python_ort_pipeline_wav(
    speak_mod,
    onnx_path: Path,
    phoneme_ids: list[int],
    speed: float,
    *,
    noise_scale: float | None = None,
    noise_w_scale: float | None = None,
) -> np.ndarray:
    """Mirror ``speak`` ORT + Piper post-effects + 24 kHz resample for one phonemize chunk.

    Piper ONNX is stochastic unless ``noise_scale`` and ``noise_w_scale`` are zero; pass ``0.0, 0.0`` for
    bit-stable cross-checks against C++ ONNX Runtime.
    """
    PiperVoice, _SynthesisConfig = _try_piper()
    voice = PiperVoice.load(onnx_path, use_cuda=False)
    session = speak_mod._piper_make_ort_session(onnx_path, use_cuda=False)
    syn = speak_mod._piper_synthesis_config_for_speed(speed)
    ns = (
        float(noise_scale)
        if noise_scale is not None
        else (syn.noise_scale if syn.noise_scale is not None else voice.config.noise_scale)
    )
    ls = syn.length_scale if syn.length_scale is not None else voice.config.length_scale
    nw = (
        float(noise_w_scale)
        if noise_w_scale is not None
        else (syn.noise_w_scale if syn.noise_w_scale is not None else voice.config.noise_w_scale)
    )

    audio = speak_mod._piper_infer_phoneme_ids_to_audio(
        session,
        voice.config,
        phoneme_ids,
        length_scale=ls,
        noise_scale=ns,
        noise_w_scale=nw,
        speaker_id=syn.speaker_id,
    )
    audio = speak_mod._piper_apply_synthesis_output_effects(audio, syn)
    native_sr = int(voice.config.sample_rate)
    if native_sr != speak_mod._OUTPUT_SAMPLE_RATE:
        audio = speak_mod._resample_linear_1d(audio, native_sr, speak_mod._OUTPUT_SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


class TestCppPiperOrtParity(unittest.TestCase):
    def setUp(self) -> None:
        if not _ONNX.is_file():
            self.skipTest("English Piper ONNX missing")
        if not _CPP_BIN.is_file():
            self.skipTest("piper_phoneme_infer not built (cmake --build moonshine-tts/build --target piper_phoneme_infer)")

    def test_cpp_matches_python_ort_for_piper_phoneme_ids(self) -> None:
        speak_mod = _try_speak()
        PiperVoice, _SynthesisConfig = _try_piper()
        sf = _try_sf()

        # Single eSpeak sentence so phonemize yields one chunk (same as one ORT call in both stacks).
        text = "Hello there."
        voice = PiperVoice.load(_ONNX, use_cuda=False)
        chunks = list(voice.phonemize(text))
        self.assertEqual(len(chunks), 1, msg="expected one phonemize chunk for test sentence")
        phoneme_ids = voice.phonemes_to_ids(chunks[0])
        speed = 1.2
        # Stochastic generator: default noise differs every run and across Python vs C++ ORT builds.
        py_wav = _python_ort_pipeline_wav(
            speak_mod, _ONNX, phoneme_ids, speed, noise_scale=0.0, noise_w_scale=0.0
        )

        with tempfile.TemporaryDirectory() as td:
            ids_path = Path(td) / "ids.json"
            out_wav = Path(td) / "cpp.wav"
            ids_path.write_text(json.dumps(phoneme_ids), encoding="utf-8")
            subprocess.run(
                [
                    str(_CPP_BIN),
                    "--onnx",
                    str(_ONNX),
                    "--ids-json",
                    str(ids_path),
                    "--lang",
                    "en_us",
                    "--speed",
                    str(speed),
                    "--noise-scale",
                    "0",
                    "--noise-w",
                    "0",
                    "-o",
                    str(out_wav),
                ],
                check=True,
                cwd=str(_REPO),
            )
            cpp_wav, sr = sf.read(str(out_wav), dtype="float32", always_2d=False)
            self.assertEqual(sr, speak_mod._OUTPUT_SAMPLE_RATE)

        self.assertEqual(py_wav.shape, cpp_wav.shape)
        max_abs = float(np.max(np.abs(py_wav - cpp_wav)))
        self.assertLess(max_abs, 5e-5, msg=f"max abs sample diff {max_abs}")


if __name__ == "__main__":
    unittest.main()
