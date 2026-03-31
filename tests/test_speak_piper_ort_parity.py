"""Parity between ``piper-tts`` PiperVoice inference and ``speak.py`` ONNX Runtime path (CPU).

Requires ``piper-tts``, ``onnxruntime``, and ``data/en_us/piper-voices/en_US-lessac-medium.onnx``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _try_imports():
    import speak  # noqa: F401

    try:
        from piper import PiperVoice
        from piper.config import SynthesisConfig
    except ImportError as e:
        raise unittest.SkipTest(f"piper-tts not installed: {e}") from e
    return speak, PiperVoice, SynthesisConfig


class TestSpeakPiperOrtParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.speak, cls.PiperVoice, cls.SynthesisConfig = _try_imports()
        cls.onnx_path = _REPO / "data" / "en_us" / "piper-voices" / "en_US-lessac-medium.onnx"

    def setUp(self) -> None:
        if not self.onnx_path.is_file():
            self.skipTest(
                "English Piper ONNX missing (run scripts/download_piper_voices_for_g2p.py or sync cpp/data)"
            )

    def test_phoneme_ids_raw_audio_matches_piper_voice_cpu(self) -> None:
        """ORT ``session.run`` inputs/outputs match ``PiperVoice.phoneme_ids_to_audio`` (no post-effects)."""
        voice = self.PiperVoice.load(self.onnx_path, use_cuda=False)
        session = self.speak._piper_make_ort_session(self.onnx_path, use_cuda=False)
        syn = self.SynthesisConfig(length_scale=1.0)
        ns = syn.noise_scale if syn.noise_scale is not None else voice.config.noise_scale
        ls = syn.length_scale if syn.length_scale is not None else voice.config.length_scale
        nw = syn.noise_w_scale if syn.noise_w_scale is not None else voice.config.noise_w_scale

        text = "Hello world."
        for phonemes in voice.phonemize(text):
            if not phonemes:
                continue
            ids = voice.phonemes_to_ids(phonemes)
            raw_pkg = voice.phoneme_ids_to_audio(ids, syn_config=syn)
            raw_ort = self.speak._piper_infer_phoneme_ids_to_audio(
                session,
                voice.config,
                ids,
                length_scale=ls,
                noise_scale=ns,
                noise_w_scale=nw,
                speaker_id=syn.speaker_id,
            )
            self.assertEqual(raw_pkg.dtype, raw_ort.dtype)
            self.assertEqual(raw_pkg.shape, raw_ort.shape)
            max_abs = float(np.max(np.abs(raw_pkg - raw_ort)))
            self.assertLess(
                max_abs,
                1e-4,
                msg=f"raw ORT vs PiperVoice phoneme_ids_to_audio max abs diff {max_abs}",
            )

    def test_full_synthesize_matches_with_post_effects_cpu(self) -> None:
        """End-to-end waveform: ``PiperVoice.synthesize`` vs ``speak._synthesize_piper_audio(..., onnxruntime)."""
        speed = 1.15
        a_pkg, sr_pkg = self.speak._synthesize_piper_audio(
            "Testing one two. Second sentence.",
            onnx_path=self.onnx_path,
            speed=speed,
            use_cuda=False,
            inference_backend="piper",
        )
        a_ort, sr_ort = self.speak._synthesize_piper_audio(
            "Testing one two. Second sentence.",
            onnx_path=self.onnx_path,
            speed=speed,
            use_cuda=False,
            inference_backend="onnxruntime",
        )
        self.assertEqual(sr_pkg, sr_ort)
        self.assertEqual(a_pkg.shape, a_ort.shape)
        np.testing.assert_allclose(a_pkg, a_ort, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
