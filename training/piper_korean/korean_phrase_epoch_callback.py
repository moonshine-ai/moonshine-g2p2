"""
Lightning callback: synthesize a fixed Korean phrase at the end of each training epoch,
write a WAV under a stable directory, and log audio to TensorBoard when available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch

from piper_train.vits.utils import audio_float_to_int16
from piper_train.vits.wavfile import write as write_wav

_LOGGER = logging.getLogger(__name__)


class KoreanPhraseEpochCallback(pl.Callback):
    def __init__(
        self,
        phoneme_ids: List[int],
        audio_dir: Path,
        sample_rate: int,
        phrase_tag: str = "ko_fixed_phrase",
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_w: float = 0.8,
    ) -> None:
        super().__init__()
        self.phoneme_ids = list(phoneme_ids)
        self.audio_dir = Path(audio_dir)
        self.sample_rate = int(sample_rate)
        self.phrase_tag = phrase_tag
        self.scales = [noise_scale, length_scale, noise_w]

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = int(trainer.current_epoch)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.audio_dir / f"epoch_{epoch:04d}.wav"

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.no_grad():
                text = torch.LongTensor(self.phoneme_ids).unsqueeze(0).to(
                    pl_module.device
                )
                text_lengths = torch.LongTensor([len(self.phoneme_ids)]).to(
                    pl_module.device
                )
                audio = pl_module(text, text_lengths, self.scales, sid=None).detach()
        finally:
            if was_training:
                pl_module.train()

        # Match validation_step normalization in VitsModel for listenable levels
        audio = audio * (1.0 / max(0.01, abs(audio.max().item())))

        audio_i16 = audio_float_to_int16(audio.cpu().numpy())
        write_wav(str(out_path), self.sample_rate, audio_i16)
        _LOGGER.info("Epoch %s sample written to %s", epoch, out_path)

        logger = trainer.logger
        if logger is None:
            return
        exp = getattr(logger, "experiment", None)
        if exp is None or not hasattr(exp, "add_audio"):
            return
        try:
            tb_audio = audio.cpu().float()
            if tb_audio.dim() == 3:
                tb_audio = tb_audio.squeeze(0)
            if tb_audio.dim() == 1:
                tb_audio = tb_audio.unsqueeze(0)
            exp.add_audio(
                f"{self.phrase_tag}/wav",
                tb_audio,
                global_step=epoch,
                sample_rate=self.sample_rate,
            )
        except Exception:
            _LOGGER.exception("TensorBoard add_audio failed (WAV still saved on disk)")


def phoneme_ids_for_text_ko(text: str) -> List[int]:
    from piper_phonemize import phoneme_ids_espeak, phonemize_espeak

    sentences = phonemize_espeak(text, "ko")
    flat = [p for sent in sentences for p in sent]
    return phoneme_ids_espeak(flat)
