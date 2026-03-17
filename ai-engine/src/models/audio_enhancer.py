"""SpeechBrain Audio Enhancement — Noise suppression for athlete microphone audio.

Pretrained: speechbrain/sepformer-wham16k-enhancement (HuggingFace)
License: Apache-2.0
Task: Single-channel speech enhancement (16kHz)
"""
import numpy as np
from loguru import logger
from ..config import settings

class AudioEnhancer:
    """SpeechBrain-based audio enhancement for athlete MEMS microphone."""

    def __init__(self):
        self.model = None
        self.sample_rate = 16000

    async def load(self):
        try:
            from speechbrain.inference.separation import SepformerSeparation
            self.model = SepformerSeparation.from_hparams(
                source=settings.audio_model,
                savedir=f"{settings.model_cache_dir}/speechbrain",
            )
            logger.info("SpeechBrain audio enhancer loaded")
        except Exception as e:
            logger.warning("SpeechBrain not available: {} — audio pass-through", e)
            self.model = None

    async def unload(self):
        self.model = None

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """Enhance audio by removing stadium noise while preserving athlete speech.
        
        Args:
            audio: Input audio array (mono, 16kHz, float32)
        Returns:
            Enhanced audio array
        """
        if self.model is None:
            return audio

        import torch
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        enhanced = self.model.separate_batch(audio_tensor)
        return enhanced[0, :, 0].numpy()
