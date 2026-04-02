import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

from config import Config


class VADProcessor:
    def __init__(self, config: Config):
        self.config = config
        print("[VAD] Loading Silero VAD model...")
        self.model = load_silero_vad(onnx=True)
        print("[VAD] Model loaded")
        self.is_speaking = False
        self.silence_samples = 0

    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.silence_samples = 0
        self.model.reset_states()

    def process(self, audio_chunk: np.ndarray) -> dict:
        """
        Process an audio chunk and return VAD result.

        Args:
            audio_chunk: float32 mono audio at config.sample_rate

        Returns:
            {
                "has_speech": bool,
                "speech_prob": float,
                "is_endpoint": bool,
            }
        """
        tensor = torch.from_numpy(audio_chunk).float()
        speech_prob = self.model(tensor, self.config.sample_rate).item()

        has_speech = speech_prob >= self.config.vad_threshold

        if has_speech:
            self.is_speaking = True
            self.silence_samples = 0
        else:
            self.silence_samples += len(audio_chunk)

        # Detect endpoint: speech was happening, then enough silence passed
        silence_ms = (self.silence_samples / self.config.sample_rate) * 1000
        is_endpoint = (
            self.is_speaking
            and silence_ms >= self.config.min_silence_duration_ms
        )

        if is_endpoint:
            self.is_speaking = False
            self.silence_samples = 0

        return {
            "has_speech": has_speech,
            "speech_prob": speech_prob,
            "is_endpoint": is_endpoint,
        }
