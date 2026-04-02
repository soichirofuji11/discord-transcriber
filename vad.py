import numpy as np
import torch
from silero_vad import load_silero_vad

from config import Config

# Silero VAD requires exactly 512 samples per call at 16kHz
_VAD_CHUNK_SIZE = 512


class VADProcessor:
    def __init__(self, config: Config):
        self.config = config
        print("[VAD] Loading Silero VAD model...")
        self.model = load_silero_vad(onnx=True)
        print("[VAD] Model loaded")
        self.is_speaking = False
        self.silence_samples = 0
        self._remainder = np.array([], dtype=np.float32)

    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.silence_samples = 0
        self._remainder = np.array([], dtype=np.float32)
        self.model.reset_states()

    def process(self, audio_chunk: np.ndarray) -> dict:
        """
        Process an audio chunk (any size) and return VAD result.
        Internally splits into 512-sample windows for Silero VAD.
        """
        audio = np.concatenate([self._remainder, audio_chunk])

        has_speech = False
        speech_prob = 0.0

        # Process all complete 512-sample windows
        pos = 0
        while pos + _VAD_CHUNK_SIZE <= len(audio):
            window = audio[pos : pos + _VAD_CHUNK_SIZE]
            tensor = torch.from_numpy(window).float()
            prob = self.model(tensor, self.config.sample_rate).item()
            if prob > speech_prob:
                speech_prob = prob
            if prob >= self.config.vad_threshold:
                has_speech = True
            pos += _VAD_CHUNK_SIZE

        # Save leftover samples for next call
        self._remainder = audio[pos:]

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
