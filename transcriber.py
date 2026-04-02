import threading
import time
from typing import Callable

import numpy as np
from faster_whisper import WhisperModel

from config import Config


class Transcriber:
    def __init__(self, config: Config, on_text_callback: Callable[[dict], None]):
        self.config = config
        self.callback = on_text_callback
        print(f"[Transcriber] Loading model '{config.model_size}' "
              f"(device={config.device}, compute_type={config.compute_type})...")
        self.model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )
        print("[Transcriber] Model loaded")
        self.audio_buffer = np.array([], dtype=np.float32)
        self.previous_text = ""
        self.lock = threading.Lock()
        self.last_transcribe_time = 0.0

    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to the rolling buffer."""
        with self.lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            max_samples = int(
                self.config.max_buffer_duration_sec * self.config.sample_rate
            )
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]

    def transcribe(self, is_endpoint: bool = False):
        """Run transcription on the current buffer."""
        now = time.time()
        if not is_endpoint:
            if now - self.last_transcribe_time < self.config.transcribe_interval_sec:
                return
        self.last_transcribe_time = now

        with self.lock:
            if len(self.audio_buffer) < self.config.sample_rate * 0.5:
                return  # Skip if less than 0.5s
            audio = self.audio_buffer.copy()

        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False,
            initial_prompt=(
                self.previous_text
                if self.config.condition_on_previous_text
                else None
            ),
            condition_on_previous_text=self.config.condition_on_previous_text,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts)

        if full_text:
            self.callback(
                {
                    "type": "final" if is_endpoint else "partial",
                    "text": full_text,
                    "language": info.language,
                    "language_probability": info.language_probability,
                }
            )

            if is_endpoint:
                self.previous_text = full_text[-500:]
                context_samples = int(
                    self.config.context_duration_sec * self.config.sample_rate
                )
                with self.lock:
                    self.audio_buffer = self.audio_buffer[-context_samples:]
