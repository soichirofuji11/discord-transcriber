"""faster-whisper transcription engine with rolling buffer and deduplication."""

import threading
import time
from typing import Callable

import numpy as np
from faster_whisper import WhisperModel

from config import Config


class Transcriber:
    def __init__(self, config: Config, on_text: Callable[[dict], None]):
        self.config = config
        self.on_text = on_text

        print(f"[Transcriber] Loading model '{config.model_size}' "
              f"(device={config.device}, compute_type={config.compute_type})...")
        self.model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )
        print("[Transcriber] Model loaded")

        self._buf = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._dirty = False          # True when add_audio() added new data
        self._last_t = 0.0           # last transcribe timestamp

        self._emitted: list[str] = []   # words already sent as "final"
        self._prev: list[str] = []      # words from the previous transcription
        self._prompt = ""               # initial_prompt for Whisper context

    # ------------------------------------------------------------------ #
    # Public API (called from the audio processing thread)
    # ------------------------------------------------------------------ #

    def add_audio(self, chunk: np.ndarray):
        with self._lock:
            self._buf = np.concatenate([self._buf, chunk])
            max_n = int(self.config.max_buffer_duration_sec * self.config.sample_rate)
            if len(self._buf) > max_n:
                self._buf = self._buf[-max_n:]
            self._dirty = True

    def transcribe(self, is_endpoint: bool = False):
        # Nothing new to process
        if not self._dirty and not is_endpoint:
            return

        # Respect interval for non-endpoint calls
        now = time.time()
        if not is_endpoint and now - self._last_t < self.config.transcribe_interval_sec:
            return
        self._last_t = now

        # Grab buffer snapshot
        with self._lock:
            if len(self._buf) < self.config.sample_rate * 0.5:
                return
            audio = self._buf.copy()
            self._dirty = False

        # --- Run inference ---
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False,
            initial_prompt=self._prompt if self.config.condition_on_previous_text else None,
            condition_on_previous_text=self.config.condition_on_previous_text,
        )
        words: list[str] = []
        for seg in segments:
            t = seg.text.strip()
            if t:
                words.extend(t.split())

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        audio_sec = round(len(audio) / self.config.sample_rate, 1)

        if not words:
            return

        meta = dict(
            language=info.language,
            language_probability=info.language_probability,
            latency_ms=elapsed_ms,
            audio_sec=audio_sec,
        )

        if is_endpoint:
            self._handle_endpoint(words, meta)
        else:
            self._handle_partial(words, meta)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _handle_endpoint(self, words: list[str], meta: dict):
        """Emit the full utterance as one final line, then reset."""
        if words:
            self.on_text({"type": "final", "text": " ".join(words), **meta})

        # Prepare for next utterance
        self._prompt = " ".join(words)[-500:]
        self._emitted = []
        self._prev = []
        with self._lock:
            self._buf = np.array([], dtype=np.float32)
            self._dirty = False

    def _handle_partial(self, words: list[str], meta: dict):
        """Update partial display. Use local agreement to stabilize the prefix."""
        agreed_n = _agree_count(self._prev, words)
        self._prev = words

        # Track confirmed words internally (for stable partial display)
        if agreed_n > len(self._emitted):
            self._emitted = words[:agreed_n]

        # Show full current transcription as partial
        self.on_text({"type": "partial", "text": " ".join(words), **meta})


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _norm(word: str) -> str:
    """Normalize a word for comparison: lowercase, strip trailing punctuation."""
    return word.lower().rstrip(".,!?;:'\"")


def _agree_count(a: list[str], b: list[str]) -> int:
    """Count how many words from the start agree (after normalization)."""
    n = 0
    for wa, wb in zip(a, b):
        if _norm(wa) == _norm(wb):
            n += 1
        else:
            break
    return n
