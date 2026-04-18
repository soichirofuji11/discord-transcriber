"""
Speaker diarization plugin — assigns speaker labels to audio segments.
Uses pyannote/speaker-diarization-3.1 via pyannote.audio 4.x.
Default OFF — enable with --diarize flag.
"""

import numpy as np
import torch
import torch.serialization

# Register safe globals BEFORE loading any pyannote checkpoint.
# pyannote checkpoints contain these custom classes in pickle format.
# PyTorch 2.6+ defaults weights_only=True and rejects them without this.
from pyannote.audio.core.task import Problem, Resolution, Specifications
from omegaconf import DictConfig, ListConfig

torch.serialization.add_safe_globals([
    torch.torch_version.TorchVersion,
    Specifications,
    Problem,
    Resolution,
    DictConfig,
    ListConfig,
])

from pyannote.audio import Pipeline  # noqa: E402

from config import Config


_MAX_HISTORY_SEC = 120  # Keep up to 2 minutes of audio for cross-segment consistency


class Diarizer:
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.sample_rate
        print("[Diarizer] Loading pyannote pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=config.hf_token,
        )
        device = config.diarization_device
        if device == "cuda":
            try:
                self.pipeline.to(torch.device("cuda"))
                print("[Diarizer] Pipeline loaded (GPU)")
            except RuntimeError as e:
                if "no kernel image" in str(e):
                    print("[Diarizer] GPU not supported, falling back to CPU")
                    self.pipeline.to(torch.device("cpu"))
                    print("[Diarizer] Pipeline loaded (CPU)")
                else:
                    raise
        else:
            self.pipeline.to(torch.device("cpu"))
            print("[Diarizer] Pipeline loaded (CPU)")

        # Rolling audio history for cross-segment speaker consistency
        self._history = np.array([], dtype=np.float32)

    def _append_history(self, audio: np.ndarray):
        """Add audio to rolling history buffer, trimming to max length."""
        self._history = np.concatenate([self._history, audio])
        max_samples = int(_MAX_HISTORY_SEC * self.sample_rate)
        if len(self._history) > max_samples:
            self._history = self._history[-max_samples:]

    def diarize(self, audio: np.ndarray) -> list[dict]:
        """
        Run speaker diarization on an audio buffer.

        Returns:
            [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5}, ...]
        """
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        input_data = {"waveform": waveform, "sample_rate": self.sample_rate}

        result = self.pipeline(input_data)
        diarization = getattr(result, "speaker_diarization", result)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
            })
        return segments

    def get_speaker_for_segment(self, segment_audio: np.ndarray) -> str:
        """
        Append segment to history, diarize the full history,
        and return the dominant speaker in the last segment's time range.
        This gives pyannote enough context to distinguish speakers consistently.
        """
        history_start_sec = len(self._history) / self.sample_rate
        self._append_history(segment_audio)
        segment_end_sec = len(self._history) / self.sample_rate

        # Diarize the full history
        segments = self.diarize(self._history)
        if not segments:
            return ""

        # Find speakers active in the time range of the new segment
        durations: dict[str, float] = {}
        for seg in segments:
            # Overlap with the new segment's time range
            overlap_start = max(seg["start"], history_start_sec)
            overlap_end = min(seg["end"], segment_end_sec)
            if overlap_end > overlap_start:
                dur = overlap_end - overlap_start
                durations[seg["speaker"]] = durations.get(seg["speaker"], 0) + dur

        if not durations:
            return ""

        return max(durations, key=durations.get)

    def diarize_and_split(
        self, segment_audio: np.ndarray, word_timestamps: list[dict]
    ) -> list[dict]:
        """
        Append segment to history, diarize, and split words by speaker.

        word_timestamps: [{"word": str, "start": float, "end": float}, ...]
            Timestamps are relative to the segment audio start.

        Returns:
            [{"speaker": "SPEAKER_00", "text": "words..."}, ...]
            If only one speaker, returns a single-element list.
        """
        history_start_sec = len(self._history) / self.sample_rate
        self._append_history(segment_audio)

        # Diarize the full history
        dia_segments = self.diarize(self._history)
        if not dia_segments:
            return [{"speaker": "", "text": " ".join(w["word"] for w in word_timestamps)}]

        # Assign each word to a speaker based on overlap with diarization segments.
        # Word timestamps are relative to segment start; shift to history timeline.
        result_parts: list[dict] = []
        current_speaker = None
        current_words: list[str] = []

        for wt in word_timestamps:
            # Convert word time to history-absolute time
            w_start = history_start_sec + wt["start"]
            w_end = history_start_sec + wt["end"]
            w_mid = (w_start + w_end) / 2

            # Find the speaker at the midpoint of this word
            speaker = self._find_speaker_at(dia_segments, w_mid, w_start, w_end)

            if speaker != current_speaker and current_words:
                result_parts.append({
                    "speaker": current_speaker or "",
                    "text": " ".join(current_words),
                })
                current_words = []
            current_speaker = speaker
            current_words.append(wt["word"])

        if current_words:
            result_parts.append({
                "speaker": current_speaker or "",
                "text": " ".join(current_words),
            })

        return result_parts

    def _find_speaker_at(
        self, dia_segments: list[dict], mid: float, start: float, end: float
    ) -> str:
        """Find the dominant speaker at a given time point."""
        # First try: which diarization segment contains the midpoint?
        for seg in dia_segments:
            if seg["start"] <= mid <= seg["end"]:
                return seg["speaker"]

        # Fallback: find segment with most overlap with the word
        best_speaker = ""
        best_overlap = 0.0
        for seg in dia_segments:
            overlap_start = max(seg["start"], start)
            overlap_end = min(seg["end"], end)
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = seg["speaker"]

        return best_speaker
