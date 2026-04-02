"""
Speaker diarization plugin — assigns speaker labels to audio segments.
Uses pyannote/speaker-diarization-3.1 via pyannote.audio 4.x.
Default OFF — enable with --diarize flag.
"""

import numpy as np
import torch
from pyannote.audio import Pipeline

from config import Config


class Diarizer:
    def __init__(self, config: Config):
        self.config = config
        print("[Diarizer] Loading pyannote pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=config.hf_token,
        )
        self.pipeline.to(torch.device("cuda"))
        print("[Diarizer] Pipeline loaded")

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
        """
        Run speaker diarization on an audio buffer.

        Returns:
            [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5}, ...]
        """
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        input_data = {"waveform": waveform, "sample_rate": sample_rate}

        diarization = self.pipeline(input_data)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
            })
        return segments

    def get_dominant_speaker(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> str:
        """
        Run diarization and return the speaker who talked the most.
        Used for assigning a single speaker label to an endpoint utterance.
        """
        segments = self.diarize(audio, sample_rate)
        if not segments:
            return ""

        # Sum duration per speaker
        durations: dict[str, float] = {}
        for seg in segments:
            dur = seg["end"] - seg["start"]
            durations[seg["speaker"]] = durations.get(seg["speaker"], 0) + dur

        return max(durations, key=durations.get)
