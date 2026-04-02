"""
Speaker diarization plugin — assigns speaker labels to transcript segments.
Uses pyannote/speaker-diarization-3.1 (requires HF token + model access).
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
            use_auth_token=config.hf_token,
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

    def assign_speakers_to_transcript(
        self,
        transcript_segments: list[dict],
        diarization_segments: list[dict],
    ) -> list[dict]:
        """
        Assign speaker labels to Whisper transcript segments.
        Uses midpoint matching.
        """
        result = []
        for t_seg in transcript_segments:
            mid_time = (t_seg["start"] + t_seg["end"]) / 2
            speaker = "Unknown"
            for d_seg in diarization_segments:
                if d_seg["start"] <= mid_time <= d_seg["end"]:
                    speaker = d_seg["speaker"]
                    break
            result.append({**t_seg, "speaker": speaker})
        return result
