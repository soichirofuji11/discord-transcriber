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


class Diarizer:
    def __init__(self, config: Config):
        self.config = config
        print("[Diarizer] Loading pyannote pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=config.hf_token,
        )
        # Try GPU first, fall back to CPU if CUDA kernel is not available
        # (e.g. RTX 5070 Ti sm_120 may not be supported by all sub-models)
        try:
            self.pipeline.to(torch.device("cuda"))
            print("[Diarizer] Pipeline loaded (GPU)")
        except RuntimeError as e:
            if "no kernel image" in str(e):
                print(f"[Diarizer] GPU not supported for this model, using CPU")
                self.pipeline.to(torch.device("cpu"))
                print("[Diarizer] Pipeline loaded (CPU)")
            else:
                raise

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
        """
        Run speaker diarization on an audio buffer.

        Returns:
            [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5}, ...]
        """
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        input_data = {"waveform": waveform, "sample_rate": sample_rate}

        result = self.pipeline(input_data)
        # pyannote 4.x returns DiarizeOutput with .speaker_diarization attribute
        diarization = getattr(result, "speaker_diarization", result)

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

        durations: dict[str, float] = {}
        for seg in segments:
            dur = seg["end"] - seg["start"]
            durations[seg["speaker"]] = durations.get(seg["speaker"], 0) + dur

        return max(durations, key=durations.get)
