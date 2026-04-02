from dataclasses import dataclass


@dataclass
class Config:
    # --- STT Model ---
    model_size: str = "large-v3"           # or "distil-large-v3"
    device: str = "cuda"
    compute_type: str = "float16"          # float16 for RTX 5070 Ti
    language: str = "en"
    beam_size: int = 5

    # --- Audio Capture ---
    sample_rate: int = 16000
    channels: int = 1
    block_duration_ms: int = 100           # 100ms chunks

    # --- VAD ---
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 500

    # --- Buffer ---
    max_buffer_duration_sec: float = 30.0
    context_duration_sec: float = 3.0

    # --- Transcription ---
    transcribe_interval_sec: float = 1.0
    condition_on_previous_text: bool = True

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8765

    # --- Plugins ---
    enable_translation: bool = False
    translation_target_lang: str = "ja"
    enable_summary: bool = False
