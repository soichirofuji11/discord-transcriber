import argparse
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- STT Model ---
    model_size: str = "distil-large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    beam_size: int = 1

    # --- Audio Capture ---
    sample_rate: int = 16000
    channels: int = 1
    block_duration_ms: int = 100

    # --- VAD ---
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 1000

    # --- Buffer ---
    max_buffer_duration_sec: float = 30.0
    context_duration_sec: float = 3.0

    # --- Transcription ---
    transcribe_interval_sec: float = 1.0
    condition_on_previous_text: bool = True

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8765

    # --- Translation Plugin ---
    enable_translation: bool = False
    gemini_api_key: str = ""
    translation_model: str = "gemini-3.1-flash-lite-preview"
    translation_target_lang: str = "ja"
    translation_batch_interval_sec: float = 1.5
    translation_context_lines: int = 5

    # --- Summary Plugin ---
    enable_summary: bool = False
    summary_model: str = "gemini-3-flash-preview"
    sessions_dir: str = "sessions"

    # --- Diarization Plugin ---
    enable_diarization: bool = False
    hf_token: str = ""
    max_speakers: int = 5

    @classmethod
    def from_args(cls) -> "Config":
        parser = argparse.ArgumentParser(description="Discord VC Transcriber")
        parser.add_argument("--fast", action="store_true",
                            help="Low-latency preset: beam=1, interval=0.5s, distil-large-v3")
        parser.add_argument("--model", default=None,
                            choices=["large-v3", "distil-large-v3", "medium", "small", "base", "tiny"],
                            help="Whisper model size")
        parser.add_argument("--beam-size", type=int, default=None,
                            help="Beam size (1=greedy/fast, 5=accurate)")
        parser.add_argument("--language", default=cls.language,
                            help="Language code (e.g. en, ja)")
        parser.add_argument("--interval", type=float, default=None,
                            help="Transcription interval in seconds")
        parser.add_argument("--no-context", action="store_true",
                            help="Disable condition_on_previous_text")
        parser.add_argument("--max-buffer", type=float, default=cls.max_buffer_duration_sec,
                            help="Max audio buffer duration in seconds")
        parser.add_argument("--context-duration", type=float, default=cls.context_duration_sec,
                            help="Context audio kept after endpoint (seconds)")
        parser.add_argument("--min-silence", type=int, default=cls.min_silence_duration_ms,
                            help="Min silence to trigger endpoint (ms)")
        parser.add_argument("--port", type=int, default=cls.port,
                            help="WebSocket server port")
        parser.add_argument("--translate", action="store_true",
                            help="Enable English→Japanese translation")
        parser.add_argument("--summarize", action="store_true",
                            help="Enable summary feature")
        parser.add_argument("--diarize", action="store_true",
                            help="Enable speaker diarization")
        args = parser.parse_args()

        # Environment variables
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        hf_token = os.environ.get("HF_TOKEN", "")

        # --fast preset defaults
        if args.fast:
            model = args.model or "distil-large-v3"
            beam_size = args.beam_size if args.beam_size is not None else 1
            interval = args.interval if args.interval is not None else 0.5
        else:
            model = args.model or cls.model_size
            beam_size = args.beam_size if args.beam_size is not None else cls.beam_size
            interval = args.interval if args.interval is not None else cls.transcribe_interval_sec

        # Auto-disable plugins if API keys missing
        enable_translation = args.translate
        enable_summary = args.summarize
        enable_diarization = args.diarize

        if enable_translation and not gemini_key:
            print("[Config] GEMINI_API_KEY not set -translation disabled")
            enable_translation = False
        if enable_summary and not gemini_key:
            print("[Config] GEMINI_API_KEY not set -summary disabled")
            enable_summary = False
        if enable_diarization and not hf_token:
            print("[Config] HF_TOKEN not set -diarization disabled")
            enable_diarization = False

        return cls(
            model_size=model,
            beam_size=beam_size,
            language=args.language,
            transcribe_interval_sec=interval,
            condition_on_previous_text=not args.no_context,
            max_buffer_duration_sec=args.max_buffer,
            context_duration_sec=args.context_duration,
            min_silence_duration_ms=args.min_silence,
            port=args.port,
            enable_translation=enable_translation,
            gemini_api_key=gemini_key,
            enable_summary=enable_summary,
            enable_diarization=enable_diarization,
            hf_token=hf_token,
        )
