import argparse
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
        args = parser.parse_args()

        # --fast preset defaults
        if args.fast:
            model = args.model or "distil-large-v3"
            beam_size = args.beam_size if args.beam_size is not None else 1
            interval = args.interval if args.interval is not None else 0.5
        else:
            model = args.model or cls.model_size
            beam_size = args.beam_size if args.beam_size is not None else cls.beam_size
            interval = args.interval if args.interval is not None else cls.transcribe_interval_sec

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
        )
