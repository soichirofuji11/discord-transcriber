"""Discord VC real-time transcription tool — entry point."""
import asyncio
import threading
import warnings
from queue import Queue

# Suppress noisy warnings that don't affect functionality
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*triton.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import uvicorn

from config import Config
from audio_capture import AudioCapture
from vad import VADProcessor
from transcriber import Transcriber
from server import app, enqueue_message, set_session, set_summarizer, register_startup_task
from plugins.session_store import Session


def main():
    config = Config.from_args()
    audio_queue: Queue = Queue()

    print(f"[Config] model={config.model_size}, beam={config.beam_size}, "
          f"interval={config.transcribe_interval_sec}s, "
          f"context={config.condition_on_previous_text}")

    # Session store
    session = Session.new()
    set_session(session)
    print(f"[Session] {session.session_id}")

    # Optional: translator
    translator = None
    if config.enable_translation:
        from plugins.translator import Translator
        translator = Translator(config, session)
        print(f"[Translator] Enabled ({config.translation_model})")

    # Optional: summarizer
    if config.enable_summary:
        from plugins.summarizer import Summarizer
        summarizer = Summarizer(config)
        set_summarizer(summarizer)
        print(f"[Summarizer] Enabled ({config.summary_model})")

    # Optional: diarizer (model loaded only when enabled)
    diarizer = None
    if config.enable_diarization:
        try:
            from plugins.diarizer import Diarizer
            diarizer = Diarizer(config)
        except Exception as e:
            print(f"[Diarizer] Failed to load, continuing without diarization: {e}")

    # --- Diarization: async post-hoc speaker labeling ---
    # Queue of (msg_id, audio_copy) to diarize in background
    diarize_queue: Queue = Queue()

    async def diarization_loop():
        """Process diarization requests without blocking transcription."""
        if not diarizer:
            return
        while True:
            if not diarize_queue.empty():
                try:
                    msg_id, audio = diarize_queue.get_nowait()
                    # Run diarization in a thread to avoid blocking the event loop
                    speaker = await asyncio.to_thread(
                        diarizer.get_dominant_speaker, audio, config.sample_rate
                    )
                    if speaker:
                        print(f"[Diarizer] msg_id={msg_id} -> {speaker}")
                        enqueue_message({
                            "type": "speaker_update",
                            "msg_id": msg_id,
                            "speaker": speaker,
                        })
                        # Update session entry
                        session.update_speaker(msg_id, speaker)
                except Exception as e:
                    print(f"[Diarizer] Error: {e}")
            await asyncio.sleep(0.1)

    if diarizer:
        register_startup_task(diarization_loop)

    # --- Translation flush loop ---
    async def translation_loop():
        """Periodically flush the translator batch buffer."""
        if not translator:
            return
        while True:
            if translator.should_flush():
                results = await translator.flush_batch()
                for item in results:
                    session.update_translation(item["original"], item["translated"])
                    enqueue_message({
                        "type": "translation",
                        "text": item["translated"],
                        "original": item["original"],
                        "msg_id": item["msg_id"],
                        "lang": "ja",
                    })
            await asyncio.sleep(0.3)

    if translator:
        register_startup_task(translation_loop)

    _msg_id_counter = [0]

    def on_text(result):
        """Called from the processing thread."""
        tag = "FINAL" if result["type"] == "final" else "partial"
        latency = result.get("latency_ms", "?")
        audio_sec = result.get("audio_sec", "?")
        print(f"[{tag}] ({latency}ms, {audio_sec}s buf) {result['text']}")

        # Assign a unique ID to final messages
        if result["type"] == "final":
            _msg_id_counter[0] += 1
            result["msg_id"] = _msg_id_counter[0]

        enqueue_message(result)

        # Store final text in session and trigger plugins
        if result["type"] == "final":
            session.add_entry(result["text"], speaker="", msg_id=result["msg_id"])
            if translator:
                translator.add_to_batch(result["text"], msg_id=result["msg_id"])
            # Queue diarization (non-blocking)
            if diarizer and transcriber.last_endpoint_audio is not None:
                diarize_queue.put((result["msg_id"], transcriber.last_endpoint_audio))
                transcriber.last_endpoint_audio = None

    # Initialize modules
    capture = AudioCapture(config, audio_queue)
    vad = VADProcessor(config)
    transcriber = Transcriber(config, on_text)

    # Audio processing loop (runs in a separate thread)
    stop_event = threading.Event()

    def audio_processing_loop():
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.5)
            except Exception:
                continue

            vad_result = vad.process(chunk)

            if vad_result["has_speech"]:
                transcriber.add_audio(chunk)

            transcriber.transcribe(is_endpoint=vad_result["is_endpoint"])

    processing_thread = threading.Thread(
        target=audio_processing_loop, daemon=True
    )
    processing_thread.start()

    capture.start()
    print(f"Listening on loopback audio...")
    print(f"Open http://{config.host}:{config.port} in your browser")

    # Run FastAPI server (blocks until shutdown)
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")

    # Cleanup
    stop_event.set()
    capture.stop()
    path = session.save(config.sessions_dir)
    print(f"[Session] Saved to {path}")
    print("Done.")


if __name__ == "__main__":
    main()
