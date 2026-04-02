"""Discord VC real-time transcription tool — entry point."""
import asyncio
import threading
from queue import Queue

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

    # Translation flush loop (runs in background on the event loop)
    translation_queue: Queue = Queue()

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

        # Assign a unique ID to final messages for translation pairing
        if result["type"] == "final":
            _msg_id_counter[0] += 1
            result["msg_id"] = _msg_id_counter[0]

        enqueue_message(result)

        # Store final text in session
        if result["type"] == "final":
            session.add_entry(result["text"], speaker=result.get("speaker", ""))
            if translator:
                translator.add_to_batch(result["text"], msg_id=result["msg_id"])

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
    # Auto-save session on exit
    path = session.save(config.sessions_dir)
    print(f"[Session] Saved to {path}")
    print("Done.")


if __name__ == "__main__":
    main()
