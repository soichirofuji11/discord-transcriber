"""Discord VC real-time transcription tool — entry point."""
import threading
from queue import Queue

import uvicorn

from config import Config
from audio_capture import AudioCapture
from vad import VADProcessor
from transcriber import Transcriber
from server import app, enqueue_message


def main():
    config = Config()
    audio_queue: Queue = Queue()

    def on_text(result):
        """Called from the processing thread."""
        tag = "FINAL" if result["type"] == "final" else "partial"
        print(f"[{tag}] {result['text']}")
        enqueue_message(result)

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
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")

    # Cleanup
    stop_event.set()
    capture.stop()
    print("Done.")


if __name__ == "__main__":
    main()
