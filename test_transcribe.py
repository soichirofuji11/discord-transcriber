"""Test script: Transcribe a WAV file using faster-whisper."""
import wave
import numpy as np
from config import Config
from transcriber import Transcriber


def main():
    config = Config()

    results = []

    def on_text(result):
        results.append(result)
        print(f"[{result['type']}] {result['text']}")
        print(f"  language={result['language']} "
              f"(prob={result['language_probability']:.2f})")

    transcriber = Transcriber(config, on_text)

    # Load WAV file
    filename = "test_capture.wav"
    with wave.open(filename, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"Loaded {filename}: {len(audio)} samples, {len(audio)/rate:.2f}s")

    # Feed audio and transcribe
    transcriber.add_audio(audio)
    transcriber.transcribe(is_endpoint=True)

    if not results:
        print("No transcription result (audio may be silent)")
    else:
        print(f"\nTotal results: {len(results)}")


if __name__ == "__main__":
    main()
