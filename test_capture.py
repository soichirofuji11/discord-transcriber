"""Test script: Capture 5 seconds of loopback audio and save to WAV."""
import time
import wave
import numpy as np
from queue import Queue

from config import Config
from audio_capture import AudioCapture


def main():
    config = Config()
    audio_queue = Queue()
    capture = AudioCapture(config, audio_queue)

    print("Recording 5 seconds of loopback audio...")
    print("Play some audio (e.g. YouTube, Discord) to test!")
    capture.start()
    time.sleep(5)
    capture.stop()

    # Collect all chunks
    chunks = []
    while not audio_queue.empty():
        chunks.append(audio_queue.get())

    if not chunks:
        print("No audio captured!")
        return

    audio = np.concatenate(chunks)
    print(f"Captured {len(audio)} samples ({len(audio)/config.sample_rate:.2f}s)")
    print(f"Max amplitude: {np.max(np.abs(audio)):.4f}")

    # Save to WAV
    filename = "test_capture.wav"
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f"Saved to {filename} - play it to verify!")


if __name__ == "__main__":
    main()
