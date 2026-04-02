"""Test script: Run VAD on the captured WAV file."""
import wave
import numpy as np
from config import Config
from vad import VADProcessor


def main():
    config = Config()
    vad = VADProcessor(config)

    # Load WAV
    filename = "test_capture.wav"
    with wave.open(filename, "rb") as wf:
        assert wf.getnchannels() == 1
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"Loaded {filename}: {len(audio)} samples, {len(audio)/rate:.2f}s")

    # Process in chunks (same size as real-time would use)
    chunk_size = int(config.sample_rate * config.block_duration_ms / 1000)
    # Silero VAD requires specific chunk sizes: 256, 512, or 768 for 16kHz
    # Use 512 (32ms) as it's the closest standard size
    chunk_size = 512

    speech_chunks = 0
    silence_chunks = 0
    endpoints = 0
    total_chunks = 0

    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i : i + chunk_size]
        result = vad.process(chunk)
        total_chunks += 1

        if result["has_speech"]:
            speech_chunks += 1
        else:
            silence_chunks += 1

        if result["is_endpoint"]:
            endpoints += 1
            t = i / config.sample_rate
            print(f"  Endpoint detected at {t:.2f}s")

        # Print first few chunks for debugging
        if total_chunks <= 5 or result["has_speech"]:
            t = i / config.sample_rate
            print(
                f"  t={t:.2f}s  speech_prob={result['speech_prob']:.3f}  "
                f"has_speech={result['has_speech']}  "
                f"is_endpoint={result['is_endpoint']}"
            )

    print(f"\nSummary:")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Speech chunks: {speech_chunks}")
    print(f"  Silence chunks: {silence_chunks}")
    print(f"  Endpoints: {endpoints}")
    print("VAD test complete!")


if __name__ == "__main__":
    main()
