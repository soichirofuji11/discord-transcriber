import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly
from math import gcd
from queue import Queue

from config import Config


class AudioCapture:
    def __init__(self, config: Config, audio_queue: Queue):
        self.config = config
        self.audio_queue = audio_queue
        self.pa = pyaudio.PyAudio()
        self.stream = None

        self._device_info = self._find_loopback_device()
        self._device_rate = int(self._device_info["defaultSampleRate"])
        self._device_channels = self._device_info["maxInputChannels"]

        # Resampling ratio (device rate -> 16kHz)
        self._need_resample = self._device_rate != config.sample_rate
        if self._need_resample:
            g = gcd(config.sample_rate, self._device_rate)
            self._up = config.sample_rate // g
            self._down = self._device_rate // g

    def _find_loopback_device(self) -> dict:
        """Find the WASAPI loopback device for the default output."""
        wasapi_info = None
        for i in range(self.pa.get_host_api_count()):
            info = self.pa.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                wasapi_info = info
                break

        if wasapi_info is None:
            raise RuntimeError("WASAPI host API not found")

        default_output = self.pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )
        default_name = default_output["name"]

        # Find corresponding loopback device
        for i in range(self.pa.get_device_count()):
            dev = self.pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_info["index"]
                and "loopback" in dev["name"].lower()
                and default_name in dev["name"]
            ):
                print(f"[AudioCapture] Using loopback: {dev['name']} "
                      f"(rate={int(dev['defaultSampleRate'])}, ch={dev['maxInputChannels']})")
                return dev

        raise RuntimeError(
            f"Loopback device for '{default_name}' not found. "
            "Check Windows sound settings."
        )

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Convert raw bytes to float32 mono 16kHz and enqueue."""
        audio = np.frombuffer(in_data, dtype=np.float32)
        # Convert to mono by averaging channels
        if self._device_channels > 1:
            audio = audio.reshape(-1, self._device_channels).mean(axis=1)
        # Resample to target sample rate
        if self._need_resample:
            audio = resample_poly(audio, self._up, self._down).astype(np.float32)
        self.audio_queue.put(audio)
        return (None, pyaudio.paContinue)

    def start(self):
        """Start audio capture."""
        block_samples = int(
            self._device_rate * self.config.block_duration_ms / 1000
        )
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=self._device_channels,
            rate=self._device_rate,
            input=True,
            input_device_index=self._device_info["index"],
            frames_per_buffer=block_samples,
            stream_callback=self._audio_callback,
        )
        self.stream.start_stream()
        print("[AudioCapture] Started")

    def stop(self):
        """Stop audio capture."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        print("[AudioCapture] Stopped")
