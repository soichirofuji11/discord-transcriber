import threading
import time

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
        self._streams: list = []

        source = config.audio_source  # "loopback", "mic", or "both"

        if source in ("loopback", "both"):
            dev = self._find_loopback_device()
            self._loopback = self._make_stream_info(dev, "loopback")
        else:
            self._loopback = None

        if source in ("mic", "both"):
            dev = self._find_mic_device()
            self._mic = self._make_stream_info(dev, "mic")
        else:
            self._mic = None

        # Synchronous mixing for "both" mode
        self._mix_mode = (source == "both")
        if self._mix_mode:
            self._lb_buf = np.array([], dtype=np.float32)
            self._mic_buf = np.array([], dtype=np.float32)
            self._buf_lock = threading.Lock()
            self._mixer_thread = None
            self._mixer_running = False

    # ------------------------------------------------------------------ #
    # Device discovery
    # ------------------------------------------------------------------ #

    def _get_wasapi_host(self) -> dict:
        for i in range(self.pa.get_host_api_count()):
            info = self.pa.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                return info
        raise RuntimeError("WASAPI host API not found")

    def _find_loopback_device(self) -> dict:
        """Find the WASAPI loopback device for the default output."""
        wasapi_info = self._get_wasapi_host()
        default_output = self.pa.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )
        default_name = default_output["name"]

        for i in range(self.pa.get_device_count()):
            dev = self.pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_info["index"]
                and "loopback" in dev["name"].lower()
                and default_name in dev["name"]
            ):
                print(f"[AudioCapture] Loopback: {dev['name']} "
                      f"(rate={int(dev['defaultSampleRate'])}, ch={dev['maxInputChannels']})")
                return dev

        raise RuntimeError(
            f"Loopback device for '{default_name}' not found. "
            "Check Windows sound settings."
        )

    def _find_mic_device(self) -> dict:
        """Find the default WASAPI input (microphone) device."""
        wasapi_info = self._get_wasapi_host()
        default_input_idx = wasapi_info["defaultInputDevice"]
        if default_input_idx < 0:
            raise RuntimeError(
                "No default input device found. "
                "Check Windows sound settings and ensure a microphone is connected."
            )
        dev = self.pa.get_device_info_by_index(default_input_idx)
        print(f"[AudioCapture] Mic: {dev['name']} "
              f"(rate={int(dev['defaultSampleRate'])}, ch={dev['maxInputChannels']})")
        return dev

    # ------------------------------------------------------------------ #
    # Stream setup
    # ------------------------------------------------------------------ #

    def _make_stream_info(self, device: dict, label: str) -> dict:
        device_rate = int(device["defaultSampleRate"])
        device_channels = device["maxInputChannels"]
        need_resample = device_rate != self.config.sample_rate

        info = {
            "device": device,
            "label": label,
            "rate": device_rate,
            "channels": device_channels,
            "need_resample": need_resample,
        }

        if need_resample:
            g = gcd(self.config.sample_rate, device_rate)
            info["up"] = self.config.sample_rate // g
            info["down"] = device_rate // g

        return info

    def _make_callback(self, stream_info: dict, buf_attr: str = None):
        """
        Create an audio callback.
        buf_attr=None  -> put directly in audio_queue (single source mode)
        buf_attr set   -> append to internal buffer (mixing mode)
        """
        channels = stream_info["channels"]
        need_resample = stream_info["need_resample"]
        up = stream_info.get("up")
        down = stream_info.get("down")
        label = stream_info["label"]
        last_log = [0.0]
        max_buf = self.config.sample_rate  # cap internal buffer at 1 sec

        def callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.float32)
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            if need_resample:
                audio = resample_poly(audio, up, down).astype(np.float32)

            # Log volume every second
            now = time.time()
            if now - last_log[0] >= 1.0:
                rms = np.sqrt(np.mean(audio ** 2))
                db = 20 * np.log10(rms + 1e-10)
                peak = np.max(np.abs(audio))
                print(f"[Audio:{label}] RMS={db:.1f}dB  peak={peak:.4f}")
                last_log[0] = now

            if buf_attr is not None:
                with self._buf_lock:
                    buf = np.concatenate([getattr(self, buf_attr), audio])
                    if len(buf) > max_buf:
                        buf = buf[-max_buf:]
                    setattr(self, buf_attr, buf)
            else:
                self.audio_queue.put(audio)

            return (None, pyaudio.paContinue)

        return callback

    # ------------------------------------------------------------------ #
    # Mixer (for "both" mode)
    # ------------------------------------------------------------------ #

    def _mixer_loop(self):
        """Read from both internal buffers, add-mix, and put in queue."""
        chunk_size = int(self.config.sample_rate * self.config.block_duration_ms / 1000)
        interval = self.config.block_duration_ms / 1000

        while self._mixer_running:
            time.sleep(interval)
            with self._buf_lock:
                lb = self._lb_buf[:chunk_size]
                mic = self._mic_buf[:chunk_size]
                self._lb_buf = self._lb_buf[len(lb):]
                self._mic_buf = self._mic_buf[len(mic):]

            n = max(len(lb), len(mic))
            if n == 0:
                continue

            # Pad the shorter one with zeros and add
            if len(lb) < n:
                lb = np.pad(lb, (0, n - len(lb)))
            if len(mic) < n:
                mic = np.pad(mic, (0, n - len(mic)))

            self.audio_queue.put((lb + mic).astype(np.float32))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def _open_stream(self, si: dict, buf_attr: str = None):
        block_samples = int(si["rate"] * self.config.block_duration_ms / 1000)
        stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=si["channels"],
            rate=si["rate"],
            input=True,
            input_device_index=si["device"]["index"],
            frames_per_buffer=block_samples,
            stream_callback=self._make_callback(si, buf_attr=buf_attr),
        )
        stream.start_stream()
        self._streams.append(stream)

    def start(self):
        if self._mix_mode:
            # Both sources write to internal buffers; mixer thread combines them
            if self._loopback:
                self._open_stream(self._loopback, buf_attr="_lb_buf")
            if self._mic:
                self._open_stream(self._mic, buf_attr="_mic_buf")
            self._mixer_running = True
            self._mixer_thread = threading.Thread(target=self._mixer_loop, daemon=True)
            self._mixer_thread.start()
        else:
            # Single source: callback puts directly in queue
            for si in (self._loopback, self._mic):
                if si is not None:
                    self._open_stream(si)

        sources = [si["label"] for si in (self._loopback, self._mic) if si]
        mode = " (mixed)" if self._mix_mode else ""
        print(f"[AudioCapture] Started ({' + '.join(sources)}{mode})")

    def stop(self):
        self._mixer_running = False
        if hasattr(self, "_mixer_thread") and self._mixer_thread:
            self._mixer_thread.join(timeout=2)
        for stream in self._streams:
            stream.stop_stream()
            stream.close()
        self._streams.clear()
        self.pa.terminate()
        print("[AudioCapture] Stopped")
