# Discord VC リアルタイム文字起こしツール — プロジェクト仕様書

> **このドキュメントをClaude Codeに渡して、ステップバイステップで実装してもらってください。**

---

## 概要

Discord のボイスチャット音声をリアルタイムで文字起こしし、ブラウザ上にオーバーレイ表示するローカルアプリケーション。

### ゴール
- **最高精度**: Whisper Large V3 / Distil-Whisper Large V3 を使用
- **最低遅延**: VAD + ローリングバッファで 1〜2 秒の遅延を目指す
- **拡張可能**: 後から翻訳（英→日）、要約機能を追加できるアーキテクチャ

### ターゲット環境
- **OS**: Windows 11
- **GPU**: NVIDIA RTX 5070 Ti (16GB VRAM)
- **Python**: 3.11 or 3.12
- **CUDA**: 12.x

---

## アーキテクチャ

```
┌─────────────────┐
│   Discord VC     │
│  (スピーカー出力) │
└────────┬────────┘
         │ WASAPI Loopback
         ▼
┌─────────────────┐
│  Audio Capture   │  ← sounddevice (WASAPI loopback)
│  16kHz mono PCM  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Silero VAD     │  ← 音声区間検出（無音スキップ）
│                  │
└────────┬────────┘
         │ 音声チャンク
         ▼
┌─────────────────┐
│  Audio Buffer    │  ← ローリングバッファ（文脈保持）
│  (max 30sec)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  faster-whisper  │  ← GPU推論 (CUDA / float16)
│  large-v3 or     │
│  distil-large-v3 │
└────────┬────────┘
         │ テキスト
         ▼
┌─────────────────┐
│  WebSocket Server│  ← FastAPI + WebSocket
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Browser UI      │  ← オーバーレイ表示
│  (HTML/JS)       │
└─────────────────┘
         │
    ┌────┴────┐  (将来の拡張)
    ▼         ▼
┌────────┐ ┌────────┐
│ 翻訳   │ │ 要約   │
│ API    │ │ LLM    │
└────────┘ └────────┘
```

---

## ディレクトリ構成

```
discord-transcriber/
├── README.md
├── requirements.txt
├── config.py              # 設定（モデル、デバイス、バッファサイズ等）
├── main.py                # エントリーポイント
├── audio_capture.py       # WASAPI loopback 音声キャプチャ
├── vad.py                 # Silero VAD ラッパー
├── transcriber.py         # faster-whisper 推論エンジン
├── server.py              # FastAPI WebSocket サーバー
├── static/
│   └── index.html         # ブラウザUI（オーバーレイ）
└── plugins/               # 将来の拡張用
    ├── __init__.py
    ├── translator.py      # 翻訳プラグイン（後で実装）
    └── summarizer.py      # 要約プラグイン（後で実装）
```

---

## 依存パッケージ (requirements.txt)

```
faster-whisper>=1.1.0
sounddevice>=0.5.0
numpy>=1.26.0
fastapi>=0.115.0
uvicorn>=0.32.0
websockets>=13.0
silero-vad>=5.1
torch>=2.4.0
```

### 補足: インストール手順

```bash
# 1. Python仮想環境を作成
python -m venv venv
venv\Scripts\activate

# 2. PyTorch (CUDA 12.x) をインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 残りの依存パッケージ
pip install -r requirements.txt
```

> **注意**: faster-whisper は cuDNN と cuBLAS が必要。
> NVIDIA CUDA Toolkit 12.x がインストール済みであること。
> `nvidia-smi` で GPU が認識されていることを確認。

---

## 各モジュールの詳細仕様

### 1. config.py — 設定

```python
from dataclasses import dataclass, field
from typing import Optional

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
    block_duration_ms: int = 100           # 100ms チャンク

    # --- VAD ---
    vad_threshold: float = 0.5             # 音声判定の閾値
    min_speech_duration_ms: int = 250      # 最低音声長
    min_silence_duration_ms: int = 500     # 無音がこれ以上続いたら区切る

    # --- Buffer ---
    max_buffer_duration_sec: float = 30.0  # 最大バッファ長
    context_duration_sec: float = 3.0      # 文脈として保持する前回の音声

    # --- Transcription ---
    transcribe_interval_sec: float = 1.0   # 文字起こし実行間隔
    condition_on_previous_text: bool = True # 前回のテキストを文脈として使う

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8765

    # --- Plugins (将来の拡張) ---
    enable_translation: bool = False
    translation_target_lang: str = "ja"
    enable_summary: bool = False
```

### 2. audio_capture.py — WASAPI Loopback 音声キャプチャ

**役割**: Discord がスピーカーに出力する音声を WASAPI loopback でキャプチャし、
16kHz mono float32 の numpy array としてキューに流す。

**実装のポイント**:
- `sounddevice` ライブラリを使用
- Windows WASAPI loopback を有効にするには、`sounddevice.query_devices()` で
  loopback 対応デバイスを探す
- **重要**: sounddevice の WASAPI loopback は Windows でのみ動作
- loopback デバイスのサンプルレートがモデルの期待値 (16kHz) と異なる場合、
  リサンプリングが必要（scipy.signal.resample や resampy を使用）
- コールバック方式でブロッキングしないようにする

```python
# 擬似コード
import sounddevice as sd
import numpy as np
from queue import Queue

class AudioCapture:
    def __init__(self, config, audio_queue: Queue):
        self.config = config
        self.audio_queue = audio_queue
        self.device_id = self._find_loopback_device()

    def _find_loopback_device(self) -> int:
        """WASAPI loopback デバイスを探す"""
        devices = sd.query_devices()
        # hostapi が WASAPI で、loopback 可能なデバイスを選択
        # sd.query_hostapis() で WASAPI の index を取得
        # デバイス名に "Loopback" が含まれるか、
        # または default output device の loopback を使う
        ...

    def _audio_callback(self, indata, frames, time, status):
        """音声データをキューに送信"""
        if status:
            print(f"Audio status: {status}")
        # float32 mono に変換して queue に put
        audio_chunk = indata[:, 0].copy()  # mono
        self.audio_queue.put(audio_chunk)

    def start(self):
        """音声キャプチャを開始"""
        self.stream = sd.InputStream(
            device=self.device_id,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype='float32',
            blocksize=int(self.config.sample_rate * self.config.block_duration_ms / 1000),
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()
```

> **注意**: sounddevice は WASAPI loopback をネイティブサポートしていない場合がある。
> その場合は `pyaudiowpatch` を代替として使用する。
> `pip install PyAudioWPATCH` でインストールし、WASAPI loopback ストリームを開く。
> pyaudiowpatch は PyAudio のフォークで、WASAPI loopback を直接サポートしている。

### 3. vad.py — Silero VAD ラッパー

**役割**: 音声チャンクが「話し声を含むか」を判定し、無音部分をスキップする。

**実装のポイント**:
- Silero VAD モデルをロード（torch.hub から取得、~2MB）
- 音声チャンクごとに speech probability を計算
- 閾値を超えた区間のみを transcriber に渡す
- 一定時間の無音が続いたら「発話の区切り」として通知

```python
# 擬似コード
import torch

class VADProcessor:
    def __init__(self, config):
        self.config = config
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True  # ONNX版のほうが軽い
        )
        self.is_speaking = False
        self.silence_samples = 0

    def process(self, audio_chunk: np.ndarray) -> dict:
        """
        Returns:
            {
                "has_speech": bool,
                "speech_prob": float,
                "is_endpoint": bool  # 発話の区切り
            }
        """
        tensor = torch.from_numpy(audio_chunk).float()
        speech_prob = self.model(tensor, self.config.sample_rate).item()

        has_speech = speech_prob >= self.config.vad_threshold

        if has_speech:
            self.is_speaking = True
            self.silence_samples = 0
        else:
            self.silence_samples += len(audio_chunk)

        # 無音が一定以上続いたら endpoint
        silence_ms = (self.silence_samples / self.config.sample_rate) * 1000
        is_endpoint = (
            self.is_speaking
            and silence_ms >= self.config.min_silence_duration_ms
        )

        if is_endpoint:
            self.is_speaking = False
            self.silence_samples = 0

        return {
            "has_speech": has_speech,
            "speech_prob": speech_prob,
            "is_endpoint": is_endpoint,
        }
```

### 4. transcriber.py — faster-whisper 推論エンジン

**役割**: バッファに溜まった音声を faster-whisper で文字起こしする。

**実装のポイント**:
- `WhisperModel` を初期化時にロード（初回のみ時間がかかる）
- ローリングバッファで最大30秒の音声を保持
- 前回の文字起こし結果を `initial_prompt` として渡すことで文脈を保持
- 確定した文と、暫定の文（partial）を区別して返す
- VAD の endpoint を受け取ったら即座に文字起こしを実行

```python
# 擬似コード
from faster_whisper import WhisperModel
import numpy as np
import threading
import time

class Transcriber:
    def __init__(self, config, on_text_callback):
        self.config = config
        self.callback = on_text_callback
        self.model = WhisperModel(
            config.model_size,
            device=config.device,
            compute_type=config.compute_type,
        )
        self.audio_buffer = np.array([], dtype=np.float32)
        self.previous_text = ""  # 文脈保持用
        self.lock = threading.Lock()
        self.last_transcribe_time = 0

    def add_audio(self, audio_chunk: np.ndarray):
        """音声チャンクをバッファに追加"""
        with self.lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            # 最大バッファ長を超えたら古い部分を削除
            max_samples = int(self.config.max_buffer_duration_sec * self.config.sample_rate)
            if len(self.audio_buffer) > max_samples:
                self.audio_buffer = self.audio_buffer[-max_samples:]

    def transcribe(self, is_endpoint: bool = False):
        """文字起こしを実行"""
        now = time.time()
        # endpoint でないなら interval を守る
        if not is_endpoint:
            if now - self.last_transcribe_time < self.config.transcribe_interval_sec:
                return
        self.last_transcribe_time = now

        with self.lock:
            if len(self.audio_buffer) < self.config.sample_rate * 0.5:
                return  # 0.5秒未満なら skip
            audio = self.audio_buffer.copy()

        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False,  # 自前VADを使うのでオフ
            initial_prompt=self.previous_text if self.config.condition_on_previous_text else None,
            condition_on_previous_text=self.config.condition_on_previous_text,
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts)

        if full_text:
            self.callback({
                "type": "final" if is_endpoint else "partial",
                "text": full_text,
                "language": info.language,
                "language_probability": info.language_probability,
            })

            if is_endpoint:
                # 文脈として最後の文を保持
                self.previous_text = full_text[-500:]  # 最後500文字
                # バッファをクリア（context分だけ残す）
                context_samples = int(self.config.context_duration_sec * self.config.sample_rate)
                with self.lock:
                    self.audio_buffer = self.audio_buffer[-context_samples:]
```

### 5. server.py — FastAPI WebSocket サーバー

**役割**: 文字起こし結果をブラウザにリアルタイム配信する。

```python
# 擬似コード
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 接続中の WebSocket クライアントを管理
connected_clients: set[WebSocket] = set()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except:
        connected_clients.discard(websocket)

async def broadcast(message: dict):
    """全クライアントにメッセージを送信"""
    dead = set()
    for client in connected_clients:
        try:
            await client.send_text(json.dumps(message))
        except:
            dead.add(client)
    connected_clients -= dead

@app.get("/")
async def root():
    return HTMLResponse(open("static/index.html").read())
```

### 6. static/index.html — ブラウザUI

**役割**: オーバーレイとして文字起こし結果を表示する。

**デザイン要件**:
- 半透明の黒背景、白テキスト
- 最新の文字起こし結果が下に表示（チャット風）
- partial（暫定）はグレー、final（確定）は白で表示
- 古いテキストは自動でフェードアウト
- フォントは等幅 or 読みやすいサンセリフ
- ウィンドウサイズ可変（OBS等に取り込みやすく）

```html
<!-- 基本構成 -->
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            background: rgba(0, 0, 0, 0.7);
            color: white;
            font-family: 'Segoe UI', sans-serif;
            font-size: 18px;
            padding: 16px;
            overflow: hidden;
        }
        .line { margin: 4px 0; transition: opacity 0.5s; }
        .partial { color: #aaa; font-style: italic; }
        .final { color: #fff; }
        .fading { opacity: 0; }
    </style>
</head>
<body>
    <div id="transcript"></div>
    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        const container = document.getElementById('transcript');
        const MAX_LINES = 10;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            // partial → 既存の partial 行を更新
            // final → 確定行として追加、partial をクリア
            // 古い行をフェードアウト
            // ...実装
        };
    </script>
</body>
</html>
```

### 7. main.py — エントリーポイント

**役割**: 全モジュールを組み合わせてパイプラインを起動する。

```python
# 擬似コード
import asyncio
import threading
from queue import Queue

from config import Config
from audio_capture import AudioCapture
from vad import VADProcessor
from transcriber import Transcriber
from server import app, broadcast

import uvicorn

def main():
    config = Config()
    audio_queue = Queue()

    # コールバック: 文字起こし結果をWebSocketで配信
    loop = asyncio.new_event_loop()

    def on_text(result):
        asyncio.run_coroutine_threadsafe(broadcast(result), loop)

    # 各モジュール初期化
    capture = AudioCapture(config, audio_queue)
    vad = VADProcessor(config)
    transcriber = Transcriber(config, on_text)

    # 音声処理ループ（別スレッド）
    def audio_processing_loop():
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break

            vad_result = vad.process(chunk)

            if vad_result["has_speech"]:
                transcriber.add_audio(chunk)

            # endpoint検出時、または定期的に文字起こし
            transcriber.transcribe(is_endpoint=vad_result["is_endpoint"])

    # スレッド起動
    processing_thread = threading.Thread(target=audio_processing_loop, daemon=True)
    processing_thread.start()

    capture.start()
    print(f"🎤 Audio capture started")
    print(f"🌐 Open http://{config.host}:{config.port} in your browser")

    # FastAPI サーバー起動（メインスレッド）
    uvicorn.run(app, host=config.host, port=config.port, loop=loop)

if __name__ == "__main__":
    main()
```

---

## 実装の進め方（Claude Code への指示順序）

### Phase 1: 基盤（まず音が文字になることを確認）

1. **Step 1**: `config.py` を作成
2. **Step 2**: `audio_capture.py` を作成し、WASAPI loopback で音声が取れることを確認
   - `pyaudiowpatch` が必要になる可能性あり
   - テスト: キャプチャした音声を WAV ファイルに保存して再生できるか確認
3. **Step 3**: `transcriber.py` を作成し、WAV ファイルを文字起こしできることを確認
4. **Step 4**: `vad.py` を作成し、VADが動作することを確認

### Phase 2: リアルタイムパイプライン

5. **Step 5**: `main.py` でキャプチャ → VAD → 文字起こし のパイプラインを繋げる
   - まずはコンソール出力で確認
6. **Step 6**: `server.py` + `static/index.html` でブラウザ表示

### Phase 3: 最適化

7. **Step 7**: 遅延のチューニング
   - `transcribe_interval_sec` を調整
   - `beam_size` を 1 にすると高速化（精度は少し落ちる）
   - `distil-large-v3` と `large-v3` を切り替えて比較
8. **Step 8**: `condition_on_previous_text` の効果を確認
9. **Step 9**: バッファサイズの最適化

### Phase 4: 拡張（後で）

10. **Step 10**: `plugins/translator.py` — DeepL API or ローカル翻訳モデル
11. **Step 11**: `plugins/summarizer.py` — Claude API or ローカル LLM で要約
12. **Step 12**: 話者分離（Speaker Diarization）の追加

---

## パフォーマンス目標

| 指標 | 目標値 | 備考 |
|------|--------|------|
| 文字起こし遅延 | < 2秒 | endpoint検出時は < 1秒 |
| WER（英語） | < 5% | クリーン音声 |
| GPU VRAM使用量 | < 6GB | large-v3 float16 |
| CPU使用率 | < 30% | 音声キャプチャ + VAD |

---

## トラブルシューティング

### WASAPI loopback が見つからない
- `sounddevice.query_devices()` の出力を確認
- Windows の「サウンド設定」でデフォルト出力デバイスを確認
- `pyaudiowpatch` に切り替える

### CUDA out of memory
- `compute_type` を `"int8_float16"` に変更（VRAM削減）
- `beam_size` を 1 に下げる
- `distil-large-v3` に切り替え（VRAM約半分）

### 文字起こしが途切れる・重複する
- `context_duration_sec` を調整（3〜5秒）
- `min_silence_duration_ms` を調整（300〜800ms）
- `condition_on_previous_text` を True にして文脈を活用

### 遅延が大きすぎる
- `transcribe_interval_sec` を 0.5 に下げる
- `beam_size` を 1 にする（greedy decoding）
- `distil-large-v3` に切り替え（6倍速）

---

## 将来の拡張メモ

### 翻訳（英→日）
- **DeepL API**: 月50万文字無料、高精度
- **ローカル**: `facebook/nllb-200-distilled-600M` を transformers で動かす
- 翻訳結果を `broadcast()` の別チャンネルで配信

### 要約
- **Claude API**: 会話のまとまりごとに要約を生成
- endpoint で区切られた会話を蓄積し、一定量溜まったら要約
- 要約結果をブラウザUIの別ペインに表示

### 話者分離
- `pyannote/speaker-diarization-3.1` を使用
- 「Speaker 1: ...」「Speaker 2: ...」の形式で表示
- VRAM追加で約2GB必要
