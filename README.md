# Discord VC Real-time Transcriber

Discord のボイスチャット音声をリアルタイムで文字起こしし、ブラウザ上にオーバーレイ表示するローカルアプリケーション。

## 機能

- **リアルタイム文字起こし** — faster-whisper (Large V3 / Distil-Large V3) による高精度 STT
- **VAD (音声区間検出)** — Silero VAD で無音をスキップし、発話の区切りを自動検出
- **リアルタイム翻訳 (英→日)** — Gemini 3.1 Flash-Lite による文脈付き翻訳
- **要約** — Gemini 3 Flash で会話ログを日本語要約 (概要 / 詳細 / アクションアイテム)
- **話者分離** — pyannote 3.1 によるスピーカーラベル付与 (オプション)
- **ブラウザオーバーレイ** — WebSocket でリアルタイム配信、OBS 取り込み対応

## 必要環境

- Windows 11
- NVIDIA GPU (CUDA 12.x)
- Python 3.12

## セットアップ

```bash
# uv をインストール (未導入の場合)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 依存パッケージをインストール
uv sync

# 話者分離を使う場合
uv sync --extra diarization
```

### 環境変数 (.env)

翻訳・要約機能を使う場合、プロジェクトルートに `.env` を作成:

```env
GEMINI_API_KEY=AIzaSy...
HF_TOKEN=hf_...          # 話者分離を使う場合のみ
```

- Gemini API キー: https://aistudio.google.com → Get API Key
- HF トークン: https://huggingface.co/settings/tokens (pyannote モデルの利用規約に事前同意が必要)

## 使い方

```bash
# 基本 (文字起こしのみ)
uv run python main.py

# 翻訳付き
uv run python main.py --translate

# 翻訳 + 要約
uv run python main.py --translate --summarize

# 低遅延モード + 翻訳
uv run python main.py --fast --translate

# フル機能
uv run python main.py --translate --summarize --diarize
```

起動後、ブラウザで `http://127.0.0.1:8765` を開く。

## CLI オプション

| オプション | 説明 | デフォルト |
|---|---|---|
| `--fast` | 低遅延プリセット (distil-large-v3, beam=1, interval=0.5s) | off |
| `--model` | Whisper モデル (large-v3, distil-large-v3, medium, etc.) | distil-large-v3 |
| `--beam-size` | ビームサイズ (1=高速, 5=高精度) | 1 |
| `--language` | 言語コード | en |
| `--interval` | 文字起こし実行間隔 (秒) | 1.0 |
| `--min-silence` | 発話区切り判定の無音時間 (ms) | 1000 |
| `--translate` | 英→日翻訳を有効化 | off |
| `--summarize` | 要約機能を有効化 | off |
| `--diarize` | 話者分離を有効化 | off |
| `--port` | WebSocket サーバーポート | 8765 |

## アーキテクチャ

```
Discord VC (スピーカー出力)
    │ WASAPI Loopback
    ▼
Audio Capture (pyaudiowpatch)
    │ 16kHz mono float32
    ▼
Silero VAD → 無音スキップ / 発話区切り検出
    │
    ▼
faster-whisper (GPU) → 文字起こし
    │
    ├→ Session Store (蓄積 + JSON 保存)
    ├→ Translator → Gemini API → 日本語翻訳
    │
    ▼
FastAPI WebSocket → ブラウザ オーバーレイ
```

## ディレクトリ構成

```
discord-transcriber/
├── main.py                 # エントリーポイント
├── config.py               # 設定 + CLI引数
├── audio_capture.py        # WASAPI loopback 音声キャプチャ
├── vad.py                  # Silero VAD ラッパー
├── transcriber.py          # faster-whisper 推論 + 重複排除
├── server.py               # FastAPI WebSocket サーバー
├── static/
│   └── index.html          # ブラウザ オーバーレイ UI
└── plugins/
    ├── session_store.py    # セッション蓄積・永続化
    ├── translator.py       # Gemini 翻訳 (文脈付き)
    ├── summarizer.py       # Gemini 要約
    └── diarizer.py         # pyannote 話者分離
```
