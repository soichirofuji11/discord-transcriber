# Real-time Audio Transcriber

リアルタイム音声文字起こし・翻訳・要約・話者分離ツール。

WASAPI loopback でシステム音声をキャプチャするため、Discord に限らず Zoom / Meet / YouTube / ゲーム等すべてのPC音声に対応する。STT はローカル GPU 実行、翻訳・要約はクラウド API (Gemini) のハイブリッド構成。

## 目次

- [動作環境](#動作環境)
- [セットアップ](#セットアップ)
- [使い方](#使い方)
- [アーキテクチャ](#アーキテクチャ)
- [GPU別 設定ガイド](#gpu別-設定ガイド)
- [コスト目安](#コスト目安)
- [既知の制限事項](#既知の制限事項)
- [トラブルシューティング](#トラブルシューティング)

---

## 動作環境

| 項目 | 要件 |
|------|------|
| OS | Windows 11（WASAPI loopback のため Windows 必須） |
| Python | 3.11 or 3.12 |
| GPU | NVIDIA GPU（CUDA Compute Capability 7.0 以上推奨） |
| VRAM | 4GB 以上（distil-large-v3 使用時）/ 6GB 以上（large-v3 使用時） |

開発・テスト環境: RTX 5070 Ti (16GB VRAM), CUDA 13.1, Windows 11

---

## セットアップ

### 1. uv のインストール

```powershell
irm https://astral.sh/uv/install.ps1 | iex
# シェルを再起動するか、以下を実行
$env:Path = "C:\Users\$env:USERNAME\.local\bin;$env:Path"
```

### 2. 依存パッケージのインストール

```powershell
uv sync
```

話者分離を使う場合:

```powershell
uv sync --extra diarization
```

### 3. PyTorch CUDA index の設定

`pyproject.toml` の index URL を自分の GPU に合わせて変更する（[GPU別 設定ガイド](#gpu別-設定ガイド)参照）:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu126"  # ← GPU に合わせて変更
explicit = true
```

変更後 `uv sync` を再実行。

### 4. 環境変数の設定

プロジェクトルートに `.env` を作成:

```env
# Gemini API（翻訳・要約に必要）
GEMINI_API_KEY=AIzaSy...

# HuggingFace（話者分離を使う場合のみ）
HF_TOKEN=hf_...
```

- Gemini API キー: [Google AI Studio](https://aistudio.google.com) → Get API Key
- HF トークン: [HuggingFace Settings](https://huggingface.co/settings/tokens)

### 5. HuggingFace モデルの利用規約に同意（話者分離を使う場合）

以下3つのページで「Accept」をクリック:

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

---

## 使い方

```powershell
# 基本（文字起こしのみ）
uv run python main.py

# 翻訳付き
uv run python main.py --translate

# 翻訳 + 要約
uv run python main.py --translate --summarize

# フル機能（翻訳 + 要約 + 話者分離）
uv run python main.py --translate --summarize --diarize

# 低遅延モード
uv run python main.py --fast --translate
```

起動後、ブラウザで `http://127.0.0.1:8765` を開く。

### CLI オプション

| オプション | 説明 | デフォルト |
|---|---|---|
| `--fast` | 低遅延プリセット (distil-large-v3, beam=1, interval=0.5s) | off |
| `--model` | Whisper モデル (large-v3, distil-large-v3, medium, etc.) | distil-large-v3 |
| `--beam-size` | ビームサイズ (1=高速, 5=高精度) | 1 |
| `--language` | 言語コード (en, ja, etc.) | en |
| `--interval` | 文字起こし実行間隔 (秒) | 1.0 |
| `--min-silence` | 発話区切り判定の無音時間 (ms) | 1000 |
| `--translate` | 英→日翻訳を有効化 | off |
| `--summarize` | 要約機能を有効化 | off |
| `--diarize` | 話者分離を有効化 | off |
| `--no-context` | 前回テキストの文脈利用を無効化 | off |
| `--port` | WebSocket サーバーポート | 8765 |

---

## アーキテクチャ

```
PC 音声出力 (Discord / Zoom / YouTube / etc.)
    │ WASAPI Loopback (pyaudiowpatch)
    ▼
Audio Capture → 16kHz mono float32
    │
    ▼
Silero VAD → 無音スキップ / 発話区切り (endpoint) 検出
    │
    ▼
faster-whisper (GPU) → 文字起こし
    │                    ├ partial: 発話中、逐次更新
    │                    └ final:   endpoint で確定
    │
    ├──→ Session Store (全文蓄積 + JSON 永続化)
    │
    ├──→ Translator (Gemini 3.1 Flash-Lite)
    │        └ 直近5件の文脈付きで翻訳、バッチ処理
    │
    ├──→ Diarizer (pyannote, 非同期・CPU)
    │        └ endpoint 後にバックグラウンドで実行
    │          結果を既存の行に後付け
    │
    ▼
FastAPI WebSocket → ブラウザ オーバーレイ UI
                     ├ 原文 (白)
                     ├ 翻訳 (水色)
                     ├ 話者ラベル (紫, 後付け)
                     └ 要約パネル (オンデマンド)
```

### 設計上のポイント

- **ブラウザ DOM は直近 25 行のみ**。全テキストはサーバー側 Session Store が保持
- **翻訳は msg_id で対応する文字起こし行の直後に挿入**。バッチ遅延があっても順序が崩れない
- **話者分離は非同期実行**。文字起こし表示を一切ブロックしない
- **要約は Session Store の全文から生成**。DOM からテキストを取得しない

### ディレクトリ構成

```
discord-transcriber/
├── main.py                 # エントリーポイント
├── config.py               # 設定 + CLI引数 + .env読み込み
├── audio_capture.py        # WASAPI loopback 音声キャプチャ
├── vad.py                  # Silero VAD ラッパー
├── transcriber.py          # faster-whisper 推論 + 重複排除
├── server.py               # FastAPI WebSocket サーバー + API
├── static/
│   └── index.html          # ブラウザ オーバーレイ UI
└── plugins/
    ├── session_store.py    # セッション蓄積・永続化
    ├── translator.py       # Gemini 翻訳 (文脈付き)
    ├── summarizer.py       # Gemini 要約 (3種テンプレート)
    └── diarizer.py         # pyannote 話者分離
```

---

## GPU別 設定ガイド

`pyproject.toml` の PyTorch index URL を GPU に合わせて設定する。

| GPU 世代 | 推奨 CUDA index | pyannote 実行 | 備考 |
|----------|----------------|--------------|------|
| RTX 5000 系 (Blackwell) | `cu126` | CPU | sm_120 カーネル未対応 |
| RTX 4000 系 (Ada) | `cu124` or `cu126` | GPU で動く可能性あり | 要動作確認 |
| RTX 3000 系 (Ampere) | `cu121` or `cu124` | GPU | |
| RTX 2000 系 (Turing) | `cu121` | GPU | Compute Capability 7.5 |
| GTX 1000 系 (Pascal) | `cu118` | GPU | CC 6.1、CC 7.0 未満は非推奨 |

変更例（RTX 3000 系の場合）:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

pyannote を GPU で実行する場合は `config.py` の `diarization_device` を `"cuda"` に変更:

```python
diarization_device: str = "cuda"
```

---

## コスト目安

| 機能 | モデル | コスト | 備考 |
|------|--------|--------|------|
| 文字起こし | faster-whisper (distil-large-v3) | 無料 | ローカル GPU 実行 |
| 翻訳 | Gemini 3.1 Flash-Lite | ~$0.035/時間 | 月100時間で ~$3.5 |
| 要約 | Gemini 3 Flash | ~$0.02/回 | オンデマンド実行 |
| 話者分離 | pyannote 3.1 | 無料 | ローカル CPU/GPU 実行 |

---

## 既知の制限事項

### pyannote のバージョン制約

`pyannote.audio` は `>=4.0.0,<4.0.2`（実質 4.0.1）に固定している。

- **pyannote 4.0.2+**: `torch==2.8.0` を厳密にピン留めしており、RTX 5070 Ti 等の Blackwell GPU（`torch>=2.9.0` が必要）と互換性がない。[pyannote/pyannote-audio#1976](https://github.com/pyannote/pyannote-audio/issues/1976)
- **pyannote 3.x**: `huggingface_hub` 1.x と非互換（`use_auth_token` 引数が廃止済み）のため使用不可

### PyTorch weights_only 問題

PyTorch 2.6 以降、`torch.load` のデフォルトが `weights_only=True` に変更された。pyannote のチェックポイントは pickle 形式のため、そのままでは読み込み拒否される。

対処として `torch.serialization.add_safe_globals()` で必要なクラス（`Specifications`, `Problem`, `Resolution`, `DictConfig`, `ListConfig`）をホワイトリスト登録している。これは PyTorch 公式 API であり、モンキーパッチではない。[pyannote/pyannote-audio#1908](https://github.com/pyannote/pyannote-audio/issues/1908)

### pyannote の GPU/CPU 実行

RTX 5070 Ti (Blackwell, sm_120) では pyannote の CUDA カーネルが対応していないため、CPU 実行にフォールバックしている。`config.py` の `diarization_device` で切り替え可能。

- 将来 pyannote が sm_120 をサポートしたら `"cuda"` に変更できる
- RTX 4000 系以前の GPU では `"cuda"` で動作する可能性がある（未検証）

### WASAPI loopback の制約

- Windows 専用（macOS / Linux では動作しない）
- ミックスされた1本のストリームをキャプチャするため、話者分離の精度に限界がある
- 声質が似た話者の区別は苦手
- 同時発話時の精度が低下する

---

## トラブルシューティング

### WASAPI loopback デバイスが見つからない

```powershell
uv run python -c "import pyaudiowpatch as p; a=p.PyAudio(); [print(a.get_device_info_by_index(i)['name']) for i in range(a.get_device_count()) if 'loopback' in a.get_device_info_by_index(i)['name'].lower()]"
```

デバイスが表示されない場合:
- Windows の「サウンド設定」でデフォルト出力デバイスを確認
- ヘッドセット / スピーカーが正しく設定されているか確認

### CUDA out of memory

```powershell
# distil-large-v3 に切り替え（VRAM 約半分）
uv run python main.py --model distil-large-v3

# beam_size を 1 に（VRAM 削減 + 高速化）
uv run python main.py --beam-size 1

# 両方
uv run python main.py --fast
```

### pyannote で "Weights only load failed" エラー

`diarizer.py` の `add_safe_globals` にエラーメッセージに表示されたクラスを追加する:

```python
torch.serialization.add_safe_globals([
    # ... 既存のクラス ...,
    NewClass,  # エラーメッセージに表示されたクラスを追加
])
```

### pyannote で "CUDA error: no kernel image is available" エラー

`config.py` の `diarization_device` を `"cpu"` に設定（デフォルトで `"cpu"`）:

```python
diarization_device: str = "cpu"
```

### 文字起こしが途切れる・遅延が大きい

```powershell
# 低遅延プリセット
uv run python main.py --fast

# 個別調整
uv run python main.py --interval 0.5 --min-silence 500
```

### 翻訳が表示されない

- `.env` に `GEMINI_API_KEY` が設定されているか確認
- `--translate` フラグを付けて起動しているか確認
- コンソールに `[Translator] Error:` が出ていないか確認
