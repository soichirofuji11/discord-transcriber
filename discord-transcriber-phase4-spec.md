# Phase 4 仕様書 — 翻訳・要約・話者分離

> Phase 1〜3 の `discord-transcriber` に追加する拡張機能の仕様。
> Claude Code に渡して実装してもらう用。

---

## 概要

| 機能 | モデル | コスト目安 | 優先度 |
|------|--------|-----------|--------|
| リアルタイム翻訳（英→日） | Gemini 3.1 Flash-Lite | ~$0.25/1M input tokens | ★★★ |
| 要約（日本語） | Gemini 3 Flash or Claude | ~$0.50〜$3/1M tokens | ★★☆ |
| 話者分離 | pyannote 3.1（ローカル） | 無料（VRAM +2GB） | ★☆☆ |

---

## 1. リアルタイム翻訳（英→日）

### モデル選定

**Gemini 3.1 Flash-Lite** を採用。理由:
- $0.25/1M input, $1.50/1M output（最安クラス）
- Google が翻訳タスクに明示的に推奨
- 2.5 Flash より 2.5x 速い TTFT
- 1M token コンテキスト
- Google AI Studio 経由で無料枠あり（レート制限付き）

**コスト試算（1時間の Discord VC）**:
- 1時間の英語会話 ≈ 8,000〜12,000 words ≈ 15,000〜20,000 tokens（input）
- 日本語翻訳出力 ≈ 同程度のトークン数
- 1時間あたり: input $0.005 + output $0.03 ≈ **約 $0.035/時間**
- 月100時間使っても **$3.5/月** 程度

### アーキテクチャ

```
transcriber.py (確定テキスト)
    │
    ▼
translator.py
    │  Gemini API (非同期)
    ▼
WebSocket → ブラウザ UI
            ├── 英語（原文）
            └── 日本語（翻訳）
```

### plugins/translator.py の仕様

```python
"""
翻訳プラグイン

設計方針:
- transcriber の on_text コールバックから呼ばれる
- 非同期で Gemini API を呼び出し、翻訳結果を broadcast する
- バッチング: 短い文は溜めてからまとめて翻訳（API呼び出し削減）
- フォールバック: API エラー時は原文をそのまま表示
"""

import asyncio
import time
from google import genai  # google-genai パッケージ

class Translator:
    def __init__(self, config):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = "gemini-3.1-flash-lite"
        self.batch_buffer = []
        self.batch_interval_sec = 1.5  # この間隔でバッチ翻訳
        self.last_flush_time = time.time()

        # システムプロンプト（翻訳品質に直結するので重要）
        self.system_prompt = """You are a real-time translator for Discord voice chat.
Translate the following English text to natural Japanese.

Rules:
- Translate naturally, not word-by-word
- Keep gaming/tech terms in their commonly used form (e.g., "GG" → "GG", "lag" → "ラグ")
- If the text contains filler words or incomplete sentences, translate the intent
- Output ONLY the Japanese translation, nothing else
- Keep it concise - this is real-time subtitles"""

    async def translate(self, text: str) -> str:
        """単一テキストを翻訳"""
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=f"{self.system_prompt}\n\nTranslate: {text}",
            config={
                "temperature": 0.1,  # 翻訳は低温で安定させる
                "max_output_tokens": 500,
            }
        )
        return response.text.strip()

    async def translate_batch(self, texts: list[str]) -> list[str]:
        """複数テキストをまとめて翻訳（API呼び出し削減）"""
        if not texts:
            return []

        numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
        prompt = f"""{self.system_prompt}

Translate each numbered line. Output format:
[1] 翻訳1
[2] 翻訳2
...

{numbered}"""

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config={"temperature": 0.1, "max_output_tokens": 1000}
        )

        # パース: [N] 翻訳 の形式を分解
        results = []
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("[") and "]" in line:
                translated = line.split("]", 1)[1].strip()
                results.append(translated)
            elif line:
                results.append(line)

        # 数が合わない場合のフォールバック
        while len(results) < len(texts):
            results.append(texts[len(results)])  # 原文をそのまま

        return results

    def add_to_batch(self, text: str):
        """バッチバッファにテキストを追加"""
        self.batch_buffer.append(text)

    async def flush_batch(self) -> list[dict]:
        """バッチバッファを翻訳して返す"""
        if not self.batch_buffer:
            return []

        texts = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.last_flush_time = time.time()

        try:
            translations = await self.translate_batch(texts)
            return [
                {"original": orig, "translated": trans}
                for orig, trans in zip(texts, translations)
            ]
        except Exception as e:
            print(f"Translation error: {e}")
            # フォールバック: 原文をそのまま返す
            return [
                {"original": t, "translated": f"[翻訳エラー] {t}"}
                for t in texts
            ]

    def should_flush(self) -> bool:
        """バッチをフラッシュすべきか判定"""
        if not self.batch_buffer:
            return False
        elapsed = time.time() - self.last_flush_time
        return elapsed >= self.batch_interval_sec or len(self.batch_buffer) >= 5
```

### config.py への追加

```python
# --- Translation Plugin ---
enable_translation: bool = True
gemini_api_key: str = ""  # 環境変数 GEMINI_API_KEY から読む
translation_model: str = "gemini-3.1-flash-lite"
translation_target_lang: str = "ja"
translation_batch_interval_sec: float = 1.5
```

### main.py への組み込み

```python
# on_text コールバックを拡張
async def on_text(result):
    # 原文を配信
    await broadcast({"type": result["type"], "text": result["text"], "lang": "en"})

    # 翻訳
    if config.enable_translation and result["type"] == "final":
        translator.add_to_batch(result["text"])
        if translator.should_flush():
            translated_results = await translator.flush_batch()
            for item in translated_results:
                await broadcast({
                    "type": "translation",
                    "text": item["translated"],
                    "original": item["original"],
                    "lang": "ja"
                })
```

### ブラウザ UI の更新

```
┌─────────────────────────────────┐
│ [EN] Yeah I think we should     │  ← 白テキスト
│ [JA] うん、そうすべきだと思う     │  ← 水色テキスト
│                                 │
│ [EN] push mid lane together     │
│ [JA] ミッドレーン一緒に押そう     │
│                                 │
│ [EN] careful they have ult...   │  ← グレー（partial）
└─────────────────────────────────┘
```

---

## 2. 要約機能（日本語）

### 設計

リアルタイムではなく、セッション終了後（またはオンデマンド）で要約を生成する。
数十分〜数時間の会話ログを蓄積し、ボタン一つで日本語要約を出す。

### モデル選定

**要約には Gemini 3 Flash** を推奨:
- $0.50/1M input, $3.00/1M output
- 1M token コンテキスト → 数時間の会話ログも一発で入る
- thinking level 調整可能（high で高品質要約）
- 翻訳モデルの 2 倍程度のコストだが、呼び出し頻度が低いので問題なし

**コスト試算（1時間の会話を要約）**:
- 1時間の会話 ≈ 15,000〜20,000 tokens (input)
- 要約出力 ≈ 500〜2,000 tokens
- 1回の要約: **約 $0.02 以下**

**代替**: Claude Sonnet（より高品質な日本語）を使いたい場合も、
インターフェースは同じに保つ（プロバイダー切替可能にする）。

### アーキテクチャ

```
transcriber.py → 確定テキスト → session_store.py（蓄積）
                                      │
                              ユーザーが「要約」ボタンを押す
                                      │
                                      ▼
                              summarizer.py → Gemini API
                                      │
                                      ▼
                              WebSocket → ブラウザ（要約パネル）
```

### plugins/session_store.py の仕様

```python
"""
セッションストア

役割:
- 文字起こし結果をタイムスタンプ付きで蓄積
- 翻訳結果も紐付けて保存
- セッション単位でファイルに永続化（JSON）
- 要約リクエスト時に全テキストを返す
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

@dataclass
class TranscriptEntry:
    timestamp: float
    text: str
    translated: str = ""
    speaker: str = ""  # 話者分離が有効な場合

@dataclass
class Session:
    session_id: str
    start_time: float
    entries: list[TranscriptEntry] = field(default_factory=list)

    def add_entry(self, text: str, translated: str = "", speaker: str = ""):
        self.entries.append(TranscriptEntry(
            timestamp=time.time(),
            text=text,
            translated=translated,
            speaker=speaker,
        ))

    def get_full_text(self, include_translation: bool = False) -> str:
        """要約用に全テキストを結合"""
        lines = []
        for e in self.entries:
            time_str = datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
            prefix = f"[{time_str}]"
            if e.speaker:
                prefix += f" {e.speaker}:"
            lines.append(f"{prefix} {e.text}")
            if include_translation and e.translated:
                lines.append(f"  → {e.translated}")
        return "\n".join(lines)

    def get_duration_minutes(self) -> float:
        if len(self.entries) < 2:
            return 0
        return (self.entries[-1].timestamp - self.entries[0].timestamp) / 60

    def save(self, output_dir: str = "sessions"):
        Path(output_dir).mkdir(exist_ok=True)
        path = Path(output_dir) / f"{self.session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = cls(
            session_id=data["session_id"],
            start_time=data["start_time"]
        )
        for e in data["entries"]:
            session.entries.append(TranscriptEntry(**e))
        return session
```

### plugins/summarizer.py の仕様

```python
"""
要約プラグイン

設計方針:
- セッション全体のテキストを受け取り、日本語で要約
- 長い会話はチャンク分割してから要約 → 最終統合
- 複数の要約テンプレートを用意（概要 / 詳細 / アクションアイテム）
"""

from google import genai

class Summarizer:
    def __init__(self, config):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.summary_model  # "gemini-3-flash" 推奨

    async def summarize(
        self,
        transcript: str,
        style: str = "overview",  # "overview" | "detailed" | "action_items"
        language: str = "ja"
    ) -> str:
        """会話ログを要約する"""

        prompts = {
            "overview": f"""以下はDiscordボイスチャットの文字起こしログです。
内容を日本語で簡潔に要約してください。

要約の形式:
- 全体の概要（2〜3文）
- 主要なトピック（箇条書き）
- 参加者の主な発言や意見

ログ:
{transcript}""",

            "detailed": f"""以下はDiscordボイスチャットの文字起こしログです。
時系列に沿って詳細に日本語で要約してください。

要約の形式:
- 会話の流れを時系列で記述
- 各トピックの議論内容を具体的に
- 重要な発言は引用形式で記載

ログ:
{transcript}""",

            "action_items": f"""以下はDiscordボイスチャットの文字起こしログです。
会話から抽出できるアクションアイテム・決定事項・TODO を日本語でリストアップしてください。

形式:
- 決定事項: 何が決まったか
- TODO: 誰が何をするか
- 未解決: 持ち越しになった議題

ログ:
{transcript}"""
        }

        prompt = prompts.get(style, prompts["overview"])

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 4000,
                "thinking": {"thinking_level": "HIGH"}  # 高品質要約
            }
        )
        return response.text.strip()

    async def summarize_long_session(self, transcript: str, **kwargs) -> str:
        """長時間セッション向け: チャンク分割 → 個別要約 → 統合"""
        MAX_CHUNK_TOKENS = 200_000  # 安全マージンを取る
        APPROX_CHARS_PER_TOKEN = 4

        max_chars = MAX_CHUNK_TOKENS * APPROX_CHARS_PER_TOKEN
        if len(transcript) <= max_chars:
            return await self.summarize(transcript, **kwargs)

        # チャンク分割（行単位で切る）
        lines = transcript.split("\n")
        chunks = []
        current = []
        current_len = 0
        for line in lines:
            if current_len + len(line) > max_chars and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += len(line) + 1
        if current:
            chunks.append("\n".join(current))

        # 各チャンクを要約
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await self.summarize(chunk, **kwargs)
            chunk_summaries.append(f"=== パート {i+1}/{len(chunks)} ===\n{summary}")

        # 統合要約
        combined = "\n\n".join(chunk_summaries)
        final_prompt = f"""以下は長時間の会話ログをパートごとに要約したものです。
これらを統合して、1つの完結した日本語要約を作成してください。
重複を排除し、全体の流れがわかるようにしてください。

{combined}"""

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=final_prompt,
            config={"temperature": 0.3, "max_output_tokens": 4000}
        )
        return response.text.strip()
```

### config.py への追加

```python
# --- Summary Plugin ---
enable_summary: bool = True
summary_model: str = "gemini-3-flash"  # 要約は Flash で十分
sessions_dir: str = "sessions"
```

### server.py への追加（API エンドポイント）

```python
@app.post("/api/summarize")
async def api_summarize(request: SummarizeRequest):
    """要約APIエンドポイント"""
    transcript = current_session.get_full_text()
    summary = await summarizer.summarize(
        transcript,
        style=request.style,  # "overview" | "detailed" | "action_items"
    )
    return {"summary": summary, "duration_minutes": current_session.get_duration_minutes()}

@app.post("/api/session/save")
async def api_save_session():
    """セッションを保存"""
    current_session.save(config.sessions_dir)
    return {"saved": current_session.session_id}

@app.get("/api/session/list")
async def api_list_sessions():
    """保存済みセッション一覧"""
    ...
```

### ブラウザ UI への追加

```
┌───────────────────────────────────────────┐
│ 📝 Transcript           [要約▼] [保存💾]  │
│──────────────────────────────────────────│
│ [EN] Yeah I think we should              │
│ [JA] うん、そうすべきだと思う              │
│ ...                                       │
│──────────────────────────────────────────│
│ 📊 Summary (on demand)                    │
│──────────────────────────────────────────│
│ 【概要】                                  │
│ ゲームの戦略について30分間議論した。        │
│ ミッドレーンのプッシュタイミングと...       │
│                                           │
│ 【主要トピック】                           │
│ ・ミッドレーン戦略                         │
│ ・ドラゴン争奪のタイミング                  │
│ ...                                       │
└───────────────────────────────────────────┘

要約ドロップダウンメニュー:
  ├── 概要（Overview）
  ├── 詳細（Detailed）
  └── アクションアイテム（Action Items）
```

---

## 3. 話者分離

### モデル

**pyannote/speaker-diarization-3.1**
- HuggingFace で配布（要アクセス承認 + HF token）
- VRAM: 追加 ~2GB（Whisper と共存可能、RTX 5070 Ti なら余裕）
- ローカル実行、追加コスト無し

### 制約事項（再掲）

WASAPI loopback はミックスされた1本のストリームなので:
- 声質が似た人の区別は苦手
- 同時発話時の精度が落ちる
- 「Speaker 1」「Speaker 2」のラベルのみ（Discord ユーザー名との紐付けは不可）
- セッション中に新しい人が入ると、ラベルがずれる可能性あり

### 実装方針

リアルタイムで話者分離するのは処理が重いので、
**セグメント単位でオフライン話者分離 → テキストに紐付け** の方式を取る。

```
audio_buffer（確定した発話区間）
    │
    ▼
pyannote diarization（GPU）
    │ タイムスタンプ付き話者ラベル
    ▼
Whisper のセグメントタイムスタンプと突き合わせ
    │
    ▼
「Speaker 1: こう言った」「Speaker 2: こう返した」
```

### plugins/diarizer.py の仕様

```python
"""
話者分離プラグイン

設計方針:
- pyannote/speaker-diarization-3.1 を使用
- VAD endpoint で区切られた音声バッファに対して実行
- Whisper の word_timestamps と突き合わせて話者ラベルを付与
- リアルタイム性よりも正確性を優先（数秒の遅延は許容）
"""

import torch
import numpy as np
from pyannote.audio import Pipeline

class Diarizer:
    def __init__(self, config):
        self.config = config
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.hf_token,
        )
        # GPU に載せる
        self.pipeline.to(torch.device("cuda"))

        # 話者の声紋キャッシュ（セッション中の一貫性を保つ）
        self.speaker_embeddings = {}

    def diarize(self, audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
        """
        音声バッファを話者分離する

        Returns:
            [
                {"speaker": "Speaker 1", "start": 0.0, "end": 2.5},
                {"speaker": "Speaker 2", "start": 2.5, "end": 5.1},
                ...
            ]
        """
        # numpy array → pyannote が受け付ける形式に変換
        # pyannote は {"waveform": tensor, "sample_rate": int} を受ける
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        input_data = {"waveform": waveform, "sample_rate": sample_rate}

        diarization = self.pipeline(input_data)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
            })
        return segments

    def assign_speakers_to_transcript(
        self,
        transcript_segments: list[dict],  # Whisper の segments（start, end, text）
        diarization_segments: list[dict],
    ) -> list[dict]:
        """
        Whisper の文字起こしセグメントに話者ラベルを付与する。
        各文字起こしセグメントの中間時点が、どの話者の区間に含まれるかで判定。
        """
        result = []
        for t_seg in transcript_segments:
            mid_time = (t_seg["start"] + t_seg["end"]) / 2
            speaker = "Unknown"
            for d_seg in diarization_segments:
                if d_seg["start"] <= mid_time <= d_seg["end"]:
                    speaker = d_seg["speaker"]
                    break
            result.append({
                **t_seg,
                "speaker": speaker,
            })
        return result
```

### config.py への追加

```python
# --- Diarization Plugin ---
enable_diarization: bool = False  # デフォルトOFF（重いので明示的に有効化）
hf_token: str = ""  # 環境変数 HF_TOKEN から読む
max_speakers: int = 5  # 最大想定話者数
```

### HuggingFace トークン取得手順

1. https://huggingface.co でアカウント作成
2. https://huggingface.co/pyannote/speaker-diarization-3.1 にアクセス
3. "Accept" で利用規約に同意（必須、承認は即時）
4. https://huggingface.co/settings/tokens でトークン生成
5. 環境変数に設定: `$env:HF_TOKEN = "hf_xxxxx"`

---

## 4. 統合後の全体アーキテクチャ

```
┌────────────┐
│ Discord VC  │
└─────┬──────┘
      │ WASAPI Loopback
      ▼
┌────────────┐    ┌──────────┐
│ Audio      │───▶│ Silero   │
│ Capture    │    │ VAD      │
└─────┬──────┘    └────┬─────┘
      │                │ 音声区間
      ▼                ▼
┌─────────────────────────────┐
│       Audio Buffer           │
└─────┬──────────────┬────────┘
      │              │
      ▼              ▼ (enable_diarization=True)
┌───────────┐  ┌──────────────┐
│ faster-   │  │ pyannote     │
│ whisper   │  │ diarization  │
│ (GPU)     │  │ (GPU)        │
└─────┬─────┘  └──────┬───────┘
      │               │
      ▼               ▼
┌─────────────────────────────┐
│  テキスト + 話者ラベル結合    │
└─────┬───────────────────────┘
      │
      ├──▶ session_store.py（蓄積）
      │
      ├──▶ translator.py ──▶ Gemini 3.1 Flash-Lite API
      │                          │
      │                          ▼
      │                    翻訳テキスト
      │
      ▼
┌──────────────┐
│ WebSocket    │──▶ ブラウザ UI
│ Server       │    ├── 英語原文
│ (FastAPI)    │    ├── 日本語翻訳
│              │    ├── 話者ラベル
│ POST /api/   │    └── 要約パネル
│   summarize  │
└──────┬───────┘
       │ (オンデマンド)
       ▼
  summarizer.py ──▶ Gemini 3 Flash API
       │
       ▼
    日本語要約
```

---

## 5. 環境変数まとめ（.env）

```env
# Gemini API（翻訳 + 要約）
GEMINI_API_KEY=AIzaSy...

# HuggingFace（話者分離、使う場合のみ）
HF_TOKEN=hf_...
```

### Gemini API キー取得手順
1. https://aistudio.google.com にアクセス
2. 「Get API Key」→「Create API Key」
3. 無料枠あり（レート制限付き）

---

## 6. 依存パッケージの追加 (requirements.txt に追記)

```
# Phase 4 追加分
google-genai>=1.0.0          # Gemini API クライアント
pyannote.audio>=3.1.0        # 話者分離（オプション）
python-dotenv>=1.0.0         # .env ファイル読み込み
```

---

## 7. 実装順序（Claude Code への指示）

### Step 10: セッションストア
```
plugins/session_store.py を実装してください。
transcriber の on_text コールバックから呼ばれ、
全テキストをタイムスタンプ付きで蓄積します。
JSON ファイルへの保存・読み込みも実装してください。
```

### Step 11: 翻訳プラグイン
```
plugins/translator.py を実装してください。
Gemini 3.1 Flash-Lite API を使い、英語テキストを日本語に翻訳します。
バッチ翻訳でAPI呼び出しを削減し、エラー時はフォールバックしてください。
main.py に組み込んで、WebSocket で翻訳結果も配信してください。
```

### Step 12: ブラウザ UI 更新
```
static/index.html を更新して、英語原文と日本語翻訳を
2行セットで表示するようにしてください。
翻訳テキストは水色で表示します。
```

### Step 13: 要約プラグイン
```
plugins/summarizer.py を実装してください。
セッションストアから全テキストを取得し、Gemini 3 Flash で日本語要約を生成します。
server.py に POST /api/summarize エンドポイントを追加し、
ブラウザ UI に「要約」ボタンと要約表示パネルを追加してください。
概要 / 詳細 / アクションアイテム の3種類の要約スタイルを選べるようにしてください。
```

### Step 14: 話者分離（オプション）
```
plugins/diarizer.py を実装してください。
pyannote/speaker-diarization-3.1 を使い、
Whisper のセグメントに話者ラベルを付与します。
config.enable_diarization = True の場合のみ有効化してください。
ブラウザ UI で話者ごとに色分け表示してください。
```

### Step 15: 統合テスト
```
全プラグインを有効にした状態で統合テストをしてください。
- Discord VC で英語の会話を流す
- 文字起こし → 翻訳 → セッション保存 が連携して動くことを確認
- 要約ボタンを押して日本語要約が生成されることを確認
- 話者分離が有効な場合、Speaker ラベルが表示されることを確認
遅延が増えすぎていないかも確認してください。
```
