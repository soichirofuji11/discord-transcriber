"""
Summary plugin — generates Japanese summaries of Discord VC transcripts.
Supports overview, detailed, and action_items styles.
Uses chunk-based summarization for long sessions.
"""

import asyncio

from google import genai

from config import Config

PROMPTS = {
    "overview": """以下はDiscordボイスチャットの文字起こしログです。
内容を日本語で簡潔に要約してください。

要約の形式:
- 全体の概要（2〜3文）
- 主要なトピック（箇条書き）
- 参加者の主な発言や意見

ログ:
{transcript}""",

    "detailed": """以下はDiscordボイスチャットの文字起こしログです。
時系列に沿って詳細に日本語で要約してください。

要約の形式:
- 会話の流れを時系列で記述
- 各トピックの議論内容を具体的に
- 重要な発言は引用形式で記載

ログ:
{transcript}""",

    "action_items": """以下はDiscordボイスチャットの文字起こしログです。
会話から抽出できるアクションアイテム・決定事項・TODOを日本語でリストアップしてください。

形式:
- 決定事項: 何が決まったか
- TODO: 誰が何をするか
- 未解決: 持ち越しになった議題

ログ:
{transcript}""",
}


class Summarizer:
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.summary_model

    async def summarize(self, transcript: str, style: str = "overview") -> str:
        """Summarize a transcript."""
        prompt_template = PROMPTS.get(style, PROMPTS["overview"])
        prompt = prompt_template.format(transcript=transcript)

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 4000},
        )
        return response.text.strip()

    async def summarize_long_session(self, transcript: str, **kwargs) -> str:
        """Chunk-based summarization for long sessions."""
        MAX_CHARS = 200_000 * 4  # ~200k tokens

        if len(transcript) <= MAX_CHARS:
            return await self.summarize(transcript, **kwargs)

        # Split into chunks by line
        lines = transcript.split("\n")
        chunks = []
        current: list[str] = []
        current_len = 0
        for line in lines:
            if current_len + len(line) > MAX_CHARS and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(line)
            current_len += len(line) + 1
        if current:
            chunks.append("\n".join(current))

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = await self.summarize(chunk, **kwargs)
            chunk_summaries.append(f"=== パート {i+1}/{len(chunks)} ===\n{summary}")

        # Merge summaries
        combined = "\n\n".join(chunk_summaries)
        merge_prompt = f"""以下は長時間の会話ログをパートごとに要約したものです。
これらを統合して、1つの完結した日本語要約を作成してください。
重複を排除し、全体の流れがわかるようにしてください。

{combined}"""

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=merge_prompt,
            config={"temperature": 0.3, "max_output_tokens": 4000},
        )
        return response.text.strip()
