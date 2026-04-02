"""
Translation plugin — real-time English to Japanese translation using Gemini.
Uses conversation context for accurate pronoun/reference resolution.
"""

import asyncio
import time

from google import genai

from config import Config
from plugins.session_store import Session

SYSTEM_PROMPT = """You are a real-time translator for Discord voice chat.
Translate English text to natural Japanese.

Rules:
- Translate naturally, not word-by-word
- Keep gaming/tech terms in their commonly used form (e.g., "GG" → "GG", "lag" → "ラグ")
- If the text contains filler words or incomplete sentences, translate the intent
- Output ONLY the Japanese translation, nothing else
- Keep it concise - this is real-time subtitles"""


class Translator:
    def __init__(self, config: Config, session: Session):
        self.config = config
        self.session = session
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.translation_model
        self.context_lines = config.translation_context_lines
        self.batch_buffer: list[dict] = []  # [{"text": ..., "msg_id": ...}]
        self.batch_interval_sec = config.translation_batch_interval_sec
        self.last_flush_time = time.time()

    def _build_prompt(self, text: str, context: list[str]) -> str:
        """Build a translation prompt with conversation context."""
        parts = [SYSTEM_PROMPT, ""]
        if context:
            parts.append("Previous conversation (for context only, do NOT translate):")
            for line in context:
                parts.append(f"- \"{line}\"")
            parts.append("")
        parts.append(f"Translate ONLY this line to Japanese:\n\"{text}\"")
        return "\n".join(parts)

    async def _call_api(self, prompt: str, max_tokens: int = 500) -> str:
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config={"temperature": 0.1, "max_output_tokens": max_tokens},
        )
        return response.text.strip()

    async def translate_with_context(self, text: str, context: list[str]) -> str:
        """Translate a single text with conversation context."""
        prompt = self._build_prompt(text, context)
        return await self._call_api(prompt)

    def add_to_batch(self, text: str, msg_id: int = 0):
        """Add text to the batch buffer with its message ID."""
        self.batch_buffer.append({"text": text, "msg_id": msg_id})

    async def flush_batch(self) -> list[dict]:
        """Translate the batch buffer and return results with msg_ids."""
        if not self.batch_buffer:
            return []

        items = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.last_flush_time = time.time()

        results = []
        # Get context from session: entries BEFORE the batch items
        # The batch items are the most recent entries, so exclude them
        n_batch = len(items)

        try:
            for i, item in enumerate(items):
                # Context = recent session entries before this batch + earlier batch items
                session_context = self.session.get_recent_texts(
                    n=self.context_lines,
                    exclude_last=n_batch - i,
                )
                # Add earlier batch items as additional context
                for prev in items[:i]:
                    session_context.append(prev["text"])
                # Keep only the last N
                context = session_context[-self.context_lines:]

                translated = await self.translate_with_context(item["text"], context)
                results.append({
                    "original": item["text"],
                    "translated": translated,
                    "msg_id": item["msg_id"],
                })
        except Exception as e:
            print(f"[Translator] Error: {e}")
            # Fallback for remaining items
            for item in items[len(results):]:
                results.append({
                    "original": item["text"],
                    "translated": item["text"],
                    "msg_id": item["msg_id"],
                })

        return results

    def should_flush(self) -> bool:
        """Check if the batch should be flushed."""
        if not self.batch_buffer:
            return False
        elapsed = time.time() - self.last_flush_time
        return elapsed >= self.batch_interval_sec or len(self.batch_buffer) >= 5
