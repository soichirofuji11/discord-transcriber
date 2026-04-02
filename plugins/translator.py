"""
Translation plugin — real-time English to Japanese translation using Gemini.
Batches short texts to reduce API calls.
"""

import asyncio
import time

from google import genai

from config import Config

SYSTEM_PROMPT = """You are a real-time translator for Discord voice chat.
Translate the following English text to natural Japanese.

Rules:
- Translate naturally, not word-by-word
- Keep gaming/tech terms in their commonly used form (e.g., "GG" → "GG", "lag" → "ラグ")
- If the text contains filler words or incomplete sentences, translate the intent
- Output ONLY the Japanese translation, nothing else
- Keep it concise - this is real-time subtitles"""


class Translator:
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model = config.translation_model
        self.batch_buffer: list[str] = []
        self.batch_interval_sec = config.translation_batch_interval_sec
        self.last_flush_time = time.time()

    async def translate(self, text: str) -> str:
        """Translate a single text."""
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=f"{SYSTEM_PROMPT}\n\nTranslate: {text}",
            config={"temperature": 0.1, "max_output_tokens": 500},
        )
        return response.text.strip()

    async def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate multiple texts in a single API call."""
        if not texts:
            return []

        if len(texts) == 1:
            result = await self.translate(texts[0])
            return [result]

        numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
        prompt = f"""{SYSTEM_PROMPT}

Translate each numbered line. Output format:
[1] 翻訳1
[2] 翻訳2
...

{numbered}"""

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=prompt,
            config={"temperature": 0.1, "max_output_tokens": 1000},
        )

        results = []
        for line in response.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("[") and "]" in line:
                translated = line.split("]", 1)[1].strip()
                results.append(translated)
            elif line:
                results.append(line)

        # Fallback: pad with originals if parse returned fewer results
        while len(results) < len(texts):
            results.append(texts[len(results)])

        return results

    def add_to_batch(self, text: str):
        """Add text to the batch buffer."""
        self.batch_buffer.append(text)

    async def flush_batch(self) -> list[dict]:
        """Translate the batch buffer and return results."""
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
            print(f"[Translator] Error: {e}")
            return [
                {"original": t, "translated": t}
                for t in texts
            ]

    def should_flush(self) -> bool:
        """Check if the batch should be flushed."""
        if not self.batch_buffer:
            return False
        elapsed = time.time() - self.last_flush_time
        return elapsed >= self.batch_interval_sec or len(self.batch_buffer) >= 5
