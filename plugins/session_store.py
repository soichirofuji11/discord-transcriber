"""
Session store — accumulates transcription results with timestamps.
Supports JSON persistence for session save/load.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class TranscriptEntry:
    timestamp: float
    text: str
    translated: str = ""
    speaker: str = ""


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

    def update_translation(self, original: str, translated: str):
        """Update the translation for a matching entry (latest match)."""
        for entry in reversed(self.entries):
            if entry.text == original and not entry.translated:
                entry.translated = translated
                return

    def get_full_text(self, include_translation: bool = False) -> str:
        """Return all text joined for summarization."""
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

    def get_entry_count(self) -> int:
        return len(self.entries)

    def save(self, output_dir: str = "sessions"):
        Path(output_dir).mkdir(exist_ok=True)
        path = Path(output_dir) / f"{self.session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        return str(path)

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = cls(
            session_id=data["session_id"],
            start_time=data["start_time"],
        )
        for e in data["entries"]:
            session.entries.append(TranscriptEntry(**e))
        return session

    @classmethod
    def new(cls) -> "Session":
        now = time.time()
        session_id = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
        return cls(session_id=session_id, start_time=now)
