"""Session management — message history, persistence, token tracking."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config


@dataclass
class Session:
    """Holds conversation state for one interactive session."""

    config: Config
    session_id: str = field(
        default_factory=lambda: f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    )
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turn_count: int = 0

    # ── Message helpers ───────────────────────────────────────

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.turn_count += 1

    def update_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    @property
    def cost(self) -> float:
        return (
            self.total_input_tokens * self.config.price_input
            + self.total_output_tokens * self.config.price_output
        ) / 1_000_000

    # ── Compaction ────────────────────────────────────────────

    def maybe_compact(self) -> int | None:
        """Trim old messages if history exceeds max_messages.

        Returns the number of messages kept, or None if no compaction was needed.
        """
        if len(self.messages) <= self.config.max_messages:
            return None
        keep = self.config.max_messages
        self.messages = self.messages[-keep:]
        return keep

    # ── Persistence ───────────────────────────────────────────

    def save(self) -> Path:
        """Save session to disk. Returns the file path."""
        self.config.sessions_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": self.session_id,
            "cwd": str(Path.cwd()),
            "model": self.config.model,
            "messages": self.messages,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turns": self.turn_count,
        }
        path = self.config.sessions_dir / f"{self.session_id}.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def _recent_sessions(cls, sessions_dir: Path, limit: int = 10) -> list[Path]:
        """Return up to *limit* session files, newest first."""
        if not sessions_dir.exists():
            return []
        return sorted(
            sessions_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

    @classmethod
    def resume(cls, target: str, config: Config) -> Session:
        """Resume a session from disk by number or ID prefix."""
        sessions_dir = config.sessions_dir
        path: Path | None = None

        if target.isdigit():
            files = cls._recent_sessions(sessions_dir, limit=100)
            idx = int(target) - 1
            if 0 <= idx < len(files):
                path = files[idx]
        elif matches := sorted(sessions_dir.glob(f"{target}*.json")):
            path = matches[0]

        if not path or not path.is_file():
            raise FileNotFoundError(f"Session not found: {target}")

        data = json.loads(path.read_text())
        session = cls(config=config)
        session.session_id = data["id"]
        session.messages = data["messages"]
        session.total_input_tokens = data.get("input_tokens", 0)
        session.total_output_tokens = data.get("output_tokens", 0)
        session.turn_count = data.get("turns", 0)

        if data.get("model"):
            config.model = data["model"]

        return session

    @classmethod
    def list_sessions(cls, config: Config) -> list[dict[str, Any]]:
        """Return metadata for the most recent saved sessions."""
        return [
            json.loads(f.read_text())
            for f in cls._recent_sessions(config.sessions_dir)
        ]
