"""Configuration for smithereens."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Runtime configuration, populated from env vars and CLI args."""

    model: str = field(
        default_factory=lambda: os.environ.get(
            "SMITHEREENS_MODEL", "ollama_chat/gemma4:26b"
        )
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("SMITHEREENS_MAX_TOKENS", "8192"))
    )
    max_tool_turns: int = 25
    max_messages: int = 40
    sessions_dir: Path = field(
        default_factory=lambda: Path.home() / ".smithereens" / "sessions"
    )
    permission_mode: str = field(
        default_factory=lambda: os.environ.get("SMITHEREENS_PERMISSIONS", "ask")
    )

    # Pricing per 1M tokens (Sonnet 4 defaults)
    price_input: float = 3.00
    price_output: float = 15.00
