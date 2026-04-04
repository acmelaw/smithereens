"""Configuration for smithereens."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Runtime configuration, populated from env vars and CLI args."""

    model: str = "ollama_chat/gemma4:26b"
    max_tokens: int = 16384         # max output tokens per response
    max_tool_turns: int = 25
    max_messages: int = 200         # ~150k context fits ~200 typical messages
    sessions_dir: Path = field(
        default_factory=lambda: Path.home() / ".smithereens" / "sessions"
    )
    permission_mode: str = "ask"

    # Pricing per 1M tokens (adjust per model)
    price_input: float = 0.00
    price_output: float = 0.00

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        """Load config from ~/.smithereens/config.toml; env vars win over file.

        Example config.toml::

            model = "ollama_chat/gemma4:26b"
            max_tokens = 16384
            permissions = "ask"   # or "allow"
        """
        if config_path is None:
            config_path = Path.home() / ".smithereens" / "config.toml"
        file: dict = {}
        if config_path.is_file():
            with config_path.open("rb") as fh:
                file = tomllib.load(fh)
        return cls(
            model=os.environ.get("SMITHEREENS_MODEL") or file.get("model", "ollama_chat/gemma4:26b"),
            max_tokens=int(os.environ.get("SMITHEREENS_MAX_TOKENS") or file.get("max_tokens", 16384)),
            permission_mode=os.environ.get("SMITHEREENS_PERMISSIONS") or file.get("permissions", "ask"),
        )
