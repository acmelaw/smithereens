"""Configuration for smithereens."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULTS = {
    "model": "ollama_chat/gemma4:26b",
    "max_tokens": 16384,
    "permissions": "ask",
}

_ENV_MAP = {
    "SMITHEREENS_MODEL": "model",
    "SMITHEREENS_MAX_TOKENS": "max_tokens",
    "SMITHEREENS_PERMISSIONS": "permissions",
}


def _load_toml(path: Path) -> dict:
    if path.is_file():
        with path.open("rb") as fh:
            return tomllib.load(fh)
    return {}


@dataclass
class Config:
    """Runtime configuration from config file, env vars, and CLI args.

    Resolution order (highest wins): CLI args → env vars → config file → defaults.

    Example ``~/.smithereens/config.toml``::

        model = "ollama_chat/gemma4:26b"
        max_tokens = 16384
        permissions = "ask"   # or "allow"
    """

    model: str = _DEFAULTS["model"]
    max_tokens: int = _DEFAULTS["max_tokens"]
    max_tool_turns: int = 25
    max_messages: int = 200  # ~150k context fits ~200 typical messages
    sessions_dir: Path = field(
        default_factory=lambda: Path.home() / ".smithereens" / "sessions"
    )
    permission_mode: str = _DEFAULTS["permissions"]

    # Pricing per 1M tokens (adjust per model)
    price_input: float = 0.00
    price_output: float = 0.00

    def __post_init__(self) -> None:
        """Apply env var overrides (env wins over dataclass defaults)."""
        _field_map = {"model": str, "max_tokens": int, "permissions": str}
        _attr_map = {"permissions": "permission_mode"}
        for env_key, cfg_key in _ENV_MAP.items():
            if value := os.environ.get(env_key):
                attr = _attr_map.get(cfg_key, cfg_key)
                convert = _field_map[cfg_key]
                setattr(self, attr, convert(value))

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        """Load from ``~/.smithereens/config.toml``, then apply env overrides."""
        if config_path is None:
            config_path = Path.home() / ".smithereens" / "config.toml"
        file = _load_toml(config_path)
        return cls(
            model=file.get("model", _DEFAULTS["model"]),
            max_tokens=int(file.get("max_tokens", _DEFAULTS["max_tokens"])),
            permission_mode=file.get("permissions", _DEFAULTS["permissions"]),
        )
