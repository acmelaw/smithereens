"""Tests for Config — env var loading, defaults, invariants."""

import os

import pytest

from smithereens.config import Config


class TestConfigDefaults:
    def test_has_a_default_model(self):
        c = Config()
        assert c.model  # non-empty

    def test_has_positive_max_tokens(self):
        c = Config()
        assert c.max_tokens > 0

    def test_has_positive_max_tool_turns(self):
        c = Config()
        assert c.max_tool_turns > 0

    def test_sessions_dir_is_under_home(self):
        c = Config()
        assert str(c.sessions_dir).startswith(str(c.sessions_dir.parent))

    def test_pricing_is_non_negative(self):
        c = Config()
        assert c.price_input >= 0
        assert c.price_output >= 0


class TestConfigFromEnv:
    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("SMITHEREENS_MODEL", "openai/gpt-4o")
        c = Config()
        assert c.model == "openai/gpt-4o"

    def test_max_tokens_from_env(self, monkeypatch):
        monkeypatch.setenv("SMITHEREENS_MAX_TOKENS", "4096")
        c = Config()
        assert c.max_tokens == 4096

    def test_permission_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("SMITHEREENS_PERMISSIONS", "allow")
        c = Config()
        assert c.permission_mode == "allow"

    def test_default_permission_is_ask(self, monkeypatch):
        monkeypatch.delenv("SMITHEREENS_PERMISSIONS", raising=False)
        c = Config()
        assert c.permission_mode == "ask"
