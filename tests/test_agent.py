"""Tests for agent — system prompt, StreamResult, tool ID generation, DSPy sigs."""

import os
from dataclasses import fields

from smithereens.agent import StreamResult, _gen_tool_id
from smithereens.dspy_tasks import CommitMessage, ConversationSummary
from smithereens.prompt import build_system_prompt

# ── System prompt ─────────────────────────────────────────────


class TestBuildSystemPrompt:
    def test_includes_working_directory(self):
        prompt = build_system_prompt()
        assert str(os.getcwd()) in prompt

    def test_includes_platform(self):
        prompt = build_system_prompt()
        uname = os.uname()
        assert uname.sysname in prompt

    def test_includes_date(self):
        from datetime import datetime

        prompt = build_system_prompt()
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in prompt

    def test_includes_identity(self):
        prompt = build_system_prompt()
        assert "smithereens" in prompt.lower()

    def test_mentions_tools(self):
        prompt = build_system_prompt()
        assert "tool" in prompt.lower()


# ── StreamResult ──────────────────────────────────────────────


class TestStreamResult:
    def test_default_values(self):
        r = StreamResult()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.finish_reason is None
        assert r.input_tokens == 0
        assert r.output_tokens == 0

    def test_custom_values(self):
        r = StreamResult(
            content="hello",
            tool_calls=[{"name": "bash"}],
            finish_reason="stop",
            input_tokens=100,
            output_tokens=50,
        )
        assert r.content == "hello"
        assert len(r.tool_calls) == 1
        assert r.finish_reason == "stop"

    def test_is_a_dataclass(self):
        assert len(fields(StreamResult)) > 0


# ── Tool ID generation ───────────────────────────────────────


class TestGenToolId:
    def test_starts_with_call_prefix(self):
        tid = _gen_tool_id()
        assert tid.startswith("call_")

    def test_is_unique(self):
        ids = {_gen_tool_id() for _ in range(100)}
        assert len(ids) == 100

    def test_reasonable_length(self):
        tid = _gen_tool_id()
        assert 10 < len(tid) < 50


# ── DSPy signatures ──────────────────────────────────────────


class TestDSPySignatures:
    def test_commit_message_has_required_fields(self):
        # DSPy signatures expose their fields via __annotations__ or model_fields
        sig_fields = CommitMessage.model_fields
        field_names = set(sig_fields.keys())
        assert "diff" in field_names
        assert "recent_log" in field_names
        assert "message" in field_names

    def test_conversation_summary_has_required_fields(self):
        sig_fields = ConversationSummary.model_fields
        field_names = set(sig_fields.keys())
        assert "conversation" in field_names
        assert "summary" in field_names
