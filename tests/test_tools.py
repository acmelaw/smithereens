"""Tests for tools — each tool's contract, permission system, dispatch."""

import os
import stat

import pytest

from smithereens.config import Config
from smithereens.tools import (
    TOOL_SCHEMAS,
    ToolError,
    _is_safe_command,
    execute_tool,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def allow_config():
    """Config that auto-approves all bash commands."""
    c = Config()
    c.permission_mode = "allow"
    return c


@pytest.fixture
def deny_config():
    """Config that blocks all non-safe commands."""
    c = Config()
    c.permission_mode = "deny"
    return c


# ── Tool schemas ──────────────────────────────────────────────


class TestToolSchemas:
    """Validate structural invariants of tool definitions."""

    def test_all_schemas_have_required_keys(self):
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"

    def test_all_tool_names_are_unique(self):
        names = [s["function"]["name"] for s in TOOL_SCHEMAS]
        assert len(names) == len(set(names))

    def test_expected_tools_exist(self):
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        for expected in ("bash", "read", "edit", "write", "glob", "grep"):
            assert expected in names


# ── Permission system ─────────────────────────────────────────


class TestSafeCommands:
    @pytest.mark.parametrize(
        "cmd",
        ["ls", "cat foo.py", "head -n 10 x", "git status", "git log --oneline"],
    )
    def test_read_only_commands_are_safe(self, cmd):
        assert _is_safe_command(cmd) is True

    @pytest.mark.parametrize(
        "cmd",
        ["rm -rf /", "pip install x", "git push", "curl http://x", "python x.py"],
    )
    def test_mutating_commands_are_unsafe(self, cmd):
        assert _is_safe_command(cmd) is False

    def test_empty_command_is_unsafe(self):
        assert _is_safe_command("") is False


# ── Dispatcher ────────────────────────────────────────────────


class TestExecuteTool:
    def test_unknown_tool_returns_error_string(self, allow_config):
        result = execute_tool("nonexistent", {}, allow_config)
        assert "Error" in result

    def test_tool_errors_are_caught_not_raised(self, allow_config):
        # Missing required arg should return error string, not raise
        result = execute_tool("read", {}, allow_config)
        assert "Error" in result

    def test_exceptions_are_caught_not_raised(self, allow_config):
        result = execute_tool("read", {"file_path": "/nonexistent/xyz"}, allow_config)
        assert "Error" in result


# ── Bash tool ─────────────────────────────────────────────────


class TestBashTool:
    def test_runs_command_and_returns_output(self, allow_config):
        result = execute_tool("bash", {"command": "echo hello"}, allow_config)
        assert "hello" in result

    def test_captures_stderr(self, allow_config):
        result = execute_tool(
            "bash", {"command": "echo err >&2"}, allow_config
        )
        assert "err" in result

    def test_reports_nonzero_exit_code(self, allow_config):
        result = execute_tool("bash", {"command": "exit 42"}, allow_config)
        assert "exit code: 42" in result

    def test_missing_command_returns_error(self, allow_config):
        result = execute_tool("bash", {}, allow_config)
        assert "Error" in result

    def test_denied_in_deny_mode(self, deny_config):
        result = execute_tool(
            "bash", {"command": "rm -rf /"}, deny_config
        )
        assert "Error" in result
        assert "denied" in result.lower() or "Permission" in result

    def test_safe_commands_pass_in_ask_mode(self):
        """Safe commands should work without interactive prompt even in ask mode."""
        config = Config()
        config.permission_mode = "ask"
        result = execute_tool("bash", {"command": "echo safe"}, config)
        assert "safe" in result

    def test_timeout_is_capped(self, allow_config):
        # Timeout > 300 should be capped to 300 (not error)
        result = execute_tool(
            "bash", {"command": "echo ok", "timeout": 9999}, allow_config
        )
        assert "ok" in result


# ── Read tool ─────────────────────────────────────────────────


class TestReadTool:
    def test_reads_existing_file(self, tmp_path, allow_config):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = execute_tool("read", {"file_path": str(f)}, allow_config)
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_includes_line_numbers(self, tmp_path, allow_config):
        f = tmp_path / "test.txt"
        f.write_text("aaa\nbbb\n")
        result = execute_tool("read", {"file_path": str(f)}, allow_config)
        assert "1" in result  # line number

    def test_respects_offset(self, tmp_path, allow_config):
        f = tmp_path / "test.txt"
        f.write_text("a\nb\nc\nd\n")
        result = execute_tool(
            "read", {"file_path": str(f), "offset": 3}, allow_config
        )
        assert "a" not in result or "3" in result  # starts at line 3
        assert "c" in result

    def test_respects_limit(self, tmp_path, allow_config):
        f = tmp_path / "test.txt"
        f.write_text("\n".join(f"line{i}" for i in range(100)))
        result = execute_tool(
            "read", {"file_path": str(f), "limit": 5}, allow_config
        )
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) <= 5

    def test_missing_file_returns_error(self, allow_config):
        result = execute_tool(
            "read", {"file_path": "/tmp/no_such_file_xyz"}, allow_config
        )
        assert "Error" in result

    def test_missing_path_returns_error(self, allow_config):
        result = execute_tool("read", {}, allow_config)
        assert "Error" in result


# ── Edit tool ─────────────────────────────────────────────────


class TestEditTool:
    def test_replaces_exact_string(self, tmp_path, allow_config):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = execute_tool(
            "edit",
            {"file_path": str(f), "old_string": "hello", "new_string": "goodbye"},
            allow_config,
        )
        assert "Edited" in result
        assert f.read_text() == "goodbye world"

    def test_error_when_string_not_found(self, tmp_path, allow_config):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = execute_tool(
            "edit",
            {"file_path": str(f), "old_string": "xyz", "new_string": "abc"},
            allow_config,
        )
        assert "Error" in result
        assert f.read_text() == "hello world"  # unchanged

    def test_error_when_ambiguous(self, tmp_path, allow_config):
        f = tmp_path / "test.txt"
        f.write_text("aa bb aa")
        result = execute_tool(
            "edit",
            {"file_path": str(f), "old_string": "aa", "new_string": "cc"},
            allow_config,
        )
        assert "Error" in result
        assert f.read_text() == "aa bb aa"  # unchanged

    def test_missing_file_returns_error(self, allow_config):
        result = execute_tool(
            "edit",
            {
                "file_path": "/tmp/no_such_xyz",
                "old_string": "x",
                "new_string": "y",
            },
            allow_config,
        )
        assert "Error" in result


# ── Write tool ────────────────────────────────────────────────


class TestWriteTool:
    def test_creates_new_file(self, tmp_path, allow_config):
        f = tmp_path / "new.txt"
        result = execute_tool(
            "write", {"file_path": str(f), "content": "hello"}, allow_config
        )
        assert "Wrote" in result
        assert f.read_text() == "hello"

    def test_creates_parent_directories(self, tmp_path, allow_config):
        f = tmp_path / "a" / "b" / "c.txt"
        result = execute_tool(
            "write", {"file_path": str(f), "content": "deep"}, allow_config
        )
        assert "Wrote" in result
        assert f.read_text() == "deep"

    def test_overwrites_existing_file(self, tmp_path, allow_config):
        f = tmp_path / "test.txt"
        f.write_text("old")
        execute_tool(
            "write", {"file_path": str(f), "content": "new"}, allow_config
        )
        assert f.read_text() == "new"

    def test_missing_path_returns_error(self, allow_config):
        result = execute_tool("write", {}, allow_config)
        assert "Error" in result


# ── Glob tool ─────────────────────────────────────────────────


class TestGlobTool:
    def test_finds_matching_files(self, tmp_path, allow_config):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = execute_tool(
            "glob", {"pattern": "*.py", "path": str(tmp_path)}, allow_config
        )
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_no_matches_returns_indicator(self, tmp_path, allow_config):
        result = execute_tool(
            "glob", {"pattern": "*.xyz", "path": str(tmp_path)}, allow_config
        )
        assert "no matches" in result.lower() or result.strip() == ""

    def test_missing_pattern_returns_error(self, allow_config):
        result = execute_tool("glob", {}, allow_config)
        assert "Error" in result


# ── Grep tool ─────────────────────────────────────────────────


class TestGrepTool:
    def test_finds_pattern_in_files(self, tmp_path, allow_config):
        (tmp_path / "a.txt").write_text("foo bar baz\n")
        (tmp_path / "b.txt").write_text("nothing here\n")
        result = execute_tool(
            "grep", {"pattern": "foo", "path": str(tmp_path)}, allow_config
        )
        assert "foo" in result

    def test_no_matches_returns_indicator(self, tmp_path, allow_config):
        (tmp_path / "a.txt").write_text("hello\n")
        result = execute_tool(
            "grep", {"pattern": "zzzzz", "path": str(tmp_path)}, allow_config
        )
        assert "no matches" in result.lower() or result.strip() == ""

    def test_missing_pattern_returns_error(self, allow_config):
        result = execute_tool("grep", {}, allow_config)
        assert "Error" in result
