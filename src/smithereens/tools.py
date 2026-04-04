"""Tool definitions and implementations.

Each tool is a plain function returning a string result.
Schemas are in OpenAI function-calling format for litellm.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tui import (
    print_dim,
    print_success,
    print_tool_header,
    print_tool_output,
)

if TYPE_CHECKING:
    from .config import Config

# ── Tool schemas (OpenAI format for litellm) ─────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command and return its output. "
                "Use for shell commands, installing packages, running tests, "
                "git operations, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default 30, max 300)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read a file and return its contents with line numbers. "
                "Use to understand code before modifying it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    },
                    "offset": {
                        "type": "number",
                        "description": "Line number to start from (1-indexed, default 1)",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max lines to read (default 2000)",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Perform exact string replacement in a file. "
                "The old_string must match exactly (including whitespace). "
                "Read the file first to get the exact text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact string to find and replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement string",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create or overwrite a file with given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": (
                "Find files matching a glob pattern. "
                "Returns paths sorted by modification time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.ts', 'src/**/*.py')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: cwd)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents using ripgrep (or grep fallback). "
                "Supports regex patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search (default: cwd)",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File pattern filter (e.g. '*.py')",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive search",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]

# ── Permission system ─────────────────────────────────────────

SAFE_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "find", "grep", "rg", "ag",
    "echo", "printf", "pwd", "date", "whoami", "uname", "env", "which",
    "file", "stat", "du", "df", "tree", "sort", "uniq", "diff",
    "md5", "shasum", "type", "readlink", "realpath",
})

SAFE_GIT_SUBCOMMANDS = frozenset({
    "log", "status", "diff", "show", "branch", "remote", "tag",
})


def _is_safe_command(command: str) -> bool:
    parts = command.strip().split()
    if not parts:
        return False
    base = parts[0]
    if base in SAFE_COMMANDS:
        return True
    if base == "git" and len(parts) > 1 and parts[1] in SAFE_GIT_SUBCOMMANDS:
        return True
    return False


def _ask_permission(command: str, config: Config) -> bool:
    """Check if a command is allowed. May set config to allow-all."""
    if config.permission_mode == "allow":
        return True
    if config.permission_mode == "deny":
        return False
    if _is_safe_command(command):
        return True
    if not sys.stdin.isatty():
        return False

    try:
        answer = input(
            f"\033[33m  Allow Bash:\033[0m {command} \033[2m[y/n/a]\033[0m "
        ).strip().lower()
        if answer == "y":
            return True
        if answer == "a":
            config.permission_mode = "allow"
            return True
        return False
    except (EOFError, KeyboardInterrupt):
        return False


# ── Tool error ────────────────────────────────────────────────


class ToolError(Exception):
    """A recoverable tool execution error reported back to the model."""


# ── Dispatcher ────────────────────────────────────────────────


def execute_tool(name: str, args: dict[str, Any], config: Config) -> str:
    """Execute a tool by name, returning the result string."""
    try:
        match name:
            case "bash":
                return _tool_bash(args, config)
            case "read":
                return _tool_read(args)
            case "edit":
                return _tool_edit(args)
            case "write":
                return _tool_write(args)
            case "glob":
                return _tool_glob(args)
            case "grep":
                return _tool_grep(args)
            case _:
                return f"Error: unknown tool '{name}'"
    except ToolError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


# ── Tool implementations ─────────────────────────────────────


def _tool_bash(args: dict[str, Any], config: Config) -> str:
    command: str = args.get("command", "")
    timeout: int = min(int(args.get("timeout", 30)), 300)

    if not command:
        raise ToolError("command is required")

    print_tool_header("Bash", command[:100])

    if not _ask_permission(command, config):
        raise ToolError("Permission denied by user")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out after {timeout}s")

    output = result.stdout
    if result.stderr:
        output = (output + "\n" + result.stderr).strip()

    if result.returncode != 0:
        output = (output + f"\n(exit code: {result.returncode})").strip()

    if output.strip():
        print_tool_output(output.strip())

    return output.strip() or "(no output)"


def _tool_read(args: dict[str, Any]) -> str:
    file_path: str = args.get("file_path", "")
    offset: int = int(args.get("offset", 1))
    limit: int = int(args.get("limit", 2000))

    if not file_path:
        raise ToolError("file_path is required")

    path = Path(file_path).expanduser()
    if not path.is_file():
        raise ToolError(f"File not found: {file_path}")

    print_tool_header("Read", str(path))

    lines = path.read_text().splitlines(keepends=False)
    total = len(lines)
    selected = lines[offset - 1 : offset - 1 + limit]

    numbered = [f"{i:6}\t{line}" for i, line in enumerate(selected, start=offset)]
    print_dim(f"  ({len(selected)} of {total} lines)")
    return "\n".join(numbered)


def _tool_edit(args: dict[str, Any]) -> str:
    file_path: str = args.get("file_path", "")
    old_string: str = args.get("old_string", "")
    new_string: str = args.get("new_string", "")

    if not file_path or not old_string:
        raise ToolError("file_path and old_string are required")

    path = Path(file_path).expanduser()
    if not path.is_file():
        raise ToolError(f"File not found: {file_path}")

    content = path.read_text()
    count = content.count(old_string)

    if count == 0:
        raise ToolError(f"old_string not found in {file_path}")
    if count > 1:
        raise ToolError(
            f"old_string matches {count} locations. Provide more context."
        )

    print_tool_header("Edit", str(path))
    content = content.replace(old_string, new_string, 1)
    path.write_text(content)
    print_success("  Edited successfully")
    return f"Edited {file_path}: replaced 1 occurrence"


def _tool_write(args: dict[str, Any]) -> str:
    file_path: str = args.get("file_path", "")
    content: str = args.get("content", "")

    if not file_path:
        raise ToolError("file_path is required")

    path = Path(file_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    print_tool_header("Write", str(path))
    path.write_text(content)
    lines = content.count("\n") + (1 if content else 0)
    print_success(f"  Wrote {lines} lines")
    return f"Wrote {file_path} ({lines} lines)"


def _tool_glob(args: dict[str, Any]) -> str:
    pattern: str = args.get("pattern", "")
    search_path: str = args.get("path", ".")

    if not pattern:
        raise ToolError("pattern is required")

    base = Path(search_path).expanduser()
    print_tool_header("Glob", pattern)

    try:
        matches = sorted(
            (p for p in base.glob(pattern) if ".git" not in p.parts),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )[:100]
    except Exception as exc:
        raise ToolError(f"Glob failed: {exc}")

    output = "\n".join(str(m) for m in matches)
    print_dim(f"  ({len(matches)} files found)")
    return output or "(no matches)"


def _tool_grep(args: dict[str, Any]) -> str:
    pattern: str = args.get("pattern", "")
    search_path: str = args.get("path", ".")
    file_glob: str = args.get("glob", "")
    case_insensitive: bool = args.get("case_insensitive", False)

    if not pattern:
        raise ToolError("pattern is required")

    print_tool_header("Grep", pattern)

    if shutil.which("rg"):
        cmd = ["rg", "--no-heading", "--line-number", "--color=never"]
        if case_insensitive:
            cmd.append("-i")
        if file_glob:
            cmd.extend(["--glob", file_glob])
        cmd.extend([pattern, search_path])
    else:
        cmd = ["grep", "-rn", "--color=never"]
        if case_insensitive:
            cmd.append("-i")
        cmd.extend([pattern, search_path])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        raise ToolError("Search timed out")

    lines = result.stdout.strip().splitlines()[:250]
    output = "\n".join(lines)
    print_dim(f"  ({len(lines)} matches)")
    return output or "(no matches)"
