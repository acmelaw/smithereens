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
from typing import Any

from .config import Config
from .tui import print_dim, print_success, print_tool_header, print_tool_output


# ── Tool error ────────────────────────────────────────────────


class ToolError(Exception):
    """A recoverable tool execution error reported back to the model."""


# ── Schema helpers ────────────────────────────────────────────


def _param(type_: str, desc: str) -> dict[str, str]:
    return {"type": type_, "description": desc}


def _schema(
    name: str,
    desc: str,
    props: dict[str, dict],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _schema(
        "bash",
        "Execute a bash command and return its output. "
        "Use for shell commands, installing packages, running tests, git operations, etc.",
        {"command": _param("string", "The bash command to execute"),
         "timeout": _param("number", "Timeout in seconds (default 30, max 300)")},
        ["command"],
    ),
    _schema(
        "read",
        "Read a file and return its contents with line numbers. "
        "Use to understand code before modifying it.",
        {"file_path": _param("string", "Absolute or relative path to the file"),
         "offset": _param("number", "Line number to start from (1-indexed, default 1)"),
         "limit": _param("number", "Max lines to read (default 2000)")},
        ["file_path"],
    ),
    _schema(
        "edit",
        "Perform exact string replacement in a file. "
        "The old_string must match exactly (including whitespace). "
        "Read the file first to get the exact text.",
        {"file_path": _param("string", "Path to the file to edit"),
         "old_string": _param("string", "Exact string to find and replace"),
         "new_string": _param("string", "Replacement string")},
        ["file_path", "old_string", "new_string"],
    ),
    _schema(
        "write",
        "Create or overwrite a file with given content.",
        {"file_path": _param("string", "Path to the file to write"),
         "content": _param("string", "Content to write")},
        ["file_path", "content"],
    ),
    _schema(
        "glob",
        "Find files matching a glob pattern. Returns paths sorted by modification time.",
        {"pattern": _param("string", "Glob pattern (e.g. '**/*.ts', 'src/**/*.py')"),
         "path": _param("string", "Directory to search in (default: cwd)")},
        ["pattern"],
    ),
    _schema(
        "grep",
        "Search file contents using ripgrep (or grep fallback). Supports regex patterns.",
        {"pattern": _param("string", "Regex pattern to search for"),
         "path": _param("string", "File or directory to search (default: cwd)"),
         "glob": _param("string", "File pattern filter (e.g. '*.py')"),
         "case_insensitive": _param("boolean", "Case-insensitive search")},
        ["pattern"],
    ),
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
    return base == "git" and len(parts) > 1 and parts[1] in SAFE_GIT_SUBCOMMANDS


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
        if answer == "a":
            config.permission_mode = "allow"
        return answer in ("y", "a")
    except (EOFError, KeyboardInterrupt):
        return False


# ── Dispatcher ────────────────────────────────────────────────


def execute_tool(name: str, args: dict[str, Any], config: Config) -> str:
    """Execute a tool by name, returning the result string."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"
    try:
        return handler(args, config)
    except ToolError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


# ── Tool implementations ─────────────────────────────────────


def _tool_bash(args: dict[str, Any], config: Config) -> str:
    command = args.get("command", "")
    timeout = min(int(args.get("timeout", 30)), 300)

    if not command:
        raise ToolError("command is required")

    print_tool_header("Bash", command[:100])

    if not _ask_permission(command, config):
        raise ToolError("Permission denied by user")

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out after {timeout}s")

    output = result.stdout
    if result.stderr:
        output = f"{output}\n{result.stderr}".strip()
    if result.returncode != 0:
        output = f"{output}\n(exit code: {result.returncode})".strip()

    if output.strip():
        print_tool_output(output.strip())
    return output.strip() or "(no output)"


def _tool_read(args: dict[str, Any], _config: Config) -> str:
    file_path = args.get("file_path", "")
    offset = int(args.get("offset", 1))
    limit = int(args.get("limit", 2000))

    if not file_path:
        raise ToolError("file_path is required")

    path = Path(file_path).expanduser()
    if not path.is_file():
        raise ToolError(f"File not found: {file_path}")

    print_tool_header("Read", str(path))

    lines = path.read_text().splitlines()
    total = len(lines)
    selected = lines[offset - 1 : offset - 1 + limit]

    numbered = [f"{i:6}\t{line}" for i, line in enumerate(selected, start=offset)]
    print_dim(f"  ({len(selected)} of {total} lines)")
    return "\n".join(numbered)


def _tool_edit(args: dict[str, Any], _config: Config) -> str:
    file_path = args.get("file_path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")

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
        raise ToolError(f"old_string matches {count} locations. Provide more context.")

    print_tool_header("Edit", str(path))
    path.write_text(content.replace(old_string, new_string, 1))
    print_success("  Edited successfully")
    return f"Edited {file_path}: replaced 1 occurrence"


def _tool_write(args: dict[str, Any], _config: Config) -> str:
    file_path = args.get("file_path", "")
    content = args.get("content", "")

    if not file_path:
        raise ToolError("file_path is required")

    path = Path(file_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    print_tool_header("Write", str(path))
    path.write_text(content)
    lines = content.count("\n") + (1 if content else 0)
    print_success(f"  Wrote {lines} lines")
    return f"Wrote {file_path} ({lines} lines)"


def _tool_glob(args: dict[str, Any], _config: Config) -> str:
    pattern = args.get("pattern", "")
    search_path = args.get("path", ".")

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

    print_dim(f"  ({len(matches)} files found)")
    return "\n".join(str(m) for m in matches) or "(no matches)"


def _tool_grep(args: dict[str, Any], _config: Config) -> str:
    pattern = args.get("pattern", "")
    search_path = args.get("path", ".")
    file_glob = args.get("glob", "")
    case_insensitive = args.get("case_insensitive", False)

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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise ToolError("Search timed out")

    lines = result.stdout.strip().splitlines()[:250]
    print_dim(f"  ({len(lines)} matches)")
    return "\n".join(lines) or "(no matches)"


# Handler registry — maps tool name to implementation function.
_TOOL_HANDLERS: dict[str, Any] = {
    "bash": _tool_bash,
    "read": _tool_read,
    "edit": _tool_edit,
    "write": _tool_write,
    "glob": _tool_glob,
    "grep": _tool_grep,
}
