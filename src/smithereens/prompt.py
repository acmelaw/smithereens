"""System prompt builder — assembles environment context for the LLM."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from . import git


def build_system_prompt() -> str:
    """Build the system prompt with environment and project context."""
    uname = os.uname()
    cwd = Path.cwd()
    today = datetime.now().strftime("%Y-%m-%d")

    parts = [
        "You are smithereens, a lightweight AI coding assistant.\n"
        "You have tools for reading, editing, and writing files, "
        "running bash commands, and searching codebases.\n"
        "\n"
        "Environment:\n"
        f"- Working directory: {cwd}\n"
        f"- Platform: {uname.sysname} {uname.machine}\n"
        f"- Date: {today}\n"
        "\n"
        "Guidelines:\n"
        "- Be concise and direct\n"
        "- Use tools to explore before making changes\n"
        "- Prefer simple solutions\n"
        "- When editing files, read them first\n"
        "- Respond in natural language. Use tools when actions are needed.",
    ]

    if claude_md := _load_claude_md_files(cwd):
        parts.append(f"\n# Project Instructions\n\n{claude_md}")

    if git_ctx := _get_git_context():
        parts.append(f"\n# Git Context\n\n{git_ctx}")

    return "\n".join(parts)


def _load_claude_md_files(start: Path) -> str:
    """Walk up collecting CLAUDE.md files (root first, most specific last)."""
    found: list[Path] = []
    d = start.resolve()
    while True:
        for candidate in [d / "CLAUDE.md", d / ".claude" / "CLAUDE.md"]:
            if candidate.is_file():
                found.append(candidate)
        parent = d.parent
        if parent == d:
            break
        d = parent

    home_claude = Path.home() / ".claude" / "CLAUDE.md"
    if home_claude.is_file() and home_claude not in found:
        found.append(home_claude)

    return "\n\n".join(f"## From {f}\n\n{f.read_text()}" for f in reversed(found))


def _get_git_context() -> str:
    """Gather current branch, status, recent commits."""
    if not git.in_repo():
        return ""

    branch = git.run("branch", "--show-current")
    parts = [f"Branch: {branch or 'detached'}"]

    if status := git.run("status", "--short"):
        parts.append("Uncommitted changes:\n" + "\n".join(status.splitlines()[:20]))

    if log := git.run("log", "--oneline", "-5"):
        parts.append(f"Recent commits:\n{log}")

    return "\n\n".join(parts)
