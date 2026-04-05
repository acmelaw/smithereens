"""Git helpers — lightweight wrappers for common git operations."""

from __future__ import annotations

import subprocess


def run(*args: str) -> str:
    """Run a git command and return stripped stdout (empty on failure)."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
        ).stdout.strip()
    except FileNotFoundError:
        return ""


def in_repo() -> bool:
    """Return True if cwd is inside a git work tree."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
