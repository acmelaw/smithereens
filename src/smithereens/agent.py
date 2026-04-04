"""Agent loop — litellm streaming with native tool calling + DSPy structured tasks.

Uses litellm's native tool calling for all providers. For Ollama models, use
the ``ollama_chat/`` prefix (not ``ollama/``) so litellm routes through the
chat completions endpoint which supports function calling.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
import litellm

from .config import Config
from .session import Session
from .tools import TOOL_SCHEMAS, execute_tool
from .tui import console, print_dim, print_error, print_warning, spinner

# ── DSPy structured modules ──────────────────────────────────


class CommitMessage(dspy.Signature):
    """Generate a concise conventional commit message from a git diff."""

    diff: str = dspy.InputField(desc="output of git diff --cached")
    recent_log: str = dspy.InputField(desc="recent git log --oneline")
    message: str = dspy.OutputField(
        desc="commit message: a short imperative subject line, optionally followed by a blank line and body"
    )


class ConversationSummary(dspy.Signature):
    """Summarize conversation history to preserve key context while reducing tokens."""

    conversation: str = dspy.InputField(desc="conversation messages as text")
    summary: str = dspy.OutputField(
        desc="concise summary preserving key decisions, files changed, and context"
    )


generate_commit_message = dspy.ChainOfThought(CommitMessage)
summarize_conversation = dspy.ChainOfThought(ConversationSummary)


def configure_dspy(model: str) -> None:
    """Configure DSPy to use the same model via litellm."""
    dspy.configure(lm=dspy.LM(model, cache=False))


# ── Git helpers ───────────────────────────────────────────────


def _git(*args: str) -> str:
    """Run a git command and return stripped stdout (empty on failure)."""
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True,
        ).stdout.strip()
    except FileNotFoundError:
        return ""


def in_git_repo() -> bool:
    """Return True if cwd is inside a git work tree."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ── System prompt ─────────────────────────────────────────────


def build_system_prompt() -> str:
    """Build the system prompt with environment and project context."""
    import os
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

    return "\n\n".join(
        f"## From {f}\n\n{f.read_text()}" for f in reversed(found)
    )


def _get_git_context() -> str:
    """Gather current branch, status, recent commits."""
    if not in_git_repo():
        return ""

    branch = _git("branch", "--show-current")
    parts = [f"Branch: {branch or 'detached'}"]

    if status := _git("status", "--short"):
        parts.append("Uncommitted changes:\n" + "\n".join(status.splitlines()[:20]))

    if log := _git("log", "--oneline", "-5"):
        parts.append(f"Recent commits:\n{log}")

    return "\n\n".join(parts)


# ── Streaming response ────────────────────────────────────────


@dataclass
class StreamResult:
    """Parsed result of one streamed LLM response."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def _gen_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def stream_response(session: Session, config: Config, system_prompt: str) -> StreamResult:
    """Stream one LLM call, printing text in real-time, accumulating tool calls."""
    spinner.start()
    first_token = True
    content_parts: list[str] = []
    tc_acc: dict[int, dict[str, str]] = {}
    input_tokens = 0
    output_tokens = 0
    finish_reason: str | None = None

    messages = [{"role": "system", "content": system_prompt}] + session.messages

    try:
        response = litellm.completion(
            model=config.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            stream=True,
            max_tokens=config.max_tokens,
            num_retries=3,
        )

        for chunk in response:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Stream text
            if delta and getattr(delta, "content", None):
                if first_token:
                    spinner.stop()
                    first_token = False
                content_parts.append(delta.content)
                sys.stdout.write(delta.content)
                sys.stdout.flush()

            # Accumulate tool calls
            if delta and getattr(delta, "tool_calls", None):
                if first_token:
                    spinner.stop()
                    first_token = False
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_acc:
                        tc_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tc_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tc_acc[idx]["name"] = tc.function.name
                            console.print(
                                f"\n  [dim]\\[calling {tc.function.name}...][/]"
                            )
                        if tc.function.arguments:
                            tc_acc[idx]["arguments"] += tc.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = getattr(chunk.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(chunk.usage, "completion_tokens", 0) or 0

    except KeyboardInterrupt:
        spinner.stop()
        raise
    except Exception as exc:
        spinner.stop()
        print_error(f"LLM error: {exc}")
        raise
    finally:
        if first_token:
            spinner.stop()

    content = "".join(content_parts)
    if content:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Build tool call list
    parsed: list[dict[str, Any]] = []
    for idx in sorted(tc_acc):
        entry = tc_acc[idx]
        try:
            args = json.loads(entry["arguments"]) if entry["arguments"] else {}
        except json.JSONDecodeError:
            args = {}
        parsed.append({
            "id": entry["id"] or _gen_tool_id(),
            "name": entry["name"],
            "arguments": args,
            "raw_arguments": entry["arguments"],
        })

    return StreamResult(
        content=content,
        tool_calls=parsed,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ── Turn processor ────────────────────────────────────────────


def process_turn(user_input: str, session: Session, config: Config) -> None:
    """Process a full user turn: send message → [tool loop] → final response."""
    session.add_user_message(user_input)
    system_prompt = build_system_prompt()
    result: StreamResult | None = None

    for _round in range(config.max_tool_turns):
        session.maybe_compact()

        result = stream_response(session, config, system_prompt)
        session.update_usage(result.input_tokens, result.output_tokens)

        # Record assistant message in history
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": result.content or None,
        }
        if result.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["raw_arguments"],
                    },
                }
                for tc in result.tool_calls
            ]
        session.messages.append(assistant_msg)

        # No tool calls → done
        if not result.tool_calls:
            break

        # Execute tools and record results
        for tc in result.tool_calls:
            tool_result = execute_tool(tc["name"], tc["arguments"], config)
            session.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_result,
            })

        sys.stdout.write("\n")
    else:
        print_warning(f"Max tool turns ({config.max_tool_turns}) reached")

    # Per-turn cost
    if result:
        cost = (
            result.input_tokens * config.price_input
            + result.output_tokens * config.price_output
        ) / 1_000_000
        print_dim(
            f"  ${cost:.4f} · {result.input_tokens:,} in / {result.output_tokens:,} out"
        )