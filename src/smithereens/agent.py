"""Agent loop — litellm streaming with tool calling.

Uses litellm's native tool calling for all providers. For Ollama models, use
the ``ollama_chat/`` prefix (not ``ollama/``) so litellm routes through the
chat completions endpoint which supports function calling.
"""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

import litellm

from .config import Config
from .prompt import build_system_prompt
from .session import Session
from .tools import TOOL_SCHEMAS, execute_tool
from .tui import console, print_dim, print_error, print_warning, spinner

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
        if kept := session.maybe_compact():
            print_dim(f"  (compacted: kept last {kept} messages)")

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
                        "arguments": json.dumps(tc["arguments"]),
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