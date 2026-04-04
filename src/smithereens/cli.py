"""Interactive REPL and slash commands."""

from __future__ import annotations

import argparse
import readline
import select
import subprocess
import sys
from pathlib import Path

from .agent import (
    _git,
    configure_dspy,
    generate_commit_message,
    in_git_repo,
    process_turn,
    summarize_conversation,
)
from .config import Config
from .session import Session
from .tui import (
    console,
    print_banner,
    print_cost,
    print_dim,
    print_error,
    print_prompt,
    print_separator,
    print_success,
    print_warning,
    spinner,
)

# ── Slash commands ────────────────────────────────────────────


def handle_command(cmd: str, session: Session, config: Config) -> bool:
    """Handle a /command. Returns True if it was recognised."""
    parts = cmd.split(maxsplit=1)
    command = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    match command:
        case "/help":
            _show_help()

        case "/cost":
            print_cost(
                session.cost,
                session.total_input_tokens,
                session.total_output_tokens,
            )

        case "/model":
            if arg:
                config.model = arg
                configure_dspy(config.model)
                print_success(f"Model set to: {config.model}")
            else:
                console.print(f"[dim]Current model:[/] {config.model}")
                console.print("[dim]Change with:[/] /model <model>")

        case "/clear":
            session.messages.clear()
            session.turn_count = 0
            print_success("Conversation cleared")

        case "/compact":
            _compact_with_dspy(session, config)

        case "/save":
            session.save()
            print_success(f"Session saved: {session.session_id}")

        case "/resume":
            if arg:
                _resume_into(arg, session, config)
            else:
                Session.list_sessions(config)

        case "/diff":
            _show_diff()

        case "/commit":
            _handle_commit(session, config)

        case "/quit" | "/exit" | "/q":
            raise SystemExit(0)

        case _ if cmd.startswith("/"):
            print_warning(f"Unknown command: {cmd} (try /help)")

        case _:
            return False

    return True


def _show_help() -> None:
    console.print("\n[bold]Commands:[/]")
    for name, desc in [
        ("/help", "Show this help"),
        ("/cost", "Show session cost"),
        ("/model", "Show/change model"),
        ("/clear", "Clear conversation"),
        ("/compact", "Compact message history (DSPy summarization)"),
        ("/save", "Save current session"),
        ("/resume", "Resume a saved session"),
        ("/commit", "Auto-generate a commit (DSPy)"),
        ("/diff", "Show git diff"),
        ("/quit", "Exit"),
    ]:
        console.print(f"  [cyan]{name:12}[/] — {desc}")
    console.print()


def _compact_with_dspy(session: Session, config: Config) -> None:
    """Use DSPy to summarize conversation history."""
    if len(session.messages) <= 4:
        print_dim("  Conversation too short to compact")
        return
    try:
        configure_dspy(config.model)
        text = "\n".join(
            f"{m['role']}: {m.get('content', '')}"
            for m in session.messages
            if isinstance(m.get("content"), str)
        )
        result = summarize_conversation(conversation=text)
        session.messages = [
            {
                "role": "user",
                "content": f"[Previous conversation summary: {result.summary}]",
            }
        ]
        print_success("Conversation compacted via DSPy summarization")
    except Exception:
        # Fallback: simple truncation
        session.maybe_compact()


def _resume_into(target: str, session: Session, config: Config) -> None:
    """Resume a session, copying state into the current session."""
    try:
        resumed = Session.resume(target, config)
        session.messages = resumed.messages
        session.turn_count = resumed.turn_count
        session.total_input_tokens = resumed.total_input_tokens
        session.total_output_tokens = resumed.total_output_tokens
        session.session_id = resumed.session_id
        print_success(
            f"Resumed session: {resumed.session_id} ({resumed.turn_count} turns)"
        )
    except FileNotFoundError as exc:
        print_error(str(exc))


def _show_diff() -> None:
    if not in_git_repo():
        print_warning("Not in a git repository")
        return
    subprocess.run(["git", "diff", "--stat"])
    print()
    subprocess.run(["git", "diff"])


def _handle_commit(session: Session, config: Config) -> None:
    """Generate a commit message with DSPy and optionally commit."""
    if not in_git_repo():
        print_warning("Not in a git repository")
        return

    # Ensure there are staged changes
    if not _git("diff", "--cached", "--stat"):
        if not _git("diff", "--stat"):
            print_warning("No changes to commit")
            return
        print_dim("  No staged changes. Staging all changes...")
        subprocess.run(["git", "add", "-A"])

    diff = _git("diff", "--cached")[:8000]
    log = _git("log", "--oneline", "-5") or "(no history)"

    try:
        configure_dspy(config.model)
        result = generate_commit_message(diff=diff, recent_log=log)
        message = result.message.strip().strip("\"'")

        console.print(f"\n[bold]Proposed commit message:[/]\n  {message}\n")
        answer = input("\033[33m  Commit? [y/n/e] \033[0m ").strip().lower()

        if answer == "e":
            message = input("  Enter message: ").strip()
            if not message:
                print_dim("  Cancelled")
                return
            answer = "y"

        if answer == "y":
            subprocess.run(["git", "commit", "-m", message], check=True)
            print_success("  Committed!")
        else:
            print_dim("  Cancelled")
    except Exception as exc:
        print_error(f"DSPy commit failed: {exc}")
        process_turn(
            "Look at `git diff --cached` and recent `git log --oneline -5`. "
            "Generate a concise commit message and run `git commit -m '<message>'`.",
            session,
            config,
        )


# ── Main REPL ─────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="smithereens — AI coding assistant",
    )
    parser.add_argument("--resume", "-r", help="Resume a saved session")
    parser.add_argument("--model", "-m", help="Model to use")
    parser.add_argument(
        "--allow", action="store_true", help="Auto-approve all tool calls"
    )
    parser.add_argument("prompt", nargs="*", help="Initial prompt (non-interactive)")
    args = parser.parse_args()

    config = Config.load()
    if args.model:
        config.model = args.model
    if args.allow:
        config.permission_mode = "allow"

    configure_dspy(config.model)

    session = Session(config=config)

    # Resume saved session
    if args.resume:
        try:
            session = Session.resume(args.resume, config)
            print_success(f"Resumed: {session.session_id}")
        except FileNotFoundError as exc:
            print_error(str(exc))
            sys.exit(1)

    # readline history
    histfile = Path.home() / ".smithereens" / "history"
    histfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(str(histfile))
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)

    # Piped (non-interactive) input
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            process_turn(piped, session, config)
        _exit_summary(session, config, histfile)
        return

    # One-shot prompt from CLI args
    if args.prompt:
        process_turn(" ".join(args.prompt), session, config)
        _exit_summary(session, config, histfile)
        return

    # Interactive REPL
    print_banner(config.model)

    try:
        while True:
            try:
                user_input = _read_multiline_input()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                spinner.stop()
                print()
                continue

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                try:
                    if handle_command(user_input, session, config):
                        continue
                except SystemExit:
                    break

            try:
                process_turn(user_input, session, config)
                print()
            except KeyboardInterrupt:
                spinner.stop()
                print("\n")
            except Exception as exc:
                print_error(str(exc))
    finally:
        _exit_summary(session, config, histfile)


def _read_multiline_input() -> str:
    """Read input, collecting multi-line pastes into a single string.

    After readline returns the first line, we drain any data already
    buffered on stdin (i.e. remaining lines from a paste) so the
    entire block is treated as one prompt instead of N separate turns.
    """
    print_prompt()
    first_line = input()
    lines = [first_line]

    # Drain buffered lines from a multi-line paste (50 ms window)
    while select.select([sys.stdin], [], [], 0.05)[0]:
        line = sys.stdin.readline()
        if not line:  # EOF
            break
        lines.append(line.rstrip("\n"))

    return "\n".join(lines)


def _exit_summary(session: Session, config: Config, histfile: Path) -> None:
    """Show cost summary, save session and history on exit."""
    if session.turn_count > 0:
        try:
            session.save()
        except Exception:
            pass
        print()
        print_separator()
        print_cost(
            session.cost,
            session.total_input_tokens,
            session.total_output_tokens,
        )
        print_dim(
            f"  {session.turn_count} turns | model: {config.model}"
            f" | session: {session.session_id}"
        )
        print_dim(f"  resume with: smithereens --resume {session.session_id}")

    console.print("\n[dim]bye![/]")

    try:
        readline.write_history_file(str(histfile))
    except Exception:
        pass
