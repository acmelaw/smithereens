"""Terminal UI — spinner, colors, display helpers using Rich."""

from __future__ import annotations

import random
import sys
import threading
import time

from rich.console import Console

# ── Console ───────────────────────────────────────────────────

console = Console(highlight=False)

# ── Spinner ───────────────────────────────────────────────────

SPINNER_VERBS = [
    "Accomplishing", "Architecting", "Baking", "Beboppin'", "Bloviating",
    "Boondoggling", "Bootstrapping", "Brewing", "Canoodling", "Caramelizing",
    "Cerebrating", "Clauding", "Cogitating", "Combobulating", "Computing",
    "Contemplating", "Cooking", "Crafting", "Crystallizing", "Deliberating",
    "Discombobulating", "Fermenting", "Finagling", "Flibbertigibbeting",
    "Gallivanting", "Generating", "Harmonizing", "Hatching", "Hullaballooing",
    "Ideating", "Imagining", "Inferring", "Lollygagging", "Manifesting",
    "Meandering", "Moonwalking", "Mulling", "Noodling", "Orchestrating",
    "Percolating", "Pondering", "Processing", "Quantumizing", "Razzmatazzing",
    "Recombobulating", "Ruminating", "Simmering", "Synthesizing", "Thinking",
    "Tinkering", "Tomfoolering", "Vibing", "Whatchamacalliting", "Working",
    "Zigzagging",
]

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Spinner:
    """Animated braille spinner shown while waiting for the API."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        self._stop.clear()
        verb = random.choice(SPINNER_VERBS)
        self._thread = threading.Thread(
            target=self._animate, args=(verb,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        # Clear spinner line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _animate(self, verb: str) -> None:
        start = time.monotonic()
        i = 0
        while not self._stop.is_set():
            elapsed = int(time.monotonic() - start)
            frame = FRAMES[i % len(FRAMES)]
            sys.stdout.write(
                f"\r\033[K\033[38;2;204;136;68m{frame} {verb}...\033[0m"
                f" \033[2m({elapsed}s)\033[0m"
            )
            sys.stdout.flush()
            i += 1
            self._stop.wait(0.08)


spinner = Spinner()

# ── Display helpers ───────────────────────────────────────────


def print_error(msg: str) -> None:
    console.print(f"[bold red]smithereens error:[/] {msg}")


def print_warning(msg: str) -> None:
    console.print(f"[yellow]smithereens warning:[/] {msg}")


def print_success(msg: str) -> None:
    console.print(f"[green]{msg}[/]")


def print_dim(msg: str) -> None:
    console.print(f"[dim]{msg}[/]")


def print_tool_header(name: str, detail: str = "") -> None:
    line = f" [bold cyan]{name}[/]"
    if detail:
        line += f" [dim]{detail}[/]"
    console.print(line)


def print_tool_output(output: str, max_lines: int = 30) -> None:
    lines = output.splitlines()
    shown = lines[:max_lines]
    for line in shown:
        console.print(f"  {line}")
    remaining = len(lines) - len(shown)
    if remaining > 0:
        console.print(f"  [dim]... ({remaining} more lines)[/]")


def print_cost(cost: float, input_tokens: int, output_tokens: int) -> None:
    console.print(
        f"\n  [dim]Session cost: ${cost:.4f}"
        f" | {input_tokens:,} input tokens"
        f" | {output_tokens:,} output tokens[/]"
    )


def print_separator() -> None:
    console.print("[dim]" + "─" * console.width + "[/]")


def print_banner(model: str) -> None:
    console.print()
    console.print("[orange3]╭──────────────────────────────────╮[/]")
    console.print(
        "[orange3]│[/]  [bold white]smithereens[/]"
        " [dim]— AI code assistant[/]  [orange3]│[/]"
    )
    console.print("[orange3]╰──────────────────────────────────╯[/]")
    console.print(f"  [dim]model: {model}[/]")
    console.print("  [dim]type /help for commands, ctrl-c to cancel[/]")
    console.print()


def print_prompt() -> None:
    sys.stdout.write("\033[38;2;204;136;68m❯\033[0m ")
    sys.stdout.flush()
