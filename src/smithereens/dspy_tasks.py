"""DSPy structured tasks — commit messages and conversation summarization."""

from __future__ import annotations

import dspy


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
