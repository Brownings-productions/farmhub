"""Request parameters shared by every case, and the reasoning tripwire.

FarmHub answers are read aloud (SPEC §1), so a model that reasons out loud before
answering is not merely slower: in the first run it spent the whole token budget
thinking and never reached an answer, which scored as a part-number failure it had not
actually made.

Thinking is therefore switched off through the chat template. That is a request
parameter, and a backend or template that does not honour it will simply ignore it and
answer exactly as before. So every case also checks the reply for reasoning and
records what it found: a silently ignored parameter must not be able to pass as a
clean run.
"""

from __future__ import annotations

import re
from typing import Any

# Honoured by the Qwen3.x chat templates. Sent on every request, including the tool and
# classifier calls, so one run means one mode throughout.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# Reasoning as it actually appeared: vLLM emits a `reasoning_content` field when a
# reasoning parser is configured, and otherwise the reasoning lands in `content`.
# These are the openers seen in the 2026-09-22 run plus the usual tag forms.
REASONING_MARKERS = re.compile(
    r"<think>|</think>|here'?s a thinking process|^\s*(okay|alright),? (so )?(let'?s|i need to)",
    re.I | re.M,
)


def payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_completion_tokens: int,
    **extra: Any,
) -> dict[str, Any]:
    """A chat request with thinking disabled and an explicit token budget."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }
    body.update(extra)
    return body


def reasoning_in(message: dict[str, Any]) -> str | None:
    """Why this reply counts as reasoning, or None if it looks like a plain answer.

    Returns the reason rather than a bool so the run records *what* was seen, which is
    what distinguishes "the parameter was ignored" from "the model is just chatty".
    """
    if message.get("reasoning_content"):
        return "reasoning_content field present"
    content = message.get("content") or ""
    if match := REASONING_MARKERS.search(content):
        return f"content opens with reasoning: {match.group(0)[:40]!r}"
    return None


def truncated(choice: dict[str, Any]) -> bool:
    """True when the model ran out of budget mid-answer.

    A truncated reply is scored as a failure either way, but knowing it was cut off
    separates "answered wrongly" from "never finished answering" — the distinction the
    first run got wrong.
    """
    return choice.get("finish_reason") == "length"


__all__ = [
    "CHAT_TEMPLATE_KWARGS",
    "REASONING_MARKERS",
    "payload",
    "reasoning_in",
    "truncated",
]
