"""``POST /v1/chat/completions`` for the Home Assistant conversation agent (SPEC §4).

OpenAI-shaped so the custom integration stays trivial, but this is not a proxy. What
the client sends is a request to *answer an utterance*, not a description of how to
answer it:

* client-supplied ``tools`` and ``tool_choice`` are ignored (§3.6);
* client-supplied ``system`` messages are dropped; FarmHub builds its own (§3.6);
* the model name in the body is ignored; the served model comes from config;
* identity comes from headers, never from message content (§3.6).

The request model accepts those fields rather than rejecting them, because the point
is that sending them changes nothing. A 422 would tell a caller which field to try
next; silently ignoring them, and saying so in the log, does not.

M1 answers a single turn. The §4 classifier, retrieval and the tool loop arrive at M6,
which is why no tool is passed to the model here yet.
"""

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from farmhub.api.identity import resolve_identity
from farmhub.core.errors import DependencyUnavailable, LLMError
from farmhub.core.protocols import ChatMessage

# A fixed, server-side system prompt. Built here rather than accepted from the client,
# because a caller that could set the system prompt could undo every instruction in it.
SYSTEM_PROMPT = (
    "You are FarmHub, the assistant for a Norwegian homestead. "
    "Answer briefly and concretely; these answers are usually read aloud. "
    "Reply in the language the question was asked in. "
    "If you do not know something, say so plainly rather than guessing: "
    "a wrong part number or torque figure is worse than no answer."
)


class IncomingMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    # Tool and multimodal payloads arrive as lists; accepted and ignored rather than
    # rejected, since only user text is used.
    content: Any = None


class ChatCompletionRequest(BaseModel):
    """What the HA integration sends. Most of it is deliberately unused."""

    model_config = ConfigDict(extra="ignore")

    messages: list[IncomingMessage] = Field(min_length=1)
    # Accepted and ignored, every one of them (§3.6).
    model: str | None = None
    tools: list[Any] | None = None
    tool_choice: Any = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def latest_user_text(messages: list[IncomingMessage]) -> str | None:
    """The current utterance: the last user message, as text.

    Only the latest one. An action turn may never receive replayed history (§3.5
    rule 4), and while M1 has no action turns, taking the whole history here would
    quietly build the habit the rule forbids.
    """
    for message in reversed(messages):
        if message.role != "user":
            continue
        if isinstance(message.content, str) and message.content.strip():
            return message.content.strip()
        if isinstance(message.content, list):
            parts = [
                part.get("text", "")
                for part in message.content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            joined = " ".join(p for p in parts if p).strip()
            if joined:
                return joined
    return None


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequest) -> dict[str, Any]:
        state = request.app.state.farmhub
        log = state.ctx.log.bind(component="api")

        identity = resolve_identity(request, state.satellites, state.sessions, log)

        if body.tools or body.tool_choice is not None:
            # Not an error, but never silent: a client trying to widen its own
            # capabilities is worth seeing in the log.
            log.warning(
                "client_supplied_tools_ignored",
                session=identity.origin.session,
                satellite=identity.origin.satellite,
                count=len(body.tools or []),
            )
        if any(m.role == "system" for m in body.messages):
            log.warning(
                "client_supplied_system_prompt_ignored",
                session=identity.origin.session,
                satellite=identity.origin.satellite,
            )

        utterance = latest_user_text(body.messages)
        if utterance is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="no user message with text content",
            )

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=utterance),
        ]

        try:
            # tools is empty by construction at M1: the tiered tool list is derived
            # from identity and taint at M6, never from the request.
            response = await state.ctx.llm.chat(messages, tools=())
        except DependencyUnavailable as exc:
            log.warning("llm_unavailable", session=identity.origin.session, error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the language model backend is not available",
            ) from exc
        except LLMError as exc:
            log.error("llm_call_failed", session=identity.origin.session, error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="the language model backend returned an unusable response",
            ) from exc
        except Exception as exc:
            # Anything the backend raises that is not already classified. Logged with a
            # traceback so it can be fixed, but never surfaced as an opaque 500 to a
            # satellite waiting to speak.
            log.exception("llm_call_raised", session=identity.origin.session)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="the language model backend is not available",
            ) from exc

        log.info(
            "chat_completion_served",
            session=identity.origin.session,
            satellite=identity.origin.satellite,
            t0_only=identity.fell_closed,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model or state.ctx.settings.llm.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.content or ""},
                    "finish_reason": response.finish_reason or "stop",
                }
            ],
            # The session FarmHub minted, so the integration can send it back as the
            # conversation id. It is a lookup key, not a credential (§3.6).
            "farmhub": {"session": identity.origin.session},
        }

    return router


__all__ = ["SYSTEM_PROMPT", "ChatCompletionRequest", "build_router", "latest_user_text"]
