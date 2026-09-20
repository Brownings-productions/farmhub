"""OpenAI-compatible LLM client (SPEC §8.2).

The only implementation of ``LLMBackend``. It speaks the OpenAI chat-completions
protocol and nothing else, so vLLM, Ollama and a test double are interchangeable by
config alone (SPEC §2: "switching to Ollama must be a config change, never a code
change"). Nothing here knows what is on the other end.

vLLM is deliberately not always running on the dev PC (docs/RUNBOOK.md), so a failure
to reach the backend is an ordinary, expected condition: ``probe`` reports it and the
module turns it into a degraded start rather than a crash.
"""

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import structlog
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    Timeout,
    omit,
)
from openai import APIError as OpenAIAPIError

from farmhub.core.config import LlmSettings
from farmhub.core.errors import DependencyUnavailable, LLMError
from farmhub.core.protocols import (
    ChatMessage,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolSpec,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam


def _as_params(messages: Sequence[ChatMessage]) -> list["ChatCompletionMessageParam"]:
    """Our message type as the SDK's.

    ``ChatMessage.role`` is already constrained to the three roles we send, so this is
    a shape cast, not a claim about untyped data.
    """
    return cast(
        "list[ChatCompletionMessageParam]",
        [{"role": m.role, "content": m.content} for m in messages],
    )


def tool_schema(spec: ToolSpec) -> dict[str, Any]:
    """Render a ``ToolSpec`` as an OpenAI function-tool definition.

    ``spec.parameters`` is already a strict schema — ``ToolSpec`` refuses to construct
    otherwise (§6, §7) — so this only wraps it. ``strict`` asks the backend to enforce
    the schema during decoding, which is what makes enums and numeric bounds in the
    tool TOML worth declaring (§3.1).
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
            "strict": True,
        },
    }


class OpenAICompatBackend:
    """An ``LLMBackend`` over any OpenAI-compatible server."""

    def __init__(
        self,
        settings: LlmSettings,
        log: structlog.typing.FilteringBoundLogger,
        *,
        http_client: Any = None,
    ) -> None:
        """``http_client`` is a seam for tests only.

        It is typed loosely on purpose: the SDK's HTTP layer is its own business, and
        naming that type here would pin us to a transitive dependency that SPEC §12
        does not list.
        """
        self._settings = settings
        self._log = log.bind(component="llm")
        # Constructing the client opens no connection, so this is safe in build_app
        # before anything has started. The module owns the lifecycle.
        self._client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            # A short connect timeout matters more than the request timeout: when vLLM
            # is simply not running, the health probe should fail in seconds so the
            # module degrades promptly, not after a minute.
            timeout=Timeout(settings.request_timeout_s, connect=settings.connect_timeout_s),
            # Retry with backoff is the SDK's, so it covers connect errors and 5xx
            # without a hand-rolled loop (§8.2).
            max_retries=settings.max_retries,
            http_client=http_client,
        )

    @property
    def model(self) -> str:
        return self._settings.model

    async def aclose(self) -> None:
        await self._client.close()

    async def probe(self) -> str:
        """Check the backend is serving, returning the model id it reports.

        Raises ``DependencyUnavailable`` when it cannot be reached, which the module
        turns into a degraded start (SPEC §6 failure policy). A backend that answers
        but does not serve the configured model is a config error in disguise, so it
        is reported as unavailable with an explicit message rather than silently
        serving something else.
        """
        try:
            listed = await self._client.models.list()
        except (APIConnectionError, APITimeoutError) as exc:
            raise DependencyUnavailable(
                f"LLM backend at {self._settings.base_url} is unreachable: {exc}"
            ) from exc
        except APIStatusError as exc:
            raise DependencyUnavailable(
                f"LLM backend at {self._settings.base_url} returned {exc.status_code}"
            ) from exc

        served = [m.id for m in listed.data]
        if self._settings.model not in served:
            raise DependencyUnavailable(
                f"LLM backend serves {served or ['nothing']}, not the configured model "
                f"{self._settings.model!r}"
            )
        return self._settings.model

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One completion. ``tools`` is built server-side, never client-supplied (§3.6)."""
        schemas = cast("list[ChatCompletionToolParam]", [tool_schema(t) for t in tools])
        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=_as_params(messages),
                tools=schemas if tools else omit,
                temperature=temperature if temperature is not None else omit,
                max_completion_tokens=max_tokens if max_tokens is not None else omit,
            )
        except (APIConnectionError, APITimeoutError) as exc:
            raise DependencyUnavailable(f"LLM backend unreachable: {exc}") from exc
        except OpenAIAPIError as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc

        if not completion.choices:
            raise LLMError("LLM returned no choices")
        choice = completion.choices[0]

        parsed_calls = tuple(_parse_tool_call(call) for call in (choice.message.tool_calls or []))
        usage = _parse_usage(completion.usage)
        self._log.info(
            "llm_completion",
            model=completion.model,
            finish_reason=choice.finish_reason,
            tool_calls=len(parsed_calls),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
        return LLMResponse(
            content=choice.message.content,
            tool_calls=parsed_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
            model=completion.model,
        )

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas as they arrive. No tools: see the protocol docstring."""
        try:
            stream = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=_as_params(messages),
                stream=True,
                temperature=temperature if temperature is not None else omit,
                max_completion_tokens=max_tokens if max_tokens is not None else omit,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (APIConnectionError, APITimeoutError) as exc:
            raise DependencyUnavailable(f"LLM backend unreachable: {exc}") from exc
        except OpenAIAPIError as exc:
            raise LLMError(f"LLM stream failed: {exc}") from exc

    async def structured(
        self,
        messages: Sequence[ChatMessage],
        schema: Mapping[str, Any],
        *,
        schema_name: str = "response",
    ) -> Mapping[str, Any]:
        """A completion constrained to ``schema``, for the §4 classifier.

        Issued with no tools, by construction: this is the call that decides which
        pipeline runs, and it must never be able to act.
        """
        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=_as_params(messages),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": dict(schema),
                        "strict": True,
                    },
                },
            )
        except (APIConnectionError, APITimeoutError) as exc:
            raise DependencyUnavailable(f"LLM backend unreachable: {exc}") from exc
        except OpenAIAPIError as exc:
            raise LLMError(f"structured call failed: {exc}") from exc

        if not completion.choices:
            raise LLMError("structured call returned no choices")
        content = completion.choices[0].message.content
        if not content:
            raise LLMError("structured call returned empty content")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            # The caller applies the safe default (§4). Log the text so a backend that
            # ignores response_format is diagnosable.
            self._log.warning("structured_output_not_json", error=str(exc), content=content[:500])
            raise LLMError(f"structured output was not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"structured output was {type(parsed).__name__}, not an object")
        return parsed


def _parse_tool_call(call: Any) -> ToolCall:
    """Turn one OpenAI tool call into a ``ToolCall``.

    The model's arguments are untrusted text: malformed JSON becomes an empty argument
    mapping rather than an exception, so the gateway can deny and audit the call by
    name (§3.7) instead of the turn dying before anything is recorded.
    """
    raw = getattr(call.function, "arguments", "") or "{}"
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError:
        arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(name=call.function.name, arguments=arguments)


def _parse_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )
