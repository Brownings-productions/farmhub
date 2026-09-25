"""The llm module and its OpenAI-compatible client (SPEC §8.2).

Everything here runs in process against a mock HTTP transport: no network, no backend,
no GPU (SPEC §9). The transport belongs to the HTTP library the openai SDK happens to
use, which is why it is reached for here and never in ``src/``.
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
import structlog
from support import make_app_context

from farmhub.core.config import LlmSettings, Settings
from farmhub.core.errors import DependencyUnavailable, LLMError
from farmhub.core.protocols import ChatMessage, HealthState, Tier, ToolSpec
from farmhub.core.registry import ModuleRegistry, ModuleStatus
from farmhub.modules.llm import LlmModule, OpenAICompatBackend
from farmhub.modules.llm.client import tool_schema

MODEL = "farmhub-primary"


def make_backend(
    handler: Callable[[httpx2.Request], httpx2.Response],
    *,
    settings: LlmSettings | None = None,
) -> OpenAICompatBackend:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return OpenAICompatBackend(
        settings if settings is not None else LlmSettings(model=MODEL, max_retries=0),
        structlog.get_logger("test"),
        http_client=client,
    )


def completion_body(
    content: str | None = "hei",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }


def models_body(*ids: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": i, "object": "model", "created": 0, "owned_by": "x"} for i in ids],
    }


# --- probe: the degraded-start signal -------------------------------------------------


async def test_probe_returns_the_served_model() -> None:
    backend = make_backend(lambda r: httpx2.Response(200, json=models_body(MODEL)))
    assert await backend.probe() == MODEL


async def test_an_unreachable_backend_is_a_dependency_failure_not_a_crash() -> None:
    """vLLM is stopped by hand on the dev PC, so this is a normal condition."""

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    backend = make_backend(refuse)
    with pytest.raises(DependencyUnavailable, match="unreachable"):
        await backend.probe()


async def test_a_backend_serving_a_different_model_is_reported_not_used() -> None:
    """Silently answering from the wrong model would invalidate the whole evaluation."""
    backend = make_backend(lambda r: httpx2.Response(200, json=models_body("some-other-model")))
    with pytest.raises(DependencyUnavailable, match="some-other-model"):
        await backend.probe()


async def test_a_5xx_from_the_backend_is_a_dependency_failure() -> None:
    backend = make_backend(lambda r: httpx2.Response(503, json={"error": "loading"}))
    with pytest.raises(DependencyUnavailable, match="503"):
        await backend.probe()


# --- chat -----------------------------------------------------------------------------


async def test_chat_returns_content_and_token_accounting() -> None:
    backend = make_backend(lambda r: httpx2.Response(200, json=completion_body("god morgen")))
    response = await backend.chat([ChatMessage("user", "hei")])
    assert response.content == "god morgen"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 11
    assert response.usage.total_tokens == 14


async def test_chat_sends_the_configured_model_and_the_messages_given() -> None:
    seen: dict[str, Any] = {}

    def capture(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=completion_body())

    backend = make_backend(capture)
    await backend.chat([ChatMessage("system", "rules"), ChatMessage("user", "hei")])
    assert seen["model"] == MODEL
    assert seen["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hei"},
    ]


async def test_no_tools_key_is_sent_when_there_are_no_tools() -> None:
    """An empty list is not the same as absent; some backends reject the empty list."""
    seen: dict[str, Any] = {}

    def capture(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=completion_body())

    backend = make_backend(capture)
    await backend.chat([ChatMessage("user", "hei")])
    assert "tools" not in seen


async def test_tool_calls_are_parsed() -> None:
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"place": "barn"}'},
        }
    ]
    backend = make_backend(
        lambda r: httpx2.Response(
            200, json=completion_body(None, tool_calls=calls, finish_reason="tool_calls")
        )
    )
    response = await backend.chat([ChatMessage("user", "weather?")])
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"place": "barn"}


@pytest.mark.parametrize("arguments", ["not json at all", "[1, 2]", ""])
async def test_unparseable_tool_arguments_become_empty_not_an_exception(arguments: str) -> None:
    """The gateway must still get a named call to deny and audit (SPEC §3.7).

    Model output is untrusted text. If malformed arguments raised here, the turn would
    die before any audit row named the tool that was attempted.
    """
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "start_watering", "arguments": arguments},
        }
    ]
    backend = make_backend(
        lambda r: httpx2.Response(200, json=completion_body(None, tool_calls=calls))
    )
    response = await backend.chat([ChatMessage("user", "water")])
    assert response.tool_calls[0].name == "start_watering"
    assert response.tool_calls[0].arguments == {}


# --- structured output, for the §4 classifier -----------------------------------------


async def test_structured_parses_the_json_object() -> None:
    body = completion_body(json.dumps({"kind": "action"}))
    backend = make_backend(lambda r: httpx2.Response(200, json=body))
    result = await backend.structured(
        [ChatMessage("user", "turn on the light")], {"type": "object"}
    )
    assert result == {"kind": "action"}


async def test_structured_sends_no_tools() -> None:
    """SPEC §4: the classifier is issued with tools=[] and can never act."""
    seen: dict[str, Any] = {}

    def capture(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return httpx2.Response(200, json=completion_body('{"kind": "question"}'))

    backend = make_backend(capture)
    await backend.structured([ChatMessage("user", "hei")], {"type": "object"})
    assert "tools" not in seen
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("content", ["not json", "[1,2,3]", '"a string"'])
async def test_structured_output_that_is_not_an_object_raises_llm_error(content: str) -> None:
    """The caller applies the §4 safe default; it must not receive a plausible lie."""
    backend = make_backend(lambda r: httpx2.Response(200, json=completion_body(content)))
    with pytest.raises(LLMError):
        await backend.structured([ChatMessage("user", "hei")], {"type": "object"})


# --- streaming ------------------------------------------------------------------------


async def test_stream_chat_yields_content_deltas() -> None:
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "god "}}]},
        {"choices": [{"index": 0, "delta": {"content": "morgen"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    backend = make_backend(
        lambda r: httpx2.Response(200, text=payload, headers={"content-type": "text/event-stream"})
    )
    deltas = [d async for d in backend.stream_chat([ChatMessage("user", "hei")])]
    assert "".join(deltas) == "god morgen"


# --- tool schema rendering ------------------------------------------------------------


def test_tool_schema_is_strict_and_carries_the_spec_schema() -> None:
    """Strict decoding is what makes the enums and bounds of §7 worth declaring."""

    async def handler(call: Any, ctx: Any) -> Any:  # pragma: no cover - never invoked
        raise AssertionError

    parameters = {
        "type": "object",
        "properties": {"duration_min": {"type": "integer", "minimum": 1, "maximum": 20}},
        "required": ["duration_min"],
        "additionalProperties": False,
    }
    spec = ToolSpec(
        name="start_greenhouse_watering",
        description="Start watering.",
        parameters=parameters,
        tier=Tier.CONFIRMED,
        handler=handler,
        allowed_scopes=frozenset({"greenhouse"}),
    )
    rendered = tool_schema(spec)
    assert rendered["type"] == "function"
    assert rendered["function"]["name"] == "start_greenhouse_watering"
    assert rendered["function"]["strict"] is True
    assert rendered["function"]["parameters"]["additionalProperties"] is False


# --- the module -----------------------------------------------------------------------


async def test_module_starts_when_the_backend_is_serving() -> None:
    backend = make_backend(lambda r: httpx2.Response(200, json=models_body(MODEL)))
    module = LlmModule(backend)
    await module.startup(make_app_context(settings=Settings(llm=LlmSettings(model=MODEL))))
    report = await module.health()
    assert report.status is HealthState.HEALTHY
    await module.shutdown()


async def test_module_exposes_no_tools() -> None:
    """SPEC §8.2. The LLM is how tools are chosen, never a tool itself."""
    backend = make_backend(lambda r: httpx2.Response(200, json=models_body(MODEL)))
    assert LlmModule(backend).tools() == []


async def test_module_start_raises_dependency_unavailable_so_the_registry_degrades() -> None:
    """This is what lets vLLM be stopped by hand without taking FarmHub down."""

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    module = LlmModule(make_backend(refuse))
    with pytest.raises(DependencyUnavailable):
        await module.startup(make_app_context())


async def test_module_health_reports_a_backend_that_went_away() -> None:
    """Reported, not assumed still there: this is what the health poll acts on."""
    serving = True

    def handler(request: httpx2.Request) -> httpx2.Response:
        if serving:
            return httpx2.Response(200, json=models_body(MODEL))
        raise httpx2.ConnectError("refused", request=request)

    module = LlmModule(make_backend(handler))
    await module.startup(make_app_context())
    serving = False
    report = await module.health()
    assert report.status is HealthState.UNHEALTHY


async def test_module_shutdown_is_idempotent() -> None:
    """The registry calls shutdown after a failed start and before every retry."""
    module = LlmModule(make_backend(lambda r: httpx2.Response(200, json=models_body(MODEL))))
    await module.shutdown()
    await module.shutdown()


async def test_the_shared_client_still_works_after_a_degraded_start() -> None:
    """Regression: a degraded start used to close the shared client for good.

    The registry calls ``shutdown`` after a degraded start and before each retry. When
    the module closed the backend there, every later request failed with "the client
    has been closed", even after vLLM came back. This runs the real registry against
    the real client: vLLM is down at startup, comes back before the first retry, and a
    completion through the same shared backend must then succeed.
    """
    backend_up = False

    def vllm(request: httpx2.Request) -> httpx2.Response:
        if not backend_up:
            raise httpx2.ConnectError("connection refused", request=request)
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json=models_body(MODEL))
        return httpx2.Response(200, json=completion_body("tilbake"))

    async def vllm_comes_back(_delay: float) -> None:
        nonlocal backend_up
        backend_up = True

    backend = make_backend(vllm)
    ctx = make_app_context(llm=backend)
    registry = ModuleRegistry([LlmModule(backend)], sleep=vllm_comes_back)
    recovered = asyncio.Event()

    async def on_recovered(_event: object) -> None:
        recovered.set()

    ctx.events.subscribe("module.recovered", on_recovered)
    await registry.startup(ctx)
    assert registry.status()["llm"] is ModuleStatus.DEGRADED

    await asyncio.wait_for(recovered.wait(), timeout=2)
    assert registry.status()["llm"] is ModuleStatus.RUNNING
    response = await ctx.llm.chat([ChatMessage(role="user", content="hei")])
    assert response.content == "tilbake"

    await registry.shutdown()
    await backend.aclose()


# --- the thinking switch reaches every call (docs/DECISIONS.md 2026-09-23) -------------


def capture_backend(
    response: httpx2.Response,
    seen: dict[str, Any],
    *,
    switches: dict[str, Any] | None = None,
    temperature: float | None = 0.0,
) -> OpenAICompatBackend:
    def capture(request: httpx2.Request) -> httpx2.Response:
        seen.update(json.loads(request.content))
        return response

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(capture))
    return OpenAICompatBackend(
        LlmSettings(model=MODEL, max_retries=0),
        structlog.get_logger("test"),
        chat_template_kwargs=switches if switches is not None else {"enable_thinking": False},
        temperature=temperature,
        http_client=client,
    )


async def test_chat_sends_the_profiles_chat_template_kwargs() -> None:
    """With thinking left on, the model reasons instead of answering (MODEL_EVAL.md)."""
    seen: dict[str, Any] = {}
    backend = capture_backend(httpx2.Response(200, json=completion_body("8,5 liter")), seen)
    await backend.chat([ChatMessage("user", "hei")])
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


async def test_structured_sends_the_chat_template_kwargs_too() -> None:
    """The §4 classifier is the call least able to afford a preamble."""
    seen: dict[str, Any] = {}
    backend = capture_backend(
        httpx2.Response(200, json=completion_body('{"kind": "question"}')), seen
    )
    await backend.structured([ChatMessage("user", "hei")], {"type": "object"})
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


async def test_stream_chat_sends_the_chat_template_kwargs_too() -> None:
    seen: dict[str, Any] = {}
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "ja"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    backend = capture_backend(
        httpx2.Response(200, text=payload, headers={"content-type": "text/event-stream"}), seen
    )
    assert [d async for d in backend.stream_chat([ChatMessage("user", "hei")])] == ["ja"]
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


async def test_a_backend_with_no_profile_sends_no_extra_body() -> None:
    """Ollama and the test double have no measured profile, and need no switches."""
    seen: dict[str, Any] = {}
    backend = capture_backend(httpx2.Response(200, json=completion_body("hei")), seen, switches={})
    await backend.chat([ChatMessage("user", "hei")])
    assert "chat_template_kwargs" not in seen


# --- the profile's temperature reaches every call (docs/DECISIONS.md 2026-09-25) -------


async def test_chat_sends_the_profiles_temperature() -> None:
    """Without it the backend's default applies, and a default nobody declared is how
    two eval runs produced tool scores that could not be compared (MODEL_EVAL.md)."""
    seen: dict[str, Any] = {}
    backend = capture_backend(
        httpx2.Response(200, json=completion_body("8,5 liter")), seen, temperature=0.0
    )
    await backend.chat([ChatMessage("user", "hei")])
    assert seen["temperature"] == 0.0


async def test_structured_sends_the_profiles_temperature() -> None:
    """§4 routes an unparseable answer to `question`, so a sampled classifier is a
    pipeline that changes its mind between identical utterances."""
    seen: dict[str, Any] = {}
    backend = capture_backend(
        httpx2.Response(200, json=completion_body('{"kind": "question"}')), seen, temperature=0.0
    )
    await backend.structured([ChatMessage("user", "hei")], {"type": "object"})
    assert seen["temperature"] == 0.0


async def test_stream_chat_sends_the_profiles_temperature() -> None:
    seen: dict[str, Any] = {}
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "ja"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    backend = capture_backend(
        httpx2.Response(200, text=payload, headers={"content-type": "text/event-stream"}),
        seen,
        temperature=0.0,
    )
    assert [d async for d in backend.stream_chat([ChatMessage("user", "hei")])] == ["ja"]
    assert seen["temperature"] == 0.0


async def test_an_explicit_temperature_beats_the_profiles() -> None:
    """A caller that needs a different setting is not silently overridden."""
    seen: dict[str, Any] = {}
    backend = capture_backend(
        httpx2.Response(200, json=completion_body("hei")), seen, temperature=0.0
    )
    await backend.chat([ChatMessage("user", "hei")], temperature=0.7)
    assert seen["temperature"] == 0.7


async def test_a_backend_with_no_profile_sends_no_temperature() -> None:
    """Ollama and the test double keep whatever default they have."""
    seen: dict[str, Any] = {}
    backend = capture_backend(
        httpx2.Response(200, json=completion_body("hei")), seen, temperature=None
    )
    await backend.chat([ChatMessage("user", "hei")])
    assert "temperature" not in seen
