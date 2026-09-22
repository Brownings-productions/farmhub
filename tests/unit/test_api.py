"""The chat and health endpoints (SPEC §3.6, §4, §11 M1)."""

from pathlib import Path

import pytest
from structlog.testing import capture_logs
from support import StubLLM
from support_api import auth, make_client, make_settings

from farmhub.api.openai_compat import SYSTEM_PROMPT, ChatCompletionRequest, latest_user_text
from farmhub.api.server import create_app
from farmhub.core.config import Settings
from farmhub.core.errors import ConfigError, DependencyUnavailable, LLMError


def body(text: str = "hvor mye olje tar traktoren?") -> dict[str, object]:
    return {"model": "ignored", "messages": [{"role": "user", "content": text}]}


# --- the server refuses to start without a token (SPEC §3.6, §7) ----------------------


def test_the_server_will_not_start_without_an_api_token() -> None:
    """The token is what authenticates HA; there is no safe default for it."""
    with pytest.raises(ConfigError, match="no API token"):
        create_app(Settings(), configure_logs=False)


# --- answering ------------------------------------------------------------------------


def test_a_valid_request_is_answered(tmp_path: Path) -> None:
    client, _llm = make_client(make_settings(tmp_path), StubLLM("omtrent 8 liter"))
    with client:
        response = client.post(
            "/v1/chat/completions",
            json=body(),
            headers=auth(**{"X-FarmHub-Device-Id": "dev-kitchen"}),
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "omtrent 8 liter"
    assert payload["choices"][0]["message"]["role"] == "assistant"


def test_the_server_builds_its_own_system_prompt(tmp_path: Path) -> None:
    client, llm = make_client(make_settings(tmp_path))
    with client:
        client.post("/v1/chat/completions", json=body("hei"), headers=auth())
    sent, _tools = llm.chats[0]
    assert sent[0].role == "system"
    assert sent[0].content == SYSTEM_PROMPT


def test_a_request_with_no_user_text_is_rejected(tmp_path: Path) -> None:
    client, _ = make_client(make_settings(tmp_path))
    with client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "assistant", "content": "hei"}]},
            headers=auth(),
        )
    assert response.status_code == 400


# --- the bearer token (SPEC §3.6) -----------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic x"}, {"Authorization": ""}],
    ids=["absent", "wrong", "wrong-scheme", "empty"],
)
def test_the_chat_endpoint_requires_the_bearer_token(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    client, llm = make_client(make_settings(tmp_path))
    with client:
        response = client.post("/v1/chat/completions", json=body(), headers=headers)
    assert response.status_code == 401
    # Nothing reached the model: rejection happens before any work is done.
    assert llm.chats == []


def test_the_health_endpoint_requires_the_bearer_token(tmp_path: Path) -> None:
    """Module health names which dependencies are down; not for anonymous callers."""
    client, _ = make_client(make_settings(tmp_path))
    with client:
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers=auth()).status_code == 200


def test_liveness_needs_no_token(tmp_path: Path) -> None:
    """A liveness probe must work before anything else, and reveals nothing."""
    client, _ = make_client(make_settings(tmp_path))
    with client:
        response = client.get("/healthz")
    assert response.status_code == 200


# --- identity and sessions (SPEC §3.6) ------------------------------------------------


def test_a_known_device_id_is_resolved_to_its_satellite(tmp_path: Path) -> None:
    client, _ = make_client(make_settings(tmp_path))
    with client:
        response = client.post(
            "/v1/chat/completions",
            json=body(),
            headers=auth(**{"X-FarmHub-Device-Id": "dev-workshop"}),
        )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "device_headers",
    [{}, {"X-FarmHub-Device-Id": "not-registered"}],
    ids=["no-identity", "unknown-identity"],
)
def test_an_unidentified_request_is_still_served(
    tmp_path: Path, device_headers: dict[str, str]
) -> None:
    """SPEC §3.6 fail closed: T0 only, not refused.

    Refusing would make a mis-registered satellite mute rather than merely limited.
    """
    client, _ = make_client(make_settings(tmp_path))
    with client:
        response = client.post("/v1/chat/completions", json=body(), headers=auth(**device_headers))
    assert response.status_code == 200


def test_the_session_is_minted_server_side_and_returned(tmp_path: Path) -> None:
    client, _ = make_client(make_settings(tmp_path))
    with client:
        response = client.post(
            "/v1/chat/completions",
            json=body(),
            headers=auth(**{"X-FarmHub-Device-Id": "dev-kitchen"}),
        )
    session = response.json()["farmhub"]["session"]
    assert session.startswith("s-")


def test_the_same_conversation_id_continues_one_session(tmp_path: Path) -> None:
    client, _ = make_client(make_settings(tmp_path))
    headers = auth(**{"X-FarmHub-Device-Id": "dev-kitchen", "X-FarmHub-Conversation-Id": "ha-1"})
    with client:
        first = client.post("/v1/chat/completions", json=body(), headers=headers)
        second = client.post("/v1/chat/completions", json=body(), headers=headers)
    assert first.json()["farmhub"]["session"] == second.json()["farmhub"]["session"]


def test_another_satellite_cannot_claim_a_session_by_conversation_id(tmp_path: Path) -> None:
    """The conversation id is a lookup key, never a credential (SPEC §3.6)."""
    client, _ = make_client(make_settings(tmp_path))
    with client:
        kitchen = client.post(
            "/v1/chat/completions",
            json=body(),
            headers=auth(
                **{"X-FarmHub-Device-Id": "dev-kitchen", "X-FarmHub-Conversation-Id": "ha-1"}
            ),
        )
        workshop = client.post(
            "/v1/chat/completions",
            json=body(),
            headers=auth(
                **{"X-FarmHub-Device-Id": "dev-workshop", "X-FarmHub-Conversation-Id": "ha-1"}
            ),
        )
    assert kitchen.json()["farmhub"]["session"] != workshop.json()["farmhub"]["session"]


# --- the backend being down (docs/RUNBOOK.md) -----------------------------------------


def test_a_stopped_backend_gives_503_not_a_crash(tmp_path: Path) -> None:
    """vLLM is stopped by hand on the dev PC; the server stays up and says why."""

    class Down(StubLLM):
        async def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            raise DependencyUnavailable("vLLM is not running")

    client, _ = make_client(make_settings(tmp_path), Down())
    with client:
        response = client.post("/v1/chat/completions", json=body(), headers=auth())
    assert response.status_code == 503


def test_an_unusable_response_gives_502(tmp_path: Path) -> None:
    class Broken(StubLLM):
        async def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            raise LLMError("garbage")

    client, _ = make_client(make_settings(tmp_path), Broken())
    with client:
        response = client.post("/v1/chat/completions", json=body(), headers=auth())
    assert response.status_code == 502


# --- picking the utterance ------------------------------------------------------------


def test_latest_user_text_takes_only_the_current_utterance() -> None:
    """SPEC §3.5 rule 4: an action turn never receives replayed history."""
    request = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ]
    )
    assert latest_user_text(request.messages) == "second"


def test_latest_user_text_handles_structured_content() -> None:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hei"}]}]
    )
    assert latest_user_text(request.messages) == "hei"


def test_latest_user_text_is_none_when_there_is_nothing_to_answer() -> None:
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "   "}])
    assert latest_user_text(request.messages) is None


def test_an_unexpected_backend_error_gives_503_and_leaks_nothing(tmp_path: Path) -> None:
    """Regression: an unclassified exception used to reach HA as an opaque 500.

    It must become a 503 with a fixed message, while the traceback goes to the log
    where it can be fixed. Nothing from the exception may appear in the response body:
    internals such as URLs or paths are for the operator, not for a satellite.
    """
    internals = "internal detail /srv/vllm http://10.0.0.5:8000"

    class Exploding(StubLLM):
        async def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError(internals)

    client, _ = make_client(make_settings(tmp_path), Exploding())
    with client, capture_logs() as logs:
        response = client.post("/v1/chat/completions", json=body(), headers=auth())
    assert response.status_code == 503
    assert internals not in response.text
    assert "RuntimeError" not in response.text
    raised = [e for e in logs if e["event"] == "llm_call_raised"]
    assert len(raised) == 1
    assert raised[0]["log_level"] == "error"
    assert raised[0]["exc_info"] is True
