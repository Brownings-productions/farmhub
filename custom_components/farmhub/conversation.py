"""The conversation agent (SPEC §4, §14 Q1).

Home Assistant handles every intent it recognises itself. This agent is the fallback
for everything else, so what arrives here is an open question or an unmatched command.

The agent sends the utterance and three pieces of context, all as headers:

* the bearer token, which authenticates Home Assistant to FarmHub (§3.6);
* ``device_id``, which says which satellite spoke. HA vouches for it; FarmHub does not
  verify it independently, so the token is the real security boundary;
* ``conversation_id``, which is a lookup key only. FarmHub mints the session and
  returns its own id, which is what we hand back to HA.

It sends no tools and no system prompt. FarmHub ignores both anyway (§3.6), but
sending them would suggest they mattered.
"""

from __future__ import annotations

from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CHAT_PATH,
    CONF_TOKEN,
    CONF_URL,
    DOMAIN,
    HEADER_CONVERSATION_ID,
    HEADER_DEVICE_ID,
    REQUEST_TIMEOUT_S,
)

# What the user hears when something is wrong. Each says what to do next, because the
# person hearing it is usually holding a tool and cannot go and read a log.
ERR_UNAVAILABLE = "FarmHub cannot reach the language model right now."
ERR_UNAUTHORISED = "FarmHub rejected the token. Check the FarmHub integration settings."
ERR_UNREACHABLE = "FarmHub is not responding. Check that the FarmHub service is running."
ERR_BAD_REPLY = "FarmHub returned something I could not read."


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the conversation agent."""
    async_add_entities([FarmHubConversationEntity(entry)])


class FarmHubConversationEntity(conversation.ConversationEntity):
    """Relays an utterance to FarmHub and speaks the answer."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Whatever the model speaks. FarmHub answers in the language it was asked in."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Forward one turn and return the answer."""
        url = str(self._entry.data[CONF_URL]).rstrip("/") + CHAT_PATH
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._entry.data[CONF_TOKEN]}",
            "Content-Type": "application/json",
        }
        # Only sent when HA actually knows them. An absent device id means FarmHub
        # serves the request T0-only and logs it (§3.6 fail closed) — which is the
        # behaviour we want, so nothing is invented here to fill the gap.
        if user_input.device_id:
            headers[HEADER_DEVICE_ID] = user_input.device_id
        if user_input.conversation_id:
            headers[HEADER_CONVERSATION_ID] = user_input.conversation_id

        payload = {"messages": [{"role": "user", "content": user_input.text}]}

        session = async_get_clientsession(self.hass)
        try:
            async with session.post(
                url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S
            ) as response:
                if response.status in (401, 403):
                    return self._error(user_input, ERR_UNAUTHORISED)
                if response.status >= 500:
                    return self._error(user_input, ERR_UNAVAILABLE)
                response.raise_for_status()
                body: dict[str, Any] = await response.json()
        except TimeoutError:
            return self._error(user_input, ERR_UNAVAILABLE)
        except Exception:  # noqa: BLE001 - the person is waiting; say something useful
            return self._error(user_input, ERR_UNREACHABLE)

        answer = _answer_of(body)
        if answer is None:
            return self._error(user_input, ERR_BAD_REPLY)

        speech = intent.IntentResponse(language=user_input.language)
        speech.async_set_speech(answer)
        return conversation.ConversationResult(
            response=speech,
            # FarmHub mints the session server-side and returns it; handing it back
            # keeps the next turn in the same conversation (§3.6).
            conversation_id=_session_of(body) or user_input.conversation_id,
        )

    def _error(
        self, user_input: conversation.ConversationInput, message: str
    ) -> conversation.ConversationResult:
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        return conversation.ConversationResult(
            response=response, conversation_id=user_input.conversation_id
        )


def _answer_of(body: dict[str, Any]) -> str | None:
    """The assistant text, or None if the reply is not shaped as expected."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) and content.strip() else None


def _session_of(body: dict[str, Any]) -> str | None:
    extra = body.get("farmhub")
    if not isinstance(extra, dict):
        return None
    session = extra.get("session")
    return session if isinstance(session, str) else None


__all__ = ["DOMAIN", "FarmHubConversationEntity", "async_setup_entry"]
