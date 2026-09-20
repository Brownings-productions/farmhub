"""FarmHub: a conversation agent backed by the local FarmHub server.

This integration exists because of SPEC §14 Q1. A standard OpenAI chat-completions
request has no field for the Home Assistant ``device_id``, and §3.6 forbids taking
satellite identity from message content — so the built-in "OpenAI Conversation"
integration cannot be used. It also builds its own prompt and tool list, which
collides with FarmHub building its own.

What this does instead is deliberately thin: forward the utterance, attach the bearer
token and the device and conversation ids as headers, and return what comes back.
Every decision about tools, prompts, tiers and scope is made server-side.
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

PLATFORMS: list[Platform] = [Platform.CONVERSATION]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up FarmHub from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_reload(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)
