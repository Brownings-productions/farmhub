"""Config flow: the FarmHub URL and the shared bearer token.

The token is stored in the config entry, which is what Home Assistant encrypts at
rest. It is never written into YAML or logged (SPEC §3.6, and §12's "no secret in
git" in spirit).

Setup verifies the URL answers before saving, so a typo is caught here rather than at
06:00 when someone asks the barn a question.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import CONF_TOKEN, CONF_URL, DEFAULT_URL, DOMAIN, HEALTH_PATH

SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL, default=DEFAULT_URL): str,
        vol.Required(CONF_TOKEN): str,
    }
)


class FarmHubConfigFlow(ConfigFlow, domain=DOMAIN):
    """Ask for the URL and token, then check they work."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            url = str(user_input[CONF_URL]).rstrip("/")
            # Liveness needs no token, so this separates "the service is not there"
            # from "the token is wrong" instead of reporting one error for both.
            session = async_get_clientsession(self.hass)
            try:
                async with session.get(url + HEALTH_PATH, timeout=10) as response:
                    response.raise_for_status()
            except Exception:  # noqa: BLE001 - shown to the user as a form error
                errors["base"] = "cannot_connect"
            else:
                await self.async_set_unique_id(url)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="FarmHub", data={CONF_URL: url, CONF_TOKEN: user_input[CONF_TOKEN]}
                )

        return self.async_show_form(step_id="user", data_schema=SCHEMA, errors=errors)
