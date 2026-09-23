"""Constants for the FarmHub conversation agent."""

DOMAIN = "farmhub"

CONF_URL = "url"
CONF_TOKEN = "token"  # noqa: S105 - a config-entry key name, not a credential

# FarmHub reads identity from headers, not from the request body: the OpenAI
# chat-completions shape has no field for a Home Assistant device_id, and SPEC §3.6
# forbids taking identity from message content.
HEADER_DEVICE_ID = "X-FarmHub-Device-Id"
HEADER_CONVERSATION_ID = "X-FarmHub-Conversation-Id"

# Where FarmHub is, from inside Home Assistant. "homeassistant.local" would be the HA
# box pointing at itself; the dev container reaches the WSL host through the gateway
# alias in deploy/dev/ha-compose.yaml. On the real ha box this becomes hub's LAN
# address, which is typed into the config flow (docs/RUNBOOK.md).
DEFAULT_URL = "http://host.docker.internal:8099"
CHAT_PATH = "/v1/chat/completions"
HEALTH_PATH = "/healthz"

# A voice turn that takes longer than this is no use to anyone standing in a barn.
REQUEST_TIMEOUT_S = 60
