# FarmHub Home Assistant integration

A conversation agent that forwards unmatched utterances to the local FarmHub service.

It exists because SPEC §14 Q1 has no good answer with the built-in integrations: a
standard OpenAI chat-completions request has no field for the Home Assistant
`device_id`, and SPEC §3.6 forbids taking satellite identity from message content. The
built-in "OpenAI Conversation" integration also builds its own prompt and tool list,
which collides with FarmHub building its own.

## What it sends

`POST <url>/v1/chat/completions` with the utterance as the single user message, and
three headers:

| Header | Meaning |
|---|---|
| `Authorization: Bearer …` | Authenticates Home Assistant to FarmHub. The only thing FarmHub verifies. |
| `X-FarmHub-Device-Id` | Which satellite spoke. HA vouches for it; FarmHub does not verify it independently. |
| `X-FarmHub-Conversation-Id` | A lookup key so a conversation continues. Never a credential. |

No tools and no system prompt. FarmHub ignores both (§3.6), and sending them would
suggest they mattered.

An absent `device_id` is left absent rather than filled in with a guess: FarmHub then
serves the request T0-only and logs it, which is the §3.6 fail-closed behaviour.

## Install

Copy this directory to `<config>/custom_components/farmhub` and restart Home
Assistant, then add the integration from Settings → Devices & Services. For
development, `deploy/dev/ha-compose.yaml` mounts it into a throwaway container.

Set the conversation agent under Settings → Voice assistants.

## Note

This directory is not part of the `farmhub` Python package. It runs inside Home
Assistant's own interpreter against Home Assistant's APIs, so it is excluded from
`mypy` and is not importable from the application. The Home Assistant conversation
API changes between releases: `deploy/dev/ha-compose.yaml` pins the version this was
written against.
