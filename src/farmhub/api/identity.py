"""Authentication and satellite identity (SPEC §3.6).

Two separate things, deliberately kept apart:

**Authentication** is the shared service-to-service bearer token. It says the caller is
Home Assistant. It is the only thing actually verified here.

**Identity** is the ``device_id`` header, which says *which satellite* Home Assistant is
relaying for. FarmHub does not verify it independently — HA vouches for it. So anyone
holding the token can assert any device_id, and the token is the real security
boundary (docs/DECISIONS.md). The registry then bounds what that identity can reach,
and §3.6 requires the listener to be firewalled to the ha host.

Identity is never taken from message content. The request body cannot influence any of
this, which is why it is resolved from headers before the body is looked at.
"""

import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog
from fastapi import Header, HTTPException, Request, status

from farmhub.core.protocols import CallOrigin, Tier
from farmhub.core.satellites import Satellite, SatelliteRegistry
from farmhub.core.sessions import SessionStore

# Headers rather than the OpenAI `user` field: FarmHub owns both ends of this call, and
# overloading a field that means something else would be a trap for the next reader.
DEVICE_ID_HEADER = "X-FarmHub-Device-Id"
CONVERSATION_ID_HEADER = "X-FarmHub-Conversation-Id"


@dataclass(frozen=True)
class ResolvedIdentity:
    """Who a request is for, and what that entitles it to."""

    origin: CallOrigin
    satellite: Satellite | None
    tier_ceiling: Tier
    # True when no usable identity was found, so the request is T0-only (§3.6).
    fell_closed: bool


def require_bearer_token(expected: str) -> Callable[[str], Awaitable[None]]:
    """Build the FastAPI dependency that checks the bearer token.

    Compared with ``compare_digest`` so a wrong token cannot be found a character at a
    time by timing the response.
    """

    async def check(authorization: str = Header(default="")) -> None:
        scheme, _, presented = authorization.partition(" ")
        if scheme.lower() != "bearer" or not presented:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="a bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Compared as bytes: compare_digest raises TypeError on a non-ASCII str, and
        # a token that is merely malformed must be a 401, not a 500.
        if not secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return check


def resolve_identity(
    request: Request,
    registry: SatelliteRegistry,
    sessions: SessionStore,
    log: structlog.typing.FilteringBoundLogger,
) -> ResolvedIdentity:
    """Turn request headers into a ``CallOrigin``, failing closed (§3.6).

    A missing or unregistered ``device_id`` is not an error: the request is served
    with T0 tools only and the condition is logged as a warning. Refusing outright
    would make a mis-registered satellite mute rather than merely limited, and §3.6
    asks for the limited behaviour.
    """
    device_id = request.headers.get(DEVICE_ID_HEADER)
    conversation_id = request.headers.get(CONVERSATION_ID_HEADER)
    satellite = registry.get(device_id)

    if satellite is None:
        # Sessions are still minted for unidentified callers, so the audit trail has
        # something to group their calls by.
        session = sessions.resolve(satellite=None, conversation_id=conversation_id)
        log.warning(
            "unidentified_request_served_t0_only",
            device_id=device_id or "<absent>",
            reason="no device id" if device_id is None else "device id not in registry",
            session=session.id,
        )
        return ResolvedIdentity(
            origin=CallOrigin(satellite=None, session=session.id, scope=frozenset()),
            satellite=None,
            tier_ceiling=Tier.READ,
            fell_closed=True,
        )

    session = sessions.resolve(satellite=satellite.name, conversation_id=conversation_id)
    return ResolvedIdentity(
        origin=CallOrigin(
            satellite=satellite.name,
            session=session.id,
            scope=satellite.scope,
        ),
        satellite=satellite,
        tier_ceiling=satellite.tier_ceiling,
        fell_closed=False,
    )
