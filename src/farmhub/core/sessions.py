"""Server-minted sessions (SPEC §3.6).

A session is minted by FarmHub and bound to the authenticated satellite identity.
Home Assistant's ``conversation_id`` is a lookup key only, never trusted alone: it
arrives in a request header and anything holding the bearer token could send any
value, so it is used to *find* a session and never to *claim* one.

The binding is what makes that safe. A session found by conversation id is returned
only if it was minted for the same identity; otherwise a fresh one is minted. A caller
cannot inherit another satellite's session by guessing its conversation id.

In memory at M1. A restart drops conversation continuity, which is acceptable: the
next utterance simply starts a new session. PendingActions, which must survive a
restart, live in Postgres instead (§3.3, M8).
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class Session:
    """One conversation, bound to the identity it was minted for."""

    id: str
    # None for an unidentified request, which is served T0-only (§3.6). Such sessions
    # are still minted so the audit trail has something to group calls by.
    satellite: str | None
    created_at: datetime
    last_seen_at: datetime

    def is_expired(self, now: datetime, ttl: timedelta) -> bool:
        return now - self.last_seen_at > ttl


class SessionStore:
    """Mints and looks up sessions. Never trusts a client-supplied id on its own."""

    def __init__(self, *, ttl_s: float = 1800.0, max_sessions: int = 1000) -> None:
        self._ttl = timedelta(seconds=ttl_s)
        self._max = max_sessions
        self._sessions: dict[str, Session] = {}
        # (satellite, conversation_id) -> session id. The satellite is part of the key,
        # so a conversation id alone can never reach another satellite's session.
        self._by_conversation: dict[tuple[str | None, str], str] = {}

    def resolve(
        self,
        *,
        satellite: str | None,
        conversation_id: str | None,
        now: datetime | None = None,
    ) -> Session:
        """The session for this identity and conversation, minting one if needed."""
        moment = now if now is not None else datetime.now(UTC)
        self._expire(moment)

        if conversation_id is not None:
            existing_id = self._by_conversation.get((satellite, conversation_id))
            existing = self._sessions.get(existing_id) if existing_id else None
            # The identity check is belt and braces: the key already includes the
            # satellite, so a mismatch here would mean the store itself is corrupt.
            if existing is not None and existing.satellite == satellite:
                refreshed = Session(
                    id=existing.id,
                    satellite=existing.satellite,
                    created_at=existing.created_at,
                    last_seen_at=moment,
                )
                self._sessions[refreshed.id] = refreshed
                return refreshed

            # Home Assistant hands FarmHub's own session id back as the conversation
            # id on the next turn (see custom_components/farmhub), so a session id is
            # a legitimate lookup key too. Same binding rule: it resolves only for the
            # identity it was minted for, so it cannot be used to claim another one.
            by_session = self._sessions.get(conversation_id)
            if by_session is not None and by_session.satellite == satellite:
                refreshed = Session(
                    id=by_session.id,
                    satellite=by_session.satellite,
                    created_at=by_session.created_at,
                    last_seen_at=moment,
                )
                self._sessions[refreshed.id] = refreshed
                return refreshed

        minted = Session(
            id=f"s-{uuid.uuid4().hex}",
            satellite=satellite,
            created_at=moment,
            last_seen_at=moment,
        )
        self._sessions[minted.id] = minted
        if conversation_id is not None:
            self._by_conversation[(satellite, conversation_id)] = minted.id
        self._evict_oldest()
        return minted

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def __len__(self) -> int:
        return len(self._sessions)

    def _expire(self, now: datetime) -> None:
        stale = [s.id for s in self._sessions.values() if s.is_expired(now, self._ttl)]
        for session_id in stale:
            self._forget(session_id)

    def _evict_oldest(self) -> None:
        """Bound the store so a flood of unidentified requests cannot exhaust memory."""
        while len(self._sessions) > self._max:
            oldest = min(self._sessions.values(), key=lambda s: s.last_seen_at)
            self._forget(oldest.id)

    def _forget(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        for key, value in list(self._by_conversation.items()):
            if value == session_id:
                del self._by_conversation[key]
