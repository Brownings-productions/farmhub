"""Audit sinks.

M0 ships the types only (see protocols.py). The one implementation here exists so an
AppContext can always be built without silently discarding audit rows: until a real sink
lands (JSONL at M5/M6, Postgres at M2), every write is refused and the gateway therefore
denies every call. Failing closed is deliberate (§3.7).
"""

from farmhub.core.errors import AuditWriteError
from farmhub.core.protocols import AuditEntry


class UnconfiguredAuditSink:
    """Refuses every write, so no tool can run before a real sink is configured."""

    async def write(self, entry: AuditEntry) -> None:
        raise AuditWriteError("no audit sink is configured; refusing to proceed without audit")
