"""All Protocol and contract types (SPEC §6).

``Module``, ``Tier``, ``ToolSpec``, ``ToolContext``, ``ToolResult``, ``PendingAction``,
``HealthReport`` and the audit types are fixed by SPEC or by docs/DECISIONS.md.

PROVISIONAL: ``LLMBackend``, ``Retriever``, ``Embedder``, ``Reranker``, ``DocumentParser``,
``Chunker`` and the small data types they use are named but not specified in SPEC §6. The
shapes here are minimal placeholders. Any change is logged in docs/DECISIONS.md at the
milestone that first uses them.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from farmhub.core.errors import ConfigError, SafetyViolation

if TYPE_CHECKING:
    from farmhub.core.context import AppContext


class Tier(IntEnum):
    """Capability tiers (§3.2)."""

    READ = 0
    COMFORT = 1
    CONFIRMED = 2
    FORBIDDEN = 3  # never returned to the model; exists so config can express it


class Decision(StrEnum):
    """Audit decisions (§3.7)."""

    EXECUTED = "executed"
    DENIED = "denied"
    PENDING = "pending"
    EXPIRED = "expired"


class AuditPhase(StrEnum):
    """Every call writes an INTENT row first, then exactly one OUTCOME row."""

    INTENT = "intent"
    OUTCOME = "outcome"


class GatewaySeal:
    """Opaque capability proving a ``ToolContext`` was minted by the gateway.

    Python cannot make this truly private. The protection is layered: handlers reject
    contexts without the seal, and tests/safety/test_gateway_only.py fails if any source
    file other than protocols.py and gateway.py references ``GATEWAY_SEAL`` or reads
    ``.handler``.
    """

    __slots__ = ()


GATEWAY_SEAL: Final = GatewaySeal()  # import only from core/gateway.py


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model. ``name`` and ``arguments`` are untrusted."""

    name: str
    arguments: Mapping[str, Any]
    call_id: uuid.UUID = field(default_factory=uuid.uuid4)


@dataclass(frozen=True)
class ToolResult:
    """What a handler returns. ``untrusted`` lets the pipeline set ``context_tainted``."""

    ok: bool
    data: Mapping[str, Any] = field(default_factory=dict)
    untrusted: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ToolContext:
    """Per-call context handed to a handler.

    Only the gateway can mint a usable one: it carries the gateway's seal, and the
    sealed handler wrapper on ``ToolSpec`` refuses anything else.
    """

    satellite: str | None
    session: str
    scope: frozenset[str]
    dry_run: bool
    context_tainted: bool
    call_id: uuid.UUID
    _seal: object = field(default=None, repr=False, compare=False, kw_only=True)


type ToolHandler = Callable[[ToolCall, ToolContext], Awaitable[ToolResult]]


class _SealedHandler:
    """Wraps a handler so it only runs when invoked by the gateway."""

    def __init__(self, fn: ToolHandler, tool_name: str) -> None:
        self._fn = fn
        self._tool_name = tool_name

    async def __call__(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        if ctx._seal is not GATEWAY_SEAL:
            raise SafetyViolation(
                f"handler for tool {self._tool_name!r} invoked outside the gateway"
            )
        return await self._fn(call, ctx)


@dataclass(frozen=True)
class ToolSpec:
    """One model-callable tool (§6). The handler is sealed at construction."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema, strict, no additionalProperties
    tier: Tier
    handler: ToolHandler
    allowed_scopes: frozenset[str]  # satellite areas, or {"*"}
    timeout_s: float = 10.0
    reads_untrusted_content: bool = False

    def __post_init__(self) -> None:
        if (
            self.parameters.get("type") != "object"
            or self.parameters.get("additionalProperties") is not False
        ):
            raise ConfigError(
                f"tool {self.name!r}: parameters must be an object schema with "
                "additionalProperties: false"
            )
        if not isinstance(self.handler, _SealedHandler):
            object.__setattr__(self, "handler", _SealedHandler(self.handler, self.name))


@dataclass(frozen=True)
class CallOrigin:
    """Who a call is on behalf of. Derived server-side; never from message content (§3.6)."""

    satellite: str | None
    session: str
    scope: frozenset[str]


class PendingState(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass(frozen=True)
class PendingAction:
    """A proposed T2 action awaiting confirmation (§3.3). Shape settles at M8."""

    id: uuid.UUID
    call_id: uuid.UUID  # the proposing call; the resolution's parent_call_id
    satellite: str | None
    session: str
    scope: frozenset[str]
    tool: str
    arguments: Mapping[str, Any]  # exactly what will execute; confirmation cannot alter it
    created_at: datetime
    expires_at: datetime
    state: PendingState = PendingState.PENDING


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class HealthReport:
    status: HealthState
    detail: str = ""


@dataclass(frozen=True)
class AuditEntry:
    """One audit row (§3.7, shaped per docs/DECISIONS.md "Audit semantics").

    A call writes an INTENT row, then exactly one OUTCOME row, linked by ``call_id``.
    A T2 confirmation or expiry is its own call whose ``parent_call_id`` is the
    proposing call. ``tool`` is the name as given (truncated); ``tier`` is None when the
    tool is unknown.
    """

    timestamp: datetime
    call_id: uuid.UUID
    phase: AuditPhase
    tool: str
    tier: Tier | None
    dry_run: bool
    satellite: str | None = None
    session: str | None = None
    arguments: Mapping[str, Any] | None = None
    parent_call_id: uuid.UUID | None = None
    pending_action_id: uuid.UUID | None = None
    decision: Decision | None = None  # outcome rows only
    result: Mapping[str, Any] | None = None
    duration_ms: float | None = None
    confirmation_channel: str | None = None
    confirming_device: str | None = None


class AuditSink(Protocol):
    """Append-only sink. ``write`` raises ``AuditWriteError`` on failure; never swallow."""

    async def write(self, entry: AuditEntry) -> None: ...


@runtime_checkable
class Module(Protocol):
    """A pluggable module (§6).

    ``startup`` raises ``ConfigError`` for invalid config (fatal) or
    ``DependencyUnavailable`` when a runtime dependency is down (degraded start). The
    registry calls ``shutdown`` after a failed or degraded startup before retrying, so
    ``shutdown`` must be idempotent and safe on a module that never fully started.
    """

    name: str
    version: str

    async def startup(self, ctx: AppContext) -> None: ...

    async def shutdown(self) -> None: ...

    def tools(self) -> list[ToolSpec]: ...

    async def health(self) -> HealthReport: ...


# --- Provisional protocols (see module docstring) -------------------------------------


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()


class LLMBackend(Protocol):
    """The only way the application talks to an LLM (§2)."""

    async def chat(
        self, messages: Sequence[ChatMessage], *, tools: Sequence[ToolSpec] = ()
    ) -> LLMResponse: ...

    async def structured(
        self, messages: Sequence[ChatMessage], schema: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_path: str
    page: int | None
    heading_path: str
    score: float


class Retriever(Protocol):
    async def retrieve(
        self, query: str, *, filters: Mapping[str, str] | None = None, limit: int = 5
    ) -> Sequence[RetrievedChunk]: ...


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], *, top_k: int = 5
    ) -> Sequence[RetrievedChunk]: ...


@dataclass(frozen=True)
class ParsedDocument:
    markdown: str
    page_count: int


class DocumentParser(Protocol):
    version: str

    async def parse(self, path: Path) -> ParsedDocument: ...


@dataclass(frozen=True)
class Chunk:
    text: str
    heading_path: str
    page: int | None
    metadata: Mapping[str, str]


class Chunker(Protocol):
    version: str

    def chunk(self, doc: ParsedDocument, *, metadata: Mapping[str, str]) -> Sequence[Chunk]: ...
