"""No tool handler can be invoked except through the gateway (docs/DECISIONS.md).

Python cannot make this impossible, so it is enforced in two layers: a runtime seal that
handlers check, and a static scan that fails if any source file other than the two that own
the mechanism touches ``.handler`` or the seal.
"""

import ast
import dataclasses
import uuid
from pathlib import Path

import pytest
from support import MemoryAuditSink, StubModule, make_app_context, make_tool

from farmhub.core.errors import SafetyViolation
from farmhub.core.gateway import Gateway
from farmhub.core.protocols import CallOrigin, Decision, Tier, ToolCall, ToolContext
from farmhub.core.registry import ModuleRegistry

SRC = Path(__file__).parents[2] / "src" / "farmhub"
ALLOWED = {SRC / "core" / "protocols.py", SRC / "core" / "gateway.py"}


def forged_context(seal: object = None) -> ToolContext:
    return ToolContext(
        satellite="kitchen",
        session="s",
        scope=frozenset({"kitchen"}),
        dry_run=False,
        context_tainted=False,
        call_id=uuid.uuid4(),
        _seal=seal,
    )


async def test_handler_refuses_a_context_without_the_seal() -> None:
    probe = make_tool("read")
    with pytest.raises(SafetyViolation, match="outside the gateway"):
        await probe.spec.handler(ToolCall("read", {}), forged_context())
    assert probe.calls == []


async def test_handler_refuses_a_forged_seal() -> None:
    probe = make_tool("read")
    with pytest.raises(SafetyViolation):
        await probe.spec.handler(ToolCall("read", {}), forged_context(seal=object()))
    assert probe.calls == []


async def test_handler_still_sealed_after_dataclasses_replace() -> None:
    probe = make_tool("read")
    clone = dataclasses.replace(probe.spec, description="copy")
    with pytest.raises(SafetyViolation):
        await clone.handler(ToolCall("read", {}), forged_context())
    assert probe.calls == []


@pytest.mark.parametrize("tier", [Tier.READ, Tier.COMFORT, Tier.FORBIDDEN])
async def test_direct_call_is_refused_at_every_tier(tier: Tier) -> None:
    probe = make_tool("t", tier)
    with pytest.raises(SafetyViolation):
        await probe.spec.handler(ToolCall("t", {}), forged_context())
    assert probe.calls == []


async def test_the_gateway_can_invoke_a_handler() -> None:
    probe = make_tool("read")
    ctx = make_app_context(audit=MemoryAuditSink())
    registry = ModuleRegistry([StubModule("m", tools=[probe.spec])])
    await registry.startup(ctx)
    outcome = await Gateway(ctx, registry).invoke(
        ToolCall("read", {}), CallOrigin("kitchen", "s", frozenset({"kitchen"}))
    )
    assert outcome.decision is Decision.EXECUTED
    assert len(probe.calls) == 1


# --- static scan ----------------------------------------------------------------------


def violations(source: str) -> list[str]:
    """Names touching the handler or the seal in ``source``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in {"handler", "GATEWAY_SEAL"}:
            found.append(node.attr)
        elif isinstance(node, ast.Name) and node.id == "GATEWAY_SEAL":
            found.append(node.id)
        elif isinstance(node, ast.ImportFrom):
            found.extend(a.name for a in node.names if a.name == "GATEWAY_SEAL")
    return found


@pytest.mark.parametrize(
    "source",
    [
        "spec.handler(call, ctx)",
        "x = tool.spec.handler",
        "from farmhub.core.protocols import GATEWAY_SEAL",
        "import farmhub.core.protocols as p\np.GATEWAY_SEAL",
        "ToolContext(_seal=GATEWAY_SEAL)",
    ],
)
def test_scanner_detects_violations(source: str) -> None:
    assert violations(source)


def test_only_the_gateway_and_protocols_touch_handlers_or_the_seal() -> None:
    offenders = {
        str(path.relative_to(SRC)): found
        for path in sorted(SRC.rglob("*.py"))
        if path not in ALLOWED and (found := violations(path.read_text()))
    }
    assert offenders == {}


def test_the_allowed_files_exist() -> None:
    assert all(p.is_file() for p in ALLOWED)
