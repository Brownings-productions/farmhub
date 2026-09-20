"""The llm module (SPEC §8.2). Exposes no tools.

Its whole job is to report whether the backend is usable. The backend object itself is
constructed by the composition root and shared through ``AppContext``, so the
orchestrator can reach it without importing this module — and so this module must not
close it.

That division matters: the registry calls ``shutdown`` after a degraded start and
before every retry, so a module that closed the shared client there would leave every
later request failing with "the client has been closed", including after vLLM came
back. The module owns readiness; the composition root owns the connection.

On the dev PC vLLM is started and stopped by hand (docs/RUNBOOK.md), so "the backend is
not there" is a normal condition, not a failure. ``startup`` raises
``DependencyUnavailable``, the registry starts the module degraded and retries with
backoff, and the health poll notices when vLLM comes back.
"""

from typing import Protocol

from farmhub.core.context import AppContext
from farmhub.core.errors import DependencyUnavailable
from farmhub.core.protocols import HealthReport, HealthState, LLMBackend, ToolSpec


class LLMService(LLMBackend, Protocol):
    """An ``LLMBackend`` whose connection has a lifecycle this module owns.

    Separate from ``LLMBackend`` because the rest of the app only ever issues
    completions: probing and closing are this module's business, and nothing outside
    it should be able to reach for them.
    """

    async def probe(self) -> str: ...

    async def aclose(self) -> None: ...


class LlmModule:
    """Owns the LLM backend connection."""

    name = "llm"
    version = "0.1.0"

    def __init__(self, backend: LLMService) -> None:
        self._backend = backend
        self._served_model: str | None = None

    async def startup(self, ctx: AppContext) -> None:
        """Probe the backend. Unreachable means a degraded start, not a crash."""
        self._served_model = await self._backend.probe()
        ctx.log.info(
            "llm_backend_ready",
            module=self.name,
            model=self._served_model,
            base_url=ctx.settings.llm.base_url,
        )

    async def shutdown(self) -> None:
        """Drop readiness state only.

        Idempotent, because the registry calls this after a failed or degraded start
        and before each retry. It deliberately does not close the backend: see the
        module docstring.
        """
        self._served_model = None

    def tools(self) -> list[ToolSpec]:
        """None. The LLM is how tools are chosen, never a tool itself (§8.2)."""
        return []

    async def health(self) -> HealthReport:
        """Re-probe. A backend that went away is reported, not assumed still there."""
        try:
            model = await self._backend.probe()
        except DependencyUnavailable as exc:
            return HealthReport(HealthState.UNHEALTHY, str(exc))
        return HealthReport(HealthState.HEALTHY, f"serving {model}")
