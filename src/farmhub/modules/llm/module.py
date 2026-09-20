"""The llm module (SPEC §8.2). Exposes no tools.

Its whole job is to own the lifecycle of the backend connection and report whether it
is usable. The backend itself is constructed in the composition root and shared through
``AppContext``, so the orchestrator can reach it without importing this module.

On the dev PC vLLM is started and stopped by hand (docs/RUNBOOK.md), so "the backend is
not there" is a normal condition, not a failure. ``startup`` raises
``DependencyUnavailable``, the registry starts the module degraded and retries with
backoff, and the health poll notices when vLLM comes back.
"""

from farmhub.core.context import AppContext
from farmhub.core.errors import DependencyUnavailable
from farmhub.core.protocols import HealthReport, HealthState, ToolSpec
from farmhub.modules.llm.client import OpenAICompatBackend


class LlmModule:
    """Owns the LLM backend connection."""

    name = "llm"
    version = "0.1.0"

    def __init__(self, backend: OpenAICompatBackend) -> None:
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
        """Idempotent: the registry calls this after a failed or degraded start too."""
        self._served_model = None
        await self._backend.aclose()

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
