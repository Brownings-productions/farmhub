"""Health and introspection (SPEC §5).

Reindex, pending actions and the audit tail land with the milestones that create them
(M3, M8, M5). M1 has health only.

These routes sit behind the same bearer token as everything else: module health names
which dependencies are down, which is not something to hand out unauthenticated.
``/healthz`` is the exception — a liveness probe has to work before anything else does,
and it reveals nothing beyond "the process is up".
"""

from typing import Any

from fastapi import APIRouter, Request

from farmhub.core.protocols import HealthState


def build_public_router() -> APIRouter:
    """Routes served without a token. Keep this as small as it is.

    A liveness probe has to answer before anything else works, including before a
    token is known to be right, and it reveals nothing beyond "the process is up".
    """
    router = APIRouter()

    @router.get("/healthz")
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    return router


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """Per-module health.

        A degraded module is reported, not hidden: on the dev PC vLLM is stopped by
        hand, and this is how you see that FarmHub knows.
        """
        state = request.app.state.farmhub
        reports = await state.app.registry.health()
        modules = {
            name: {"status": report.status.value, "detail": report.detail}
            for name, report in reports.items()
        }
        degraded = [
            name for name, report in reports.items() if report.status is not HealthState.HEALTHY
        ]
        return {
            "status": "degraded" if degraded else "ok",
            "degraded": degraded,
            "modules": modules,
            "dry_run": state.ctx.settings.dry_run,
        }

    return router


__all__ = ["build_public_router", "build_router"]
