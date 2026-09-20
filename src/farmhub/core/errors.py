"""Exception hierarchy.

The split between ConfigError and DependencyUnavailable is load-bearing: the module
registry fails fast on the former and starts the module degraded on the latter
(docs/DECISIONS.md, "Two failure kinds").
"""


class FarmHubError(Exception):
    """Base class for all FarmHub errors."""


class ConfigError(FarmHubError):
    """Invalid configuration, including bad tool files. Always fatal at startup."""


class ModuleLoadError(FarmHubError):
    """A module could not be loaded, or violates a registry invariant."""


class DependencyUnavailable(FarmHubError):
    """A runtime dependency (HA, database, ...) is unreachable.

    Raised from ``Module.startup`` to request a degraded start: the module is marked
    unhealthy, its tools are withheld from every tool list, and startup is retried.
    """


class SafetyViolation(FarmHubError):
    """Code attempted something the SPEC §3 rules forbid."""


class LLMError(FarmHubError):
    """The LLM backend failed, or returned something that could not be used.

    Distinct from ``DependencyUnavailable``: that means the backend is down and the
    module should start degraded, while this means one call failed. The §4 classifier
    turns it into the safe default rather than guessing.
    """


class AuditWriteError(FarmHubError):
    """An audit row could not be written. The gateway treats this as deny (§3.7)."""
