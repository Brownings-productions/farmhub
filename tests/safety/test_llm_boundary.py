"""SPEC §2: the LLM is reached only through the ``LLMBackend`` protocol.

``lint-imports`` enforces the same rule structurally, but it runs as a separate CI
step. This lives in the safety suite so the boundary is checked by the same run that
checks every other §3 guarantee, and so a source file that imports vLLM fails the test
suite rather than only the linter.
"""

import ast
from pathlib import Path

SRC = Path(__file__).parents[2] / "src" / "farmhub"


def _imported_roots(path: Path) -> set[str]:
    """Every top-level package name ``path`` imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_source_file_imports_vllm() -> None:
    """vLLM is a container, not a library.

    SPEC §2 permits importing it inside ``modules/llm`` only; in fact nothing imports
    it at all, and that is what keeps "switching to Ollama is a config change, never a
    code change" true. If this ever needs relaxing, the SPEC needs changing first.
    """
    offenders = [
        str(path.relative_to(SRC)) for path in SRC.rglob("*.py") if "vllm" in _imported_roots(path)
    ]
    assert offenders == [], f"these files import vllm: {offenders}"


def test_only_the_llm_module_imports_the_openai_sdk() -> None:
    """The SDK is an implementation detail of one module (SPEC §8.2).

    Anywhere else it would mean a second way to talk to the model, outside the
    protocol that the orchestrator and the gateway are built around.
    """
    allowed = SRC / "modules" / "llm"
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if "openai" in _imported_roots(path) and allowed not in path.parents
    ]
    assert offenders == [], f"these files import the openai SDK: {offenders}"
