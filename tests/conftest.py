import os
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """No FARMHUB_* variable or config/farmhub.toml from the host leaks into a test."""
    for key in [k for k in os.environ if k.startswith("FARMHUB_")]:
        monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)
    yield
