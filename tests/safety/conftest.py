"""tests/safety must never be skipped or xfailed (CLAUDE.md).

Any skipped or xfailed test in this directory is converted into a failure, so a safety
test cannot quietly stop guarding anything.
"""

from collections.abc import Generator
from typing import Any

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None]:
    outcome: Any = yield
    report = outcome.get_result()
    if report.skipped:  # includes xfail: pytest reports an expected failure as "skipped"
        report.outcome = "failed"
        if hasattr(report, "wasxfail"):
            # Left in place, pytest still files the report under "xfailed" and exits 0.
            del report.wasxfail
        report.longrepr = f"tests/safety must never be skipped or xfailed: {item.nodeid}"
