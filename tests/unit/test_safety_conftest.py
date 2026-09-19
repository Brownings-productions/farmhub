"""The guard in tests/safety/conftest.py really turns skips and xfails into failures."""

from pathlib import Path

import pytest

GUARD = Path(__file__).parents[1] / "safety" / "conftest.py"


@pytest.mark.parametrize(
    "body",
    [
        "import pytest\ndef test_x():\n    pytest.skip('nope')\n",
        "import pytest\n@pytest.mark.skip\ndef test_x():\n    pass\n",
        "import pytest\n@pytest.mark.xfail\ndef test_x():\n    assert False\n",
    ],
    ids=["runtime-skip", "skip-marker", "xfail-marker"],
)
def test_skips_and_xfails_fail_in_the_safety_directory(
    pytester: pytest.Pytester, body: str
) -> None:
    pytester.makeconftest(GUARD.read_text())
    pytester.makepyfile(test_guarded=body)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    outcomes = result.parseoutcomes()
    # A skip raised during setup surfaces as an error, one raised in the body as a failure.
    assert outcomes.get("failed", 0) + outcomes.get("errors", 0) == 1
    assert "skipped" not in outcomes
    assert "xfailed" not in outcomes
    assert result.ret != 0


def test_passing_tests_are_unaffected(pytester: pytest.Pytester) -> None:
    pytester.makeconftest(GUARD.read_text())
    pytester.makepyfile(test_guarded="def test_x():\n    assert True\n")
    pytester.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=1)
