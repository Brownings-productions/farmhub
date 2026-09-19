import io
import json
from pathlib import Path

from farmhub.core.logging import configure_logging, get_logger


class FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_json_lines_go_to_the_file(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "farmhub.jsonl"
    configure_logging("INFO", log_file, stream=io.StringIO())
    get_logger("test").info("hello", thing=42)
    line = log_file.read_text().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"] == "hello"
    assert record["thing"] == 42
    assert record["level"] == "info"
    assert "timestamp" in record


def test_console_is_pretty_on_a_tty() -> None:
    stream = FakeTty()
    configure_logging("INFO", None, stream=stream)
    get_logger("test").info("hello", thing=42)
    output = stream.getvalue()
    assert "hello" in output
    assert not output.lstrip().startswith("{")


def test_console_is_json_when_not_a_tty() -> None:
    stream = io.StringIO()
    configure_logging("INFO", None, stream=stream)
    get_logger("test").info("hello")
    assert json.loads(stream.getvalue().strip().splitlines()[-1])["event"] == "hello"


def test_reconfiguring_replaces_handlers(tmp_path: Path) -> None:
    log_file = tmp_path / "a.jsonl"
    for _ in range(3):
        configure_logging("INFO", log_file, stream=io.StringIO())
    get_logger("test").info("once")
    assert log_file.read_text().count('"once"') == 1


def test_level_filters_records(tmp_path: Path) -> None:
    log_file = tmp_path / "a.jsonl"
    configure_logging("WARNING", log_file, stream=io.StringIO())
    get_logger("test").info("quiet")
    get_logger("test").warning("loud")
    text = log_file.read_text()
    assert "loud" in text
    assert "quiet" not in text
