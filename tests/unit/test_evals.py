"""The eval harness's own logic (SPEC §2, §11 M1).

The harness needs a GPU; its rules do not. These cover the parts that decide whether a
run is trustworthy — the pin check, the startup-log parsing and the profile emitter —
without starting anything. No test in this repo may require a GPU (§9).
"""

import json
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from evals import report  # noqa: E402
from evals.backend import parse_startup  # noqa: E402
from evals.cases.tools import _violations  # noqa: E402
from evals.profile import Candidate, Matrix, ProfileError  # noqa: E402

SHA = "1355db6a052410cfd62085d94b58866fd0f2c3c5"


def candidate(**overrides: object) -> Candidate:
    data: dict = {
        "name": "c",
        "repo_id": "org/repo",
        "revision": SHA,
        "quantization": "modelopt_fp4",
    }
    data.update(overrides)
    return Candidate(**data)


# --- pinning (SPEC §2) ----------------------------------------------------------------


@pytest.mark.parametrize(
    "revision",
    ["main", "v1.0", "1355db6", "refs/pr/3", "HEAD", "A" * 40, SHA.upper(), ""],
)
def test_anything_that_is_not_a_commit_sha_is_refused(revision: str) -> None:
    """A tag is not a pin: an upstream tag was moved under a released model."""
    with pytest.raises(ProfileError, match="commit sha"):
        candidate(revision=revision).validate()


def test_a_commit_sha_is_accepted() -> None:
    candidate().validate()


def test_the_shipped_candidates_are_all_pinned() -> None:
    """Whatever is in data/candidates.json must be runnable without editing."""
    matrix = Matrix.load(REPO / "evals" / "data" / "candidates.json")
    assert matrix.candidates
    for entry in matrix.candidates:
        entry.validate()


def test_the_matrix_expands_over_kv_dtypes() -> None:
    """SPEC §2: each candidate is measured at both dtypes, never assumed."""
    matrix = Matrix([candidate(name="a"), candidate(name="b")])
    expanded = matrix.expand(["auto", "fp8"])
    assert [c.name for c in expanded] == ["a", "a-kvfp8", "b", "b-kvfp8"]
    assert [c.kv_cache_dtype for c in expanded] == ["auto", "fp8", "auto", "fp8"]


def test_the_candidate_env_carries_the_pin() -> None:
    env = candidate().env()
    assert env["VLLM_REVISION"] == SHA
    assert env["VLLM_MODEL"] == "org/repo"


# --- reading what vLLM says about itself ----------------------------------------------

LOG = """
INFO 09-20 12:00:00 Starting vLLM API server
INFO 09-20 12:00:10 Using AWQ Marlin kernel; the model has no native FP4 support on this device
INFO 09-20 12:01:02 Model loading took 17.42 GiB and 61.2 seconds
INFO 09-20 12:01:30 model weights take 17.42 GiB; KV cache takes 8.10 GiB
INFO 09-20 12:01:31 GPU KV cache size: 98,304 tokens
INFO 09-20 12:01:31 Maximum concurrency for 32,768 tokens per request: 3.00x
"""


def test_the_memory_numbers_come_from_the_startup_log(tmp_path: Path) -> None:
    startup = parse_startup(LOG, tmp_path / "x.log")
    assert startup.weights_gb == 17.42
    assert startup.kv_cache_gb == 8.10
    assert startup.kv_cache_tokens == 98304
    assert startup.max_concurrency == 3.00


def test_usable_context_is_the_cache_divided_by_concurrent_sessions() -> None:
    """The number §2 cares about, not the configured max_model_len (§1: 2-3 users)."""
    startup = parse_startup(LOG, Path("x"))
    assert startup.usable_context_at(3) == 32768


def test_an_unparsed_number_stays_none_rather_than_zero(tmp_path: Path) -> None:
    """A missing number must be visible. Zero would look like a measurement."""
    startup = parse_startup("INFO nothing useful here\n", tmp_path / "x.log")
    assert startup.weights_gb is None
    assert startup.kv_cache_gb is None
    assert startup.usable_context_at(3) is None


def test_the_kernel_actually_selected_is_captured(tmp_path: Path) -> None:
    """On this card an NVFP4 checkpoint can silently fall back to Marlin W4A16."""
    startup = parse_startup(LOG, tmp_path / "x.log")
    joined = " ".join(startup.kernel_lines).lower()
    assert "marlin" in joined
    assert "no native fp4" in joined


# --- emitting a profile (SPEC §2: measured, not estimated) ----------------------------


def test_a_measured_candidate_emits_a_pasteable_profile() -> None:
    entry = {
        "candidate": candidate(name="qwen").__dict__,
        "startup": {"weights_gb": 17.42, "kv_cache_gb": 8.1},
        "aux_reserve_gb": 5.0,
    }
    toml = report.profile_toml(entry, "evals/2026-09-20T12-00-00Z", 31.8)
    assert "[profiles.qwen]" in toml
    assert f'revision = "{SHA}"' in toml
    assert "weights_gb = 17.42" in toml
    assert 'measured_by = "evals/2026-09-20T12-00-00Z"' in toml


def test_an_unmeasured_candidate_emits_no_profile_at_all() -> None:
    """The §2 rule exists to stop exactly this; a plausible guess would defeat it."""
    entry = {
        "candidate": candidate(name="qwen").__dict__,
        "startup": {"weights_gb": None, "kv_cache_gb": None, "log_path": "captured.log"},
    }
    toml = report.profile_toml(entry, "evals/run", 31.8)
    # Everything emitted is commented out, so pasting it defines no profile at all.
    assert tomllib.loads(toml) == {}
    assert "not emitted" in toml


def test_the_emitted_profile_is_accepted_by_config_validation() -> None:
    """The two halves must agree: what the harness writes, config must load.

    Without this they drift, and the operator finds out by pasting a block that fails
    at startup.
    """
    from farmhub.core.config import ModelProfile

    entry = {
        "candidate": candidate(name="qwen", gpu_memory_utilization=0.82).__dict__,
        "startup": {"weights_gb": 17.42, "kv_cache_gb": 8.1},
        "aux_reserve_gb": 5.0,
    }
    toml = report.profile_toml(entry, "evals/2026-09-20T12-00-00Z", 31.8)
    parsed = tomllib.loads(toml)["profiles"]["qwen"]
    ModelProfile(**parsed)


# --- scoring rules --------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "duration_min": {"type": "integer", "minimum": 1, "maximum": 20},
        "zone": {"type": "string", "enum": ["beds", "benches"]},
    },
    "required": ["duration_min", "zone"],
}
SCHEMAS = {"water": SCHEMA}


def test_arguments_within_the_schema_are_clean() -> None:
    assert _violations("water", {"duration_min": 10, "zone": "beds"}, SCHEMAS) == []


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"duration_min": 120, "zone": "beds"}, "above maximum"),
        ({"duration_min": 0, "zone": "beds"}, "below minimum"),
        ({"duration_min": 5, "zone": "barn"}, "outside enum"),
        ({"duration_min": 5, "zone": "beds", "force": True}, "invented parameter"),
        ({"zone": "beds"}, "missing required"),
        ({"duration_min": "ten", "zone": "beds"}, "not an integer"),
    ],
)
def test_schema_violations_are_caught(args: dict, expected: str) -> None:
    """§7's declarations are only worth writing if a model that ignores them is scored."""
    problems = _violations("water", args, SCHEMAS)
    assert any(expected in p for p in problems), problems


def test_a_tool_that_was_never_offered_is_flagged() -> None:
    assert _violations("call_service", {}, SCHEMAS) == ["unknown tool 'call_service'"]


# --- the synthetic data is well formed ------------------------------------------------


@pytest.mark.parametrize(
    "name", ["tools.json", "grounding.json", "classifier.json", "latency.json", "candidates.json"]
)
def test_every_data_file_is_valid_json(name: str) -> None:
    json.loads((REPO / "evals" / "data" / name).read_text(encoding="utf-8"))


def test_the_tool_schemas_follow_the_spec_7_rules() -> None:
    """Enums or bounds on every parameter, explicit required, no free text (§3.1, §7)."""
    data = json.loads((REPO / "evals" / "data" / "tools.json").read_text(encoding="utf-8"))
    for tool in data["tools"]:
        schema = tool["function"]["parameters"]
        assert schema["additionalProperties"] is False
        assert "required" in schema
        for name, spec in schema["properties"].items():
            assert "description" in spec, name
            if spec["type"] == "integer":
                assert "minimum" in spec and "maximum" in spec, name
            if spec["type"] == "string":
                assert "enum" in spec or "maxLength" in spec, name


# --- the v0.29.0 startup log (real lines, from evals/results/logs) ---------------------

V029_LOG = """
INFO 09-22 18:07:48 [default_loader.py:430] Loading weights took 42.17 seconds
WARNING 09-22 18:07:48 [marlin.py:34] Your GPU does not have native support for FP4 \
computation but FP4 quantization is being used. Weight-only FP4 compression will be \
used leveraging the Marlin kernel.
INFO 09-22 18:07:56 [model_runner.py:404] Model loading took 19.55 GiB memory and \
1679.837327 seconds
INFO 09-22 18:09:43 [gpu_worker.py:625] Available KV cache memory: 4.9 GiB
INFO 09-22 18:09:43 [kv_cache_utils.py:2032] GPU KV cache size: 207,842 tokens, \
Maximum concurrency for 32,768 tokens per request: 6.34x
"""


def test_the_parser_reads_the_v029_memory_lines(tmp_path: Path) -> None:
    """vLLM reworded both lines in v0.29.0, and the old patterns silently missed them.

    A miss is not harmless: the run still scores, but the profile cannot be emitted,
    so the numbers SPEC §2 requires to be measured go missing.
    """
    startup = parse_startup(V029_LOG, tmp_path / "log.txt")
    assert startup.weights_gb == 19.55
    assert startup.kv_cache_gb == 4.9
    assert startup.kv_cache_tokens == 207842
    assert startup.max_concurrency == 6.34


def test_the_older_wording_still_parses(tmp_path: Path) -> None:
    older = (
        "INFO model weights take 16.20GiB; non_torch_memory takes 0.5GiB\n"
        "INFO GPU KV cache size: 100,000 tokens\n"
    )
    startup = parse_startup(older, tmp_path / "log.txt")
    assert startup.weights_gb == 16.20
    assert startup.kv_cache_tokens == 100000


def test_vision_tower_is_not_reported_loaded_from_backend_chatter(tmp_path: Path) -> None:
    """vLLM names the encoder's attention backend even when the tower is skipped.

    Treating that as "the vision tower loaded" made every text-only run look like it
    had paid for the tower, which is exactly the memory SPEC §2 wants excluded.
    """
    log = (
        "INFO [cuda.py:551] Using backend AttentionBackendEnum.FLASH_ATTN for vit attention\n"
        "INFO [mm_encoder_attention.py:372] Using FLASH_ATTN for MMEncoderAttention.\n"
    )
    assert parse_startup(log, tmp_path / "log.txt").vision_tower_loaded is None
