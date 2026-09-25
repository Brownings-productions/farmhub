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
from evals.backend import memory_used_gib, parse_startup  # noqa: E402
from evals.cases import common  # noqa: E402
from evals.cases import tools as tools_case  # noqa: E402
from evals.cases.tools import _violations  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "evals" / "data"
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
    assert startup.weights_gib == 17.42
    assert startup.kv_cache_gib == 8.10
    assert startup.kv_cache_tokens == 98304
    assert startup.max_concurrency == 3.00


def test_usable_context_is_the_cache_divided_by_concurrent_sessions() -> None:
    """The number §2 cares about, not the configured max_model_len (§1: 2-3 users)."""
    startup = parse_startup(LOG, Path("x"))
    assert startup.usable_context_at(3) == 32768


def test_an_unparsed_number_stays_none_rather_than_zero(tmp_path: Path) -> None:
    """A missing number must be visible. Zero would look like a measurement."""
    startup = parse_startup("INFO nothing useful here\n", tmp_path / "x.log")
    assert startup.weights_gib is None
    assert startup.kv_cache_gib is None
    assert startup.usable_context_at(3) is None


def test_the_kernel_actually_selected_is_captured(tmp_path: Path) -> None:
    """On this card an NVFP4 checkpoint can silently fall back to Marlin W4A16."""
    startup = parse_startup(LOG, tmp_path / "x.log")
    joined = " ".join(startup.kernel_lines).lower()
    assert "marlin" in joined
    assert "no native fp4" in joined


# --- the outside view of the card ------------------------------------------------------


class _Ran:
    """A stand-in for one `nvidia-smi` invocation."""

    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_memory_used_is_read_in_gib(monkeypatch: pytest.MonkeyPatch) -> None:
    """nvidia-smi reports MiB; the profile and the table are GiB throughout."""
    monkeypatch.setattr("evals.backend.subprocess.run", lambda *a, **k: _Ran(0, "26680\n"))
    assert memory_used_gib() == 26.05


@pytest.mark.parametrize("ran", [_Ran(1, ""), _Ran(0, ""), _Ran(0, "not a number\n")])
def test_an_unreadable_card_is_none_rather_than_zero(
    monkeypatch: pytest.MonkeyPatch, ran: _Ran
) -> None:
    """Zero would read as "the card is empty", which is a measurement, not a failure."""
    monkeypatch.setattr("evals.backend.subprocess.run", lambda *a, **k: ran)
    assert memory_used_gib() is None


# --- emitting a profile (SPEC §2: measured, not estimated) ----------------------------


def test_a_measured_candidate_emits_a_pasteable_profile() -> None:
    entry = {
        "candidate": candidate(name="qwen").__dict__,
        "startup": {"weights_gib": 17.42, "kv_cache_gib": 8.1},
        "aux_reserve_gib": 5.0,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    toml = report.profile_toml(entry, "evals/2026-09-20T12-00-00Z", 31.8)
    assert "[profiles.qwen]" in toml
    assert f'revision = "{SHA}"' in toml
    assert "weights_gib = 17.42" in toml
    assert 'measured_by = "evals/2026-09-20T12-00-00Z"' in toml
    # Both request parameters come from what the run actually sent, so the profile
    # serves the model the way it was scored.
    assert "chat_template_kwargs = { enable_thinking = false }" in toml
    assert "temperature = 0.0" in toml


def test_a_run_that_did_not_pin_sampling_emits_no_profile() -> None:
    """A score taken at the backend's default belongs to no reproducible profile."""
    entry = {
        "candidate": candidate(name="qwen").__dict__,
        "startup": {"weights_gib": 17.42, "kv_cache_gib": 8.1},
        "aux_reserve_gib": 5.0,
    }
    toml = report.profile_toml(entry, "evals/run", 31.8)
    assert tomllib.loads(toml) == {}
    assert "no temperature" in toml


def test_an_unmeasured_candidate_emits_no_profile_at_all() -> None:
    """The §2 rule exists to stop exactly this; a plausible guess would defeat it."""
    entry = {
        "candidate": candidate(name="qwen").__dict__,
        "startup": {"weights_gib": None, "kv_cache_gib": None, "log_path": "captured.log"},
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
        "startup": {"weights_gib": 17.42, "kv_cache_gib": 8.1},
        "aux_reserve_gib": 5.0,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
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
    assert startup.weights_gib == 19.55
    assert startup.kv_cache_gib == 4.9
    assert startup.kv_cache_tokens == 207842
    assert startup.max_concurrency == 6.34


def test_the_older_wording_still_parses(tmp_path: Path) -> None:
    older = (
        "INFO model weights take 16.20GiB; non_torch_memory takes 0.5GiB\n"
        "INFO GPU KV cache size: 100,000 tokens\n"
    )
    startup = parse_startup(older, tmp_path / "log.txt")
    assert startup.weights_gib == 16.20
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


# --- thinking disabled, and the tripwire for when it is ignored ------------------------


def test_every_case_asks_for_thinking_to_be_off() -> None:
    """One run means one mode. A case that forgot the flag would not be comparable."""
    body = common.payload("m", [{"role": "user", "content": "hei"}], max_completion_tokens=64)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_completion_tokens"] == 64


def test_extra_request_fields_survive() -> None:
    body = common.payload(
        "m", [{"role": "user", "content": "hei"}], max_completion_tokens=64, tools=[{"x": 1}]
    )
    assert body["tools"] == [{"x": 1}]


def test_a_reply_that_reasons_anyway_is_caught() -> None:
    """The parameter can be silently ignored by a backend or a chat template.

    This is the exact opener from the 2026-09-22 run, when thinking was still on. If it
    ever appears in a run that asked for thinking to be off, the quality scores from
    that run mean nothing, so it must not go unnoticed.
    """
    message = {"content": "Here's a thinking process:\n\n1. **Analyze User Input:**"}
    assert common.reasoning_in(message) is not None


def test_reasoning_tags_and_the_reasoning_field_are_caught() -> None:
    assert common.reasoning_in({"content": "<think>hmm</think>8,5 liter"}) is not None
    assert common.reasoning_in({"reasoning_content": "hmm", "content": "8,5 liter"}) is not None


def test_a_plain_answer_is_not_called_reasoning() -> None:
    assert common.reasoning_in({"content": "Motoroljen er 8,5 liter med filter."}) is None
    assert common.reasoning_in({"content": "SKF 6205-2RS."}) is None
    assert common.reasoning_in({}) is None


def test_truncation_is_read_from_the_finish_reason() -> None:
    assert common.truncated({"finish_reason": "length"}) is True
    assert common.truncated({"finish_reason": "stop"}) is False
    assert common.truncated({}) is False


def test_the_run_table_shows_reasoning_and_answer_time() -> None:
    """A run that reasoned must say so in the same table as the scores it invalidates."""
    result = {
        "run_id": "r1",
        "environment": {"vllm_image": "i", "gpu": "g", "driver": "d", "memory_total": "m"},
        "aux_reserve_gib": 5.0,
        "concurrent_sessions": 3,
        "card_total_gib": 31.8,
        "candidates": [
            {
                "candidate": {"name": "c"},
                "loaded": True,
                "kernel": "marlin-fallback",
                "startup": {"weights_gib": 19.55, "kv_cache_gib": 4.9, "kv_cache_tokens": 207842},
                "usable_context_at_concurrency": 69280,
                "latency": {
                    "cold_ttft_s": 0.4,
                    "warm_ttft_s_median": 0.08,
                    "reasoning_emitted": 2,
                    "single": {"answer_s_median": 3.2, "gen_tokens_per_s_median": 61.0},
                    "at_concurrency": {
                        "answer_s_median": 5.1,
                        "gen_tokens_per_s_median": 38.0,
                    },
                },
                "tools": {"score": 0.88, "truncated": 1},
                "grounding": {"score": 0.83},
                "classifier": {"json_validity": 1.0},
                "profile_toml": "",
            }
        ],
    }
    table = report.markdown(result)
    assert "3.2 / 5.1" in table
    assert "61.0 / 38.0" in table
    assert report.total(result["candidates"][0], "reasoning_emitted") == 2
    assert report.total(result["candidates"][0], "truncated") == 1


# --- sampling is pinned, and repeats make instability visible -------------------------


def test_every_request_pins_temperature() -> None:
    """Two runs disagreed about one tool case on the same weights (MODEL_EVAL.md)."""
    body = common.payload("m", [{"role": "user", "content": "hei"}], max_completion_tokens=64)
    assert body["temperature"] == 0.0


def test_the_tool_score_is_reported_as_a_range() -> None:
    assert report._range({"score_min": 0.75, "score_max": 0.88}) == "0.75 to 0.88"
    assert report._range({"score_min": 0.75, "score_max": 0.75}) == "0.75"
    assert report._range({"score": 0.5}) == "0.5"


def test_out_of_enum_buckets_are_reported_separately() -> None:
    """Substituting a real area is the dangerous outcome and needs its own number."""
    cell = report._buckets({"out_of_enum": {"declined": 4, "substituted": 1, "invented": 0}})
    assert cell == "4/1/0"


# --- what the model did with a value that is not in the enum --------------------------

AREA_SCHEMAS = {
    "set_indoor_light": {
        "properties": {"area": {"type": "string", "enum": ["kitchen", "workshop", "hall"]}},
        "required": ["area"],
    }
}
BARN_CASE = {"id": "out-of-enum-area", "out_of_enum_param": "area"}


def test_declining_an_out_of_enum_request_is_its_own_bucket() -> None:
    record = {"called": None}
    assert tools_case._out_of_enum_bucket(BARN_CASE, record, AREA_SCHEMAS) == "declined"


def test_substituting_a_different_real_area_is_distinguished_from_inventing_one() -> None:
    """The 2026-09-23 failure: barn was asked for, workshop was called.

    Schema-valid, audited as permitted, and the wrong room — which is why it must not be
    counted with the invented values that §7's schema already stops.
    """
    substituted = {"called": "set_indoor_light", "args": {"area": "workshop"}}
    invented = {"called": "set_indoor_light", "args": {"area": "barn"}}
    assert tools_case._out_of_enum_bucket(BARN_CASE, substituted, AREA_SCHEMAS) == "substituted"
    assert tools_case._out_of_enum_bucket(BARN_CASE, invented, AREA_SCHEMAS) == "invented"


def test_a_case_with_no_enum_parameter_is_not_bucketed() -> None:
    record = {"called": None}
    assert tools_case._out_of_enum_bucket({"id": "plain"}, record, AREA_SCHEMAS) is None


def test_the_out_of_enum_cases_cover_both_languages() -> None:
    """A satellite in the barn is heard in Norwegian, so the cases have to be too."""
    data = json.loads((DATA_DIR / "tools.json").read_text(encoding="utf-8"))
    cases = [c for c in data["cases"] if "out_of_enum_param" in c]
    assert len(cases) >= 5
    assert any("fjøset" in c["utterance"] for c in cases)
    assert any("garage" in c["utterance"] for c in cases)
    enums = {
        name: (schema["properties"].get(c["out_of_enum_param"]) or {}).get("enum")
        for c in cases
        for name, schema in (
            (t["function"]["name"], t["function"]["parameters"]) for t in data["tools"]
        )
        if c["out_of_enum_param"] in schema["properties"]
    }
    # Every case names a parameter that some offered tool actually constrains by enum,
    # or the bucketing would silently call everything "invented".
    assert all(values for values in enums.values())


def test_every_out_of_enum_case_has_an_in_enum_control() -> None:
    """A decline must be attributable to the enum, not to the phrasing.

    "Kan du skru på lyset i stabburet?" declined 3/3 and looked like correct enum
    discipline. Its control — the same polite form naming the workshop, which *is* in the
    enum — declined 2 of 3 too, so the model was refusing the phrasing (MODEL_EVAL.md,
    2026-09-25). Every out-of-enum case needs that twin or its score means nothing.
    """
    data = json.loads((DATA_DIR / "tools.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    needs_control = {c["id"] for c in cases if c.get("out_of_enum_param")}
    controlled = {target for c in cases for target in c.get("control_for", [])}
    assert needs_control - controlled == set()
    # A control must itself expect a call, or it controls for nothing.
    for case in cases:
        if case.get("control_for"):
            assert case.get("expect_tool") is not None, case["id"]


def test_the_long_context_latency_prompt_is_rag_sized() -> None:
    """M4 returns top-5 reranked chunks; prefill at that size is what Marlin taxes."""
    data = json.loads((DATA_DIR / "latency.json").read_text(encoding="utf-8"))
    contexts = data["long_context"]["contexts"]
    for context in contexts:
        # ~10,100 tokens as measured. The run records the server's own prompt_tokens;
        # this only guards against the file being trimmed to something that no longer
        # tests prefill.
        assert 15000 < len(context) < 30000
        assert "Nm" in context
        assert any(maker in context for maker in ("SKF", "FAG", "NSK", "NTN"))


def test_every_rag_stream_gets_its_own_cold_context() -> None:
    """Sharing one context measures the prefix cache instead of load.

    With a single context the three concurrent streams reused what the single stream had
    just cached and came out *faster* than it (docs/MODEL_EVAL.md, 2026-09-25). So there
    has to be one context per stream, each differing from its first line.
    """
    data = json.loads((DATA_DIR / "latency.json").read_text(encoding="utf-8"))
    contexts = data["long_context"]["contexts"]
    assert len(contexts) >= data["concurrent"] + 1
    assert len(set(contexts)) == len(contexts)
    # Differing somewhere is not enough: a shared opening is a shared prefix.
    assert len({c.splitlines()[0] for c in contexts}) == len(contexts)
    # Same size, or the comparison between them is not a comparison.
    lengths = [len(c) for c in contexts]
    assert (max(lengths) - min(lengths)) / min(lengths) < 0.1
