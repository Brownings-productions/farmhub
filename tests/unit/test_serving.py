"""The profile and the vLLM server's environment must not drift apart (SPEC §2).

A profile records what was measured; `deploy/vllm/.env` decides what is served. Nothing
connected them, and the eval harness rewrites that file for every candidate it runs — so
after a matrix run it holds the last candidate measured rather than the chosen one.
"""

import tempfile
from pathlib import Path

import pytest

from farmhub.core.config import Settings, load_settings
from farmhub.core.errors import ConfigError
from farmhub.core.serving import (
    OWNED_KEYS,
    compare_env,
    env_from_profile,
    parse_env_file,
    render_env_file,
    verify_settings,
)

# The measured candidate A profile, which is the one whose flags matter most: without
# --language-model-only the vision tower loads and its memory numbers stop describing it.
PROFILE_TOML = """
[llm]
profile = "primary"
model = "farmhub-primary"
serving_env_file = "{env_path}"

[profiles.primary]
repo_id = "nvidia/Qwen3.6-35B-A3B-NVFP4"
revision = "1355db6a052410cfd62085d94b58866fd0f2c3c5"
quantization = "modelopt_fp4"
kv_cache_dtype = "auto"
gpu_memory_utilization = 0.82
max_model_len = 32768
weights_gib = 19.55
kv_cache_gib = 4.9
aux_reserve_gib = 5.0
card_total_gib = 31.84
measured_by = "evals/2026-09-25T15-42-33Z"
chat_template_kwargs = {{ enable_thinking = false }}
temperature = 0.0
server_args = ["--block-size", "128", "--max-num-seqs", "8", "--tool-call-parser", \
"hermes", "--language-model-only"]
"""

# What an operator's file carries that no profile should touch.
OPERATOR_KEYS = {
    "VLLM_IMAGE_TAG": "v0.29.0",
    "HF_CACHE_DIR": "/home/someone/.cache/huggingface",
    "VLLM_PORT": "8000",
    "VLLM_BIND_HOST": "127.0.0.1",
    "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
}


def settings_with(tmp_path: Path, env_text: str | None = None) -> Settings:
    """Settings pointing at a throwaway .env, written when ``env_text`` is given."""
    env_path = tmp_path / "vllm.env"
    if env_text is not None:
        env_path.write_text(env_text, encoding="utf-8")
    config = tmp_path / "farmhub.toml"
    config.write_text(PROFILE_TOML.format(env_path=env_path.as_posix()), encoding="utf-8")
    return load_settings(config)


def rendered(tmp_path: Path) -> str:
    settings = settings_with(tmp_path)
    profile = settings.active_profile()
    assert profile is not None
    return render_env_file(profile, settings.llm.model, OPERATOR_KEYS)


# --- rendering -------------------------------------------------------------------------


def test_the_rendered_file_carries_the_profiles_flags(tmp_path: Path) -> None:
    values = parse_env_file_text(rendered(tmp_path))
    assert values["VLLM_MODEL"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert values["VLLM_REVISION"] == "1355db6a052410cfd62085d94b58866fd0f2c3c5"
    assert values["VLLM_BLOCK_SIZE"] == "128"
    assert values["VLLM_TOOL_CALL_PARSER"] == "hermes"
    # The flag compose reads separately is split out, so it is not passed twice.
    assert values["VLLM_EXTRA_ARGS"] == "--language-model-only"
    assert "--block-size" not in values["VLLM_EXTRA_ARGS"]


def test_the_operators_own_keys_survive_a_profile_switch(tmp_path: Path) -> None:
    """Image, cache, port, bind address and the WSL flag describe the machine."""
    values = parse_env_file_text(rendered(tmp_path))
    for key, value in OPERATOR_KEYS.items():
        assert values[key] == value


def test_a_rendered_file_passes_the_comparison(tmp_path: Path) -> None:
    """Render then verify: the whole point is that this round trip is clean."""
    text = rendered(tmp_path)
    settings = settings_with(tmp_path, text)
    assert verify_settings(settings) is None


# --- the mismatch that matters ---------------------------------------------------------


def test_a_hand_edited_file_that_drops_the_text_only_flag_fails(tmp_path: Path) -> None:
    """The failure this exists to catch (docs/MODEL_EVAL.md).

    Without --language-model-only the vision tower loads, weights grow and the KV cache
    shrinks — and every other check still passes, because config validation is arithmetic
    over declared numbers rather than a probe.
    """
    edited = rendered(tmp_path).replace("VLLM_EXTRA_ARGS=--language-model-only", "VLLM_EXTRA_ARGS=")
    settings = settings_with(tmp_path, edited)
    with pytest.raises(ConfigError, match="VLLM_EXTRA_ARGS"):
        verify_settings(settings)


def test_a_file_left_behind_by_another_candidate_fails(tmp_path: Path) -> None:
    """After a matrix run the file holds the last candidate measured, not the chosen one."""
    other = "\n".join(
        [
            "VLLM_MODEL=gghfez/Mistral-Small-3.2-24B-Instruct-hf-AWQ",
            "VLLM_REVISION=9f337c5a5e23a14e9e665df29a16ca2fc519149b",
            "VLLM_SERVED_MODEL_NAME=farmhub-primary",
            "VLLM_QUANTIZATION=awq_marlin",
            "VLLM_KV_CACHE_DTYPE=auto",
            "VLLM_MAX_MODEL_LEN=32768",
            "VLLM_GPU_MEMORY_UTILIZATION=0.82",
            "VLLM_BLOCK_SIZE=16",
            "VLLM_MAX_NUM_SEQS=8",
            "VLLM_TOOL_CALL_PARSER=mistral",
            "VLLM_EXTRA_ARGS=--chat-template /vllm-workspace/examples/x.jinja",
        ]
    )
    settings = settings_with(tmp_path, other)
    with pytest.raises(ConfigError) as caught:
        verify_settings(settings)
    message = str(caught.value)
    for key in ("VLLM_MODEL", "VLLM_QUANTIZATION", "VLLM_BLOCK_SIZE", "VLLM_EXTRA_ARGS"):
        assert key in message


def test_a_mismatch_names_keys_and_never_values(tmp_path: Path) -> None:
    """The report reaches a log file and a terminal, beside secrets."""
    edited = rendered(tmp_path) + "\nVLLM_MODEL=someone/private-model\n"
    settings = settings_with(tmp_path, edited)
    with pytest.raises(ConfigError) as caught:
        verify_settings(settings)
    assert "someone/private-model" not in str(caught.value)
    assert "nvidia/Qwen3.6-35B-A3B-NVFP4" not in str(caught.value)


@pytest.mark.parametrize(
    "key",
    ["VLLM_MODEL", "VLLM_REVISION", "VLLM_QUANTIZATION", "VLLM_MAX_MODEL_LEN", "VLLM_BLOCK_SIZE"],
)
def test_every_owned_key_is_guarded(tmp_path: Path, key: str) -> None:
    """A key this module renders but does not compare would drift silently."""
    edited = "\n".join(
        line if not line.startswith(f"{key}=") else f"{key}=wrong"
        for line in rendered(tmp_path).splitlines()
    )
    settings = settings_with(tmp_path, edited)
    with pytest.raises(ConfigError, match=key):
        verify_settings(settings)


# --- nothing to compare is not a pass --------------------------------------------------


def test_a_missing_file_says_so_rather_than_passing_quietly(tmp_path: Path) -> None:
    settings = settings_with(tmp_path)  # no file written
    note = verify_settings(settings)
    assert note is not None
    assert "does not exist" in note


def test_no_profile_means_nothing_to_compare() -> None:
    """Ollama and the test double have no measured profile, and must still start."""
    note = verify_settings(load_settings())
    assert note is not None
    assert "no active profile" in note


def test_the_check_can_be_switched_off(tmp_path: Path) -> None:
    """Right when vLLM runs on another host whose .env this machine cannot see."""
    config = tmp_path / "farmhub.toml"
    config.write_text(
        PROFILE_TOML.format(env_path=(tmp_path / "vllm.env").as_posix()), encoding="utf-8"
    )
    settings = load_settings(config, overrides={"llm": {"serving_env_file": None}})
    note = verify_settings(settings)
    assert note is not None
    assert "unset" in note


# --- the mapping itself ----------------------------------------------------------------


def test_server_args_ending_in_a_dangling_flag_is_an_error(tmp_path: Path) -> None:
    """A profile recording `--block-size` with no value records nothing useful."""
    settings = settings_with(tmp_path)
    profile = settings.active_profile()
    assert profile is not None
    broken = profile.model_copy(update={"server_args": ["--block-size"]})
    with pytest.raises(ConfigError, match="block-size"):
        env_from_profile(broken, "farmhub-primary")


def test_compare_env_reads_quoted_values(tmp_path: Path) -> None:
    """compose strips one layer of quotes; a parser that did not would cry mismatch."""
    quoted = "\n".join(
        f'{key}="{value}"' for key, value in parse_env_file_text(rendered(tmp_path)).items()
    )
    path = tmp_path / "quoted.env"
    path.write_text(quoted, encoding="utf-8")
    settings = settings_with(tmp_path)
    profile = settings.active_profile()
    assert profile is not None
    assert compare_env(profile, settings.llm.model, path) == ()


def test_owned_keys_are_exactly_what_is_rendered(tmp_path: Path) -> None:
    """Whatever is rendered must be compared, or it can drift unnoticed."""
    settings = settings_with(tmp_path)
    profile = settings.active_profile()
    assert profile is not None
    assert set(env_from_profile(profile, settings.llm.model)) == set(OWNED_KEYS)


def parse_env_file_text(text: str, *, at: Path | None = None) -> dict[str, str]:
    """``parse_env_file`` over a string, via a file the caller does not have to name."""
    path = at if at is not None else Path(tempfile.mkdtemp()) / "parsed.env"
    path.write_text(text, encoding="utf-8")
    return parse_env_file(path)
