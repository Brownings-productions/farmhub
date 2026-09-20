# Runbook

Stub. Sections are filled in as the milestones that need them land.

## Configuration

- Copy `config/farmhub.example.toml` to `config/farmhub.toml` (gitignored).
- Validate: `uv run farmhub --config config/farmhub.toml config check`. It exits 2 on invalid
  config and prints the effective safety settings first.
- `dry_run` is ON unless something turns it off. When it is off, startup logs a warning.

## vLLM: starting and stopping (M1)

The inference backend runs as a container defined in `deploy/vllm/compose.yaml`. That
file is identical on the dev PC and on hub; only who starts it differs.

First time:

```sh
cp deploy/vllm/env.example deploy/vllm/.env   # .env is gitignored
$EDITOR deploy/vllm/.env                      # model, revision, memory, kv-cache dtype
./deploy/vllm/vllm.sh config                  # shows the exact command that will run
```

Day to day on the dev PC:

```sh
./deploy/vllm/vllm.sh up        # background; does not wait for the model to load
./deploy/vllm/vllm.sh wait      # blocks until /health answers (cold load takes minutes)
./deploy/vllm/vllm.sh status    # container state plus GPU memory in use
./deploy/vllm/vllm.sh logs -f
./deploy/vllm/vllm.sh down      # frees the VRAM
```

**Start and stop are manual on the dev PC, on purpose.** That machine is used for other
things and a 30 GB model must not hold the GPU all day. Docker Desktop on WSL2 has no
systemd socket activation and vLLM has no idle shutdown, so anything automatic would be
a bespoke sidecar. `compose.yaml` sets `restart: "no"` so the container never returns on
its own after a Docker Desktop restart.

*Deferred option, not built:* an idle-stop sidecar that polls vLLM's `/metrics` for time
since the last request and runs `down` after N minutes. Worth doing only if the manual
cycle becomes annoying.

On hub it is always on: `deploy/systemd/farmhub-vllm.service` runs the same compose file
at boot. Install instructions are in the unit file.

**FarmHub does not require vLLM to be running.** The `llm` module starts degraded when
the backend is unreachable and the registry retries with backoff, so bringing vLLM up or
down under a running FarmHub is safe — answers fail cleanly in the meantime.

### Pinning a model revision

SPEC §2 requires every profile to pin a commit sha. A tag is not a pin: one NVFP4 upload
of Qwen3.6-35B-A3B was silently replaced on 2026-07-10 with weights that loop, and a tag
would have followed it on the next pull.

```sh
./deploy/vllm/vllm.sh resolve nvidia/Qwen3.6-35B-A3B-NVFP4
```

Paste the result into `VLLM_REVISION`, and into the matching FarmHub profile. If a sha
you already pinned has moved, find out what changed before following it.

### Checkpoint cache and disk

`HF_CACHE_DIR` must point inside the Linux filesystem (`~/.cache/huggingface`), never at
`/mnt/c` or `/mnt/d` — the Windows file bridge is far too slow for a 20 GB checkpoint and
vLLM mmaps the safetensors at load.

Budget against `C:`, not against what `df` reports inside WSL: that figure is the virtual
disk's ceiling, not free space. At 17–32 GB per checkpoint only a few candidates fit at
once, so prune between evaluation runs. Deleting files does not shrink the WSL virtual
disk by itself.

### Ollama fallback

`deploy/ollama/compose.yaml` runs Ollama instead, for when vLLM will not start on a given
card or when developing against a small model. Switching is a config change only: point
`llm.base_url` at `http://127.0.0.1:11434/v1`. No Python changes (SPEC §2).

Neither vLLM nor Ollama has any authentication of its own, so both stay bound to
loopback on every host. FarmHub is the only client, and it carries the bearer token of
§3.6.

## Home Assistant side (SPEC §3.4), filled in at M5

FarmHub cannot verify these from Python. Check each by hand before enabling any T2 tool:

- [ ] Every irrigation, watering and ventilation script enforces a maximum runtime inside
      the script itself (not in FarmHub). Record the value next to the tool's `max_runtime_s`.
- [ ] Heating and pump control exist only as deterministic HA automations and fixed intents.
- [ ] The HA user behind FarmHub's token: confirm whether "non-admin" actually restricts
      service calls. Assume it does not (docs/DECISIONS.md).
- [ ] Each valve and pump circuit has an independent hardware cutoff that does not route
      through Home Assistant (SPEC §10).
