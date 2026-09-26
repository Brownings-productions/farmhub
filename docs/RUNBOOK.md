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

### The serving profile: what every field means

A `[profiles.*]` block is a record of a measured evaluation run, not a set of knobs to
tune. **The evals harness writes it; nobody types it** (`docs/MODEL_EVAL.md`), and
`config check` refuses a profile that breaks any of the rules below.

Memory fields are **GiB** — the unit vLLM reports and the eval records — which is why
they are named `_gib`. No conversion happens anywhere:

| Field | Comes from | Notes |
|---|---|---|
| `weights_gib` | vLLM's startup log | Carries over to `hub`: same card, same checkpoint |
| `kv_cache_gib` | vLLM's startup log | Does **not** carry over: it is whatever was left on this machine (SPEC §2) |
| `aux_reserve_gib` | declared | Whisper + BGE-M3 + reranker beside vLLM. ~5 in Phase 1, 0 on `hub`. Still an **estimate** — no helper model has been measured yet |
| `card_total_gib` | `nvidia-smi` | 31.84 for the RTX 5090 (32,607 MiB) |

Validation rejects `weights_gib + kv_cache_gib + aux_reserve_gib` over the card, a
`gpu_memory_utilization` that does not leave `aux_reserve_gib` free, and
`weights_gib + kv_cache_gib` over vLLM's own share. That last check ignores peak
activation and CUDA graph memory (~1.2 GiB on the measured profile), so **a profile that
only just passes has not actually been tried** — vLLM's own preflight is the real guard.

Two fields are request parameters rather than memory, and both are required:

- **`chat_template_kwargs`** is sent on every request, including the classifier's. For
  the M1 candidates it is `{ enable_thinking = false }`. Leave thinking on and the model
  spends its whole token budget reasoning: part-number grounding scored 2 of 6 by never
  reaching an answer, and what a satellite reads aloud is a truncated ramble. It lives in
  the profile because the next model may spell the switch differently. An empty table is
  allowed and startup warns; a **missing** key fails startup.
- **`temperature`** is what the profile was measured at, used whenever the caller names
  none. Omitting it would mean the backend's own default, which is how two eval runs
  produced tool scores that could not be compared with each other.

Both are printed by `config check` and logged at startup, so a running FarmHub says on
its first lines how it is sampling.

### Phase 1: switching FarmHub off for gaming

The dev PC is both the runtime host and the gaming machine, and the 5090 cannot do both at
once. This is manual on purpose (SPEC §14 Q9): nothing auto-starts, so the card is never
reclaimed at a moment nobody chose.

**Set up once.** The monitor plugs into the **motherboard**, not the 5090 — that is what
gives FarmHub the whole card, and a monitor on the 5090 costs ~1.66 GiB, which pushes the
helper models below `aux_reserve_gib`. Games still render on the 5090: set each one to
**High performance** in Windows → Settings → System → Display → Graphics.

**Two desktop shortcuts.** `FarmHub OFF.bat`:

```bat
wsl bash -lc "cd ~/code/farmhub && ./deploy/vllm/vllm.sh down"
```

`FarmHub ON.bat`:

```bat
wsl bash -lc "cd ~/code/farmhub && ./deploy/vllm/vllm.sh up && ./deploy/vllm/vllm.sh wait"
```

`wait` blocks until `/health` answers, so the window closing means the model is actually
serving rather than merely started. Expect a few minutes on a warm checkpoint cache.

**Two rules.**

- **Never use them during an evaluation run.** The harness brings vLLM up and down itself
  and writes `deploy/vllm/.env` as it goes; interfering mid-run produces measurements that
  describe neither candidate. Check first: `docker ps` showing `farmhub-vllm` while you did
  not start it means a run owns it.
- **Never start vLLM before `.env` has been regenerated from the active profile** — see
  the next section. After any evaluation run that file holds the *last candidate measured*,
  so starting it blind serves the wrong model.

**What happens while it is off.** FarmHub keeps running, degraded: the `llm` module reports
unhealthy, the registry retries with backoff, and a satellite is told FarmHub is unavailable
instead of waiting for a timeout. Home Assistant's own intents and every automation keep
working — heating, pumps and irrigation never depended on this machine being awake (SPEC §2,
§3.2). Nothing needs restarting on the FarmHub side when vLLM comes back; the health probe
notices within `llm.health_interval_s`.

**After a Windows reboot**, nothing comes back by itself. Start Docker Desktop, then
`FarmHub ON.bat`.

### Switching the serving profile

`deploy/vllm/.env` decides what vLLM actually serves. **Do not hand-edit it.** It is
rendered from the chosen profile, because the two drifting apart is invisible otherwise:
the eval harness rewrites that file for every candidate it runs, so after a matrix run it
holds the *last candidate measured* rather than the one you chose.

```sh
uv run farmhub vllm env --profile primary          # print it, change nothing
uv run farmhub vllm env --profile primary --write  # render deploy/vllm/.env
./deploy/vllm/vllm.sh up && ./deploy/vllm/vllm.sh wait
```

Your own keys are read from the existing file and kept: `VLLM_IMAGE_TAG`, `HF_CACHE_DIR`,
`VLLM_PORT`, `VLLM_BIND_HOST`, `VLLM_WSL2_ENABLE_PIN_MEMORY`. Everything the profile
determines is replaced, so no flag survives from the model that was served before.

**The check that makes it stick.** `farmhub config check` and server startup compare the
file with the active profile and fail on a mismatch, naming the keys that differ and never
printing a value — that file sits beside secrets. A missing file, no profile, or
`llm.serving_env_file` unset means nothing was compared, and the output says so rather
than implying a pass.

```
$ uv run farmhub config check
error: deploy/vllm/.env does not match profile 'primary': VLLM_EXTRA_ARGS differ. …
        Regenerate with `farmhub vllm env --profile primary --write`
```

Why it matters concretely: candidate A is a multimodal checkpoint run text-only, and the
flag that keeps the vision tower out lives in `VLLM_EXTRA_ARGS`. Lose it and the tower
loads — weights grow, the KV cache shrinks, and the profile's `weights_gib` and
`kv_cache_gib` stop describing what is running, while every other check still passes
(that validation is arithmetic over declared numbers, not a probe).

It cannot stop you editing the file and running `vllm.sh up` by hand. It stops FarmHub
*serving* against a configuration its profile never measured.

### Checkpoint cache and disk

`HF_CACHE_DIR` must point inside the Linux filesystem (`~/.cache/huggingface`), never at
`/mnt/c` or `/mnt/d` — the Windows file bridge is far too slow for a 20 GB checkpoint and
vLLM mmaps the safetensors at load.

Budget against `C:`, not against what `df` reports inside WSL: that figure is the virtual
disk's ceiling, not free space. At 17–32 GB per checkpoint only a few candidates fit at
once, so prune between evaluation runs. Deleting files does not shrink the WSL virtual
disk by itself.

### WSL2: pinned memory

On the dev PC, vLLM refuses to start with "UVA is not available" unless
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` is set in `.env` (it is in `env.example`). vLLM turns
pinned memory off under WSL by default. On hub, leave it at `0`: native Linux ignores it.

### Ollama fallback

`deploy/ollama/compose.yaml` runs Ollama instead, for when vLLM will not start on a given
card or when developing against a small model. Switching is a config change only: point
`llm.base_url` at `http://127.0.0.1:11434/v1`. No Python changes (SPEC §2).

Neither vLLM nor Ollama has any authentication of its own, so both stay bound to
loopback on every host. FarmHub is the only client, and it carries the bearer token of
§3.6.

## The Home Assistant integration (M1, Q1)

`custom_components/farmhub/` is a conversation agent that forwards unmatched
utterances to FarmHub. See its own README for what it sends and why the built-in
integrations do not fit.

### Development

The real `ha` box does not exist yet, and when it does it must never be used for
development — it runs heating and pumps. Use the throwaway container instead:

```sh
docker compose -f deploy/dev/ha-compose.yaml up -d
# http://localhost:8123 -> create a throwaway account
# Settings > Devices & Services > Add Integration > FarmHub
#   URL:   http://host.docker.internal:8099
#   Token: the value of FARMHUB_API__TOKEN
# Settings > Voice assistants -> set the conversation agent to FarmHub
docker compose -f deploy/dev/ha-compose.yaml down -v   # -v discards the config
```

The integration is mounted read-only from the working tree, so editing it and
restarting the container is the whole edit loop.

### Networking, both halves

FarmHub runs on the WSL host; Home Assistant runs in a container. `127.0.0.1` inside
the container is the container's own loopback and will not reach FarmHub.

- **Container side:** `extra_hosts: host.docker.internal:host-gateway`, already in the
  compose file. The integration's URL is `http://host.docker.internal:8099`.
- **FarmHub side:** it must bind to something the container can reach.
  `config/farmhub.dev.toml` sets `api.bind_host = "0.0.0.0"` for exactly this, and
  startup logs `api_bind_not_loopback` every time, so the setting cannot travel to
  `hub` unnoticed.

```sh
export FARMHUB_API__TOKEN="$(openssl rand -hex 32)"
uv run farmhub --config config/farmhub.dev.toml serve
```

**If the container gets a refused connection**, it is almost always that FarmHub is
still on loopback: check the startup log for `api_bind_host`. On `hub` the bind address
is the LAN address and §3.6 requires it to be firewalled to the `ha` host —
`host.docker.internal` is a development-only arrangement.

### The Home Assistant version is pinned

The conversation-agent API changes between Home Assistant releases. `ha-compose.yaml`
pins the version the integration was written against; raise it deliberately and re-test
the agent when you do.

### The API token

It has no safe default, so `farmhub serve` refuses to start without one (§3.6, §7):

```sh
export FARMHUB_API__TOKEN="$(openssl rand -hex 32)"     # or
# api.token_file = "/run/secrets/farmhub-token"          # in farmhub.toml
```

Never in git, and never in a config file that is committed. `farmhub config check`
reports only whether a token is set, never its value, and the same is true of the
startup log.

## Home Assistant side (SPEC §3.4), filled in at M5

FarmHub cannot verify these from Python. Check each by hand before enabling any T2 tool:

- [ ] Every irrigation, watering and ventilation script enforces a maximum runtime inside
      the script itself (not in FarmHub). Record the value next to the tool's `max_runtime_s`.
- [ ] Heating and pump control exist only as deterministic HA automations and fixed intents.
- [ ] The HA user behind FarmHub's token: confirm whether "non-admin" actually restricts
      service calls. Assume it does not (docs/DECISIONS.md).
- [ ] Each valve and pump circuit has an independent hardware cutoff that does not route
      through Home Assistant (SPEC §10).
