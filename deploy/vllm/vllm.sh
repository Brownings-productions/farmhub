#!/usr/bin/env bash
# Start and stop the vLLM backend (SPEC §2, §8.2).
#
# Why this is manual rather than automatic: the dev PC is used for other things, and a
# 30 GB model must not hold the GPU all day. Docker Desktop on WSL2 has no systemd
# socket activation and vLLM has no idle shutdown, so anything "automatic" would be a
# bespoke sidecar. `down` frees the VRAM, which is the actual requirement.
#
# On hub the same compose file runs from deploy/systemd/farmhub-vllm.service instead.
#
# FarmHub tolerates vLLM being absent: the llm module starts degraded and the registry
# retries with backoff, so bringing this up or down under a running FarmHub is fine.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$HERE/compose.yaml"
ENV_FILE="$HERE/.env"

usage() {
    cat <<'USAGE'
usage: vllm.sh <command>

  up                 start vLLM in the background (does not wait for the model to load)
  down               stop it and free the VRAM
  restart            down, then up
  wait [timeout_s]   block until /health answers, default 900s
  status             container state, health, and GPU memory in use
  logs [-f]          container logs
  config             the fully resolved compose config, including the vllm command line
  resolve <repo>     print the current commit sha of a Hugging Face repo, to pin in .env

Copy env.example to .env first; it carries the model profile.
USAGE
}

require_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "error: $ENV_FILE not found. Copy env.example to .env and edit it." >&2
        exit 1
    fi
}

# --env-file is passed explicitly rather than relying on compose's project-directory
# lookup, so the behaviour does not depend on the working directory.
dc() {
    require_env_file
    docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

# Read one value out of .env without sourcing it, so a stray line in that file cannot
# execute anything here.
port() {
    require_env_file
    local p
    p="$(grep -E '^VLLM_PORT=' "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
    echo "${p:-8000}"
}

cmd_wait() {
    local timeout="${1:-900}"
    local url="http://127.0.0.1:$(port)/health"
    local deadline=$((SECONDS + timeout))
    echo "waiting for $url (timeout ${timeout}s; a cold checkpoint load takes minutes)"
    while ((SECONDS < deadline)); do
        if curl -fsS -o /dev/null --max-time 5 "$url" 2>/dev/null; then
            echo "vLLM is serving after ${SECONDS}s"
            return 0
        fi
        if [[ -z "$(dc ps -q vllm 2>/dev/null)" ]]; then
            echo "error: the container exited. Last lines:" >&2
            dc logs --tail 40 vllm >&2 || true
            return 1
        fi
        sleep 5
    done
    echo "error: vLLM did not answer within ${timeout}s" >&2
    dc logs --tail 40 vllm >&2 || true
    return 1
}

cmd_status() {
    dc ps
    echo
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU memory:"
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
    fi
}

# A revision must be a commit sha, not a tag (SPEC §2): one NVFP4 upload of
# Qwen3.6-35B-A3B was silently replaced with weights that loop, and a tag would have
# followed it. This asks the Hub API what main currently points at.
cmd_resolve() {
    local repo="${1:-}"
    if [[ -z "$repo" ]]; then
        echo "usage: vllm.sh resolve <org/repo>" >&2
        exit 1
    fi
    local sha
    sha="$(curl -fsS "https://huggingface.co/api/models/${repo}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')"
    echo "$sha"
    echo "# pin this in deploy/vllm/.env as VLLM_REVISION, and in the FarmHub profile" >&2
}

case "${1:-}" in
    up)      shift; dc up -d "$@"; echo "started. ./vllm.sh wait to block until it serves." ;;
    down)    shift; dc down "$@" ;;
    restart) dc down; dc up -d; echo "restarted. ./vllm.sh wait to block until it serves." ;;
    wait)    shift; cmd_wait "$@" ;;
    status)  cmd_status ;;
    logs)    shift; dc logs "$@" ;;
    config)  dc config ;;
    resolve) shift; cmd_resolve "$@" ;;
    -h|--help|help|"") usage ;;
    *)       echo "unknown command: $1" >&2; echo >&2; usage >&2; exit 1 ;;
esac
