#!/usr/bin/env bash
# llama-bench-remote.sh – Run llama-bench on a remote homelab host over SSH.
#
# Usage:
#   ./scripts/llama-bench-remote.sh \
#       --remote-host user@hostname \
#       --model path/to/model.gguf \
#       [llama-bench options...]
#
# The script forwards all options after --model directly to llama-bench on the
# remote host.  Connection settings (port, identity file) can be overridden
# via environment variables:
#
#   SSH_PORT         Remote SSH port (default: 22)
#   SSH_IDENTITY     Path to SSH identity file
#   LLAMA_CPP_DIR    Remote directory containing llama-bench (default: ~/llama.cpp)
#
# Example – reproduce the invocation from the project README:
#
#   SSH_IDENTITY=~/.ssh/homelab_ed25519 \
#   LLAMA_CPP_DIR=/opt/llama.cpp \
#   ./scripts/llama-bench-remote.sh \
#       --remote-host user@my-homelab-server \
#       --model ~/models/Qwen3.5-35B-A3B-MXFP4_MOE.gguf \
#       --n-prompt 1024 \
#       --n-gen 0 \
#       --batch-size 128,256,512,1024 \
#       --ubatch-size 128,256,512 \
#       --n-gpu-layers 99 \
#       --n-cpu-moe 38 \
#       --flash-attn 1

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (can be overridden via environment)
# ---------------------------------------------------------------------------
SSH_PORT="${SSH_PORT:-22}"
SSH_IDENTITY="${SSH_IDENTITY:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-~/llama.cpp}"

# ---------------------------------------------------------------------------
# Parse --remote-host and --model out of the argument list; pass the rest
# through to llama-bench unmodified.
# ---------------------------------------------------------------------------
REMOTE_HOST=""
MODEL=""
BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote-host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            BENCH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$REMOTE_HOST" ]]; then
    echo "ERROR: --remote-host is required." >&2
    echo "Usage: $0 --remote-host user@host --model path/to/model.gguf [llama-bench opts]" >&2
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build remote command
# ---------------------------------------------------------------------------
REMOTE_CMD="${LLAMA_CPP_DIR}/llama-bench --model ${MODEL}"
for arg in "${BENCH_ARGS[@]}"; do
    # Quote each argument to survive the SSH invocation
    REMOTE_CMD="${REMOTE_CMD} $(printf '%q' "$arg")"
done

# ---------------------------------------------------------------------------
# Build SSH command
# ---------------------------------------------------------------------------
SSH_CMD=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_IDENTITY" ]]; then
    SSH_CMD+=(-i "$SSH_IDENTITY")
fi
SSH_CMD+=("$REMOTE_HOST" "$REMOTE_CMD")

echo "==> Connecting to ${REMOTE_HOST} (port ${SSH_PORT})"
echo "==> Running: ${REMOTE_CMD}"
echo ""

exec "${SSH_CMD[@]}"
