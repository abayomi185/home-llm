#!/usr/bin/env bash
# run-remote.sh – Run an arbitrary command on a remote homelab host over SSH.
#
# Usage:
#   ./scripts/run-remote.sh --remote-host user@hostname -- <command> [args...]
#
# Environment variables:
#   SSH_PORT         Remote SSH port (default: 22)
#   SSH_IDENTITY     Path to SSH identity file
#
# Example:
#   SSH_IDENTITY=~/.ssh/homelab_ed25519 \
#   ./scripts/run-remote.sh --remote-host user@my-server -- nvidia-smi

set -euo pipefail

SSH_PORT="${SSH_PORT:-22}"
SSH_IDENTITY="${SSH_IDENTITY:-}"

REMOTE_HOST=""
REMOTE_ARGS=()
SEPARATOR_SEEN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote-host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --)
            SEPARATOR_SEEN=true
            shift
            ;;
        *)
            if $SEPARATOR_SEEN; then
                REMOTE_ARGS+=("$1")
            else
                echo "ERROR: Unknown option: $1" >&2
                echo "Usage: $0 --remote-host user@host -- <command> [args...]" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$REMOTE_HOST" ]]; then
    echo "ERROR: --remote-host is required." >&2
    exit 1
fi

if [[ ${#REMOTE_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: No command specified after --." >&2
    exit 1
fi

REMOTE_CMD=""
for arg in "${REMOTE_ARGS[@]}"; do
    REMOTE_CMD="${REMOTE_CMD} $(printf '%q' "$arg")"
done
REMOTE_CMD="${REMOTE_CMD# }"  # strip leading space

SSH_CMD=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_IDENTITY" ]]; then
    SSH_CMD+=(-i "$SSH_IDENTITY")
fi
SSH_CMD+=("$REMOTE_HOST" "$REMOTE_CMD")

echo "==> Connecting to ${REMOTE_HOST}"
echo "==> Running: ${REMOTE_CMD}"
echo ""

exec "${SSH_CMD[@]}"
