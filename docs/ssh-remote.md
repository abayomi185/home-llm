# SSH Remote Scripts

This page documents the pure-Bash scripts in `scripts/` that let you kick off
tasks on remote homelab hosts **without installing Python**.

---

## `llama-bench-remote.sh`

Runs `llama-bench` on a remote host over SSH.

### Usage

```bash
./scripts/llama-bench-remote.sh \
    --remote-host <user@host> \
    --model <remote-model-path> \
    [llama-bench options...]
```

All options after `--model` are forwarded verbatim to `llama-bench`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SSH_PORT` | `22` | SSH port on the remote host |
| `SSH_IDENTITY` | _(agent / default keys)_ | Path to SSH private key |
| `LLAMA_CPP_DIR` | `~/llama.cpp` | Directory containing `llama-bench` on the remote host |

### Example – reproduce the project README benchmark

```bash
SSH_IDENTITY=~/.ssh/homelab_ed25519 \
LLAMA_CPP_DIR=/opt/llama.cpp \
./scripts/llama-bench-remote.sh \
    --remote-host ubuntu@192.168.1.100 \
    --model ~/models/Qwen3.5-35B-A3B-MXFP4_MOE.gguf \
    --n-prompt 1024 \
    --n-gen 0 \
    --batch-size 128,256,512,1024 \
    --ubatch-size 128,256,512 \
    --n-gpu-layers 99 \
    --n-cpu-moe 38 \
    --flash-attn 1
```

The script echoes the remote command before running it so you can verify it
looks correct.

---

## `run-remote.sh`

Runs an **arbitrary** command on a remote host over SSH.

### Usage

```bash
./scripts/run-remote.sh --remote-host <user@host> -- <command> [args...]
```

The `--` separator is required.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SSH_PORT` | `22` | SSH port on the remote host |
| `SSH_IDENTITY` | _(agent / default keys)_ | Path to SSH private key |

### Examples

```bash
# Check GPU status
./scripts/run-remote.sh --remote-host ubuntu@192.168.1.100 -- nvidia-smi

# List available models
./scripts/run-remote.sh --remote-host ubuntu@192.168.1.100 -- ls ~/models

# With custom port and identity
SSH_PORT=2222 SSH_IDENTITY=~/.ssh/homelab_ed25519 \
./scripts/run-remote.sh --remote-host ubuntu@192.168.1.100 -- uname -a
```

---

## SSH Key Setup

If you haven't already set up passwordless SSH access to your homelab host:

```bash
# Generate a key (skip if you already have one)
ssh-keygen -t ed25519 -f ~/.ssh/homelab_ed25519 -C "homelab"

# Copy public key to the remote host
ssh-copy-id -i ~/.ssh/homelab_ed25519.pub ubuntu@192.168.1.100

# Test access
ssh -i ~/.ssh/homelab_ed25519 ubuntu@192.168.1.100 echo "OK"
```

Once configured, set `SSH_IDENTITY=~/.ssh/homelab_ed25519` (or add it to your
`~/.ssh/config`) so the scripts pick it up automatically.

---

## `~/.ssh/config` Shortcut

Adding a Host block to `~/.ssh/config` simplifies all invocations:

```
Host homelab
    HostName 192.168.1.100
    User ubuntu
    Port 22
    IdentityFile ~/.ssh/homelab_ed25519
```

Then use `homelab` as `--remote-host`:

```bash
./scripts/llama-bench-remote.sh \
    --remote-host homelab \
    --model ~/models/MyModel.gguf \
    --n-prompt 512 --n-gen 128
```
