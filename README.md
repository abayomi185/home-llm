# home-llm

> Tooling and utilities for running, testing, and interacting with LLMs running remotely in your homelab.

---

## Features

- **CLI tool** (`home-llm`) for managing and benchmarking remote LLM hosts
- **`llama-bench` wrapper** – build and dispatch parameterised benchmark commands over SSH
- **Shell scripts** for users who prefer pure-Bash SSH workflows (no Python required)
- **Configuration file** (`~/.config/home-llm/config.toml`) for storing host details and benchmark defaults

---

## Quick Start

### Installation

```bash
pip install -e .
```

Or install with dev dependencies:

```bash
pip install -e ".[dev]"
```

### Configuration

Create `~/.config/home-llm/config.toml`:

```toml
default_host = "gpu-server"

[hosts.gpu-server]
description = "Main GPU server in the homelab"
llama_cpp_dir = "/opt/llama.cpp"
models_dir = "/mnt/models"

[hosts.gpu-server.ssh]
host = "192.168.1.100"
user = "ubuntu"
port = 22
identity_file = "~/.ssh/homelab_ed25519"

[benchmark]
n_prompt = 512
n_gen = 128
n_gpu_layers = 99
flash_attn = true
batch_size = [512]
ubatch_size = [512]
```

### CLI Usage

```bash
# List configured hosts
home-llm hosts list

# Run llama-bench on the default host
home-llm bench run \
  --model ~/models/Qwen3.5-35B-A3B-MXFP4_MOE.gguf \
  --n-prompt 1024 \
  --n-gen 0 \
  --batch-size 128,256,512,1024 \
  --ubatch-size 128,256,512 \
  --n-gpu-layers 99 \
  --n-cpu-moe 38

# Print the command without running it
home-llm bench print \
  --model ~/models/Qwen3.5-35B-A3B-MXFP4_MOE.gguf \
  --batch-size 128,256,512

# Run an arbitrary remote command
home-llm ssh run -- nvidia-smi

# Open an interactive shell
home-llm ssh shell
```

### Shell Script Usage

For users who prefer shell scripts without a Python dependency, see [docs/ssh-remote.md](docs/ssh-remote.md).

---

## Repository Layout

```
home-llm/
├── home_llm/          # Python package
│   ├── cli.py         # Click-based CLI
│   ├── bench.py       # llama-bench parameter builder
│   ├── ssh.py         # SSH utilities (paramiko + system ssh)
│   └── config.py      # TOML configuration loader
├── scripts/
│   ├── llama-bench-remote.sh  # Bash wrapper for llama-bench over SSH
│   └── run-remote.sh          # Generic remote command runner
├── docs/
│   ├── benchmarking.md        # Benchmarking guide
│   └── ssh-remote.md          # SSH shell script guide
├── tests/             # pytest test suite
└── pyproject.toml
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Docs

- [Benchmarking Guide](docs/benchmarking.md)
- [SSH Remote Scripts](docs/ssh-remote.md)