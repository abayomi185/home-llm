# Benchmarking Guide

This guide covers how to run `llama-bench` on a remote homelab host using **home-llm**.

---

## Prerequisites

- `llama.cpp` compiled on the remote host (provides `llama-bench`)
- SSH access to the remote host
- A `~/.config/home-llm/config.toml` with at least one host entry (see [README](../README.md))

---

## Using the CLI

### Basic benchmark run

```bash
home-llm bench run \
  --model ~/models/Qwen3.5-35B-A3B-MXFP4_MOE.gguf \
  --n-prompt 1024 \
  --n-gen 0
```

This uses config defaults for batch sizes, GPU layers, and flash-attention.

### Full parameter example

```bash
home-llm bench run \
  --model ~/models/Qwen3.5-35B-A3B-MXFP4_MOE.gguf \
  --n-prompt 1024 \
  --n-gen 0 \
  --batch-size 128,256,512,1024 \
  --ubatch-size 128,256,512 \
  --n-gpu-layers 99 \
  --n-cpu-moe 38 \
  --flash-attn
```

This is equivalent to running on the remote host:

```bash
~/llama.cpp/llama-bench \
  --model ~/models/Qwen3.5-35B-A3B-MXFP4_MOE.gguf \
  --n-prompt 1024 \
  --n-gen 0 \
  --batch-size 128,256,512,1024 \
  --ubatch-size 128,256,512 \
  --n-gpu-layers 99 \
  --n-cpu-moe 38 \
  --flash-attn 1
```

### Preview without running

Use `--dry-run` (or the `bench print` sub-command) to inspect the command without executing it:

```bash
home-llm bench run --dry-run \
  --model ~/models/MyModel.gguf \
  --batch-size 128,512

# or equivalently:
home-llm bench print \
  --model ~/models/MyModel.gguf \
  --batch-size 128,512
```

### Targeting a specific host

```bash
home-llm bench run --host my-other-server \
  --model ~/models/MyModel.gguf
```

---

## Parameter Reference

| CLI Flag | llama-bench Flag | Default (config) | Description |
|---|---|---|---|
| `--model` | `--model` | _(required)_ | Path to GGUF model on the remote host |
| `--n-prompt` | `--n-prompt` | 512 | Number of prompt tokens to evaluate |
| `--n-gen` | `--n-gen` | 128 | Number of tokens to generate (0 = prompt-only) |
| `--batch-size` | `--batch-size` | 512 | Comma-separated logical batch sizes |
| `--ubatch-size` | `--ubatch-size` | 512 | Comma-separated micro-batch sizes |
| `--n-gpu-layers` | `--n-gpu-layers` | 99 | Number of layers to offload to GPU |
| `--n-cpu-moe` | `--n-cpu-moe` | _(unset)_ | Number of MoE expert layers kept on CPU |
| `--flash-attn` | `--flash-attn 1` | true | Enable flash attention |
| `--output` | `--output` | _(unset)_ | Output format: csv / json / md / sql |

---

## Storing Defaults in Config

You can set your preferred defaults in `~/.config/home-llm/config.toml` so you don't need to repeat them on every invocation:

```toml
[benchmark]
n_prompt = 1024
n_gen = 0
n_gpu_layers = 99
flash_attn = true
batch_size  = [128, 256, 512, 1024]
ubatch_size = [128, 256, 512]
```

CLI flags always take precedence over config defaults.

---

## Using the Shell Script

If you don't want the Python CLI, use the bundled shell script directly:

```bash
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

See [ssh-remote.md](ssh-remote.md) for full shell script documentation.
