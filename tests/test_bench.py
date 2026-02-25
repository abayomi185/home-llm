"""Tests for home_llm.bench – llama-bench parameter builder."""

import pytest
from home_llm.bench import LlamaBenchParams, params_from_dict


class TestLlamaBenchParamsToArgs:
    def test_required_model(self):
        p = LlamaBenchParams(model="/models/test.gguf")
        args = p.to_args()
        assert "--model" in args
        idx = args.index("--model")
        assert args[idx + 1] == "/models/test.gguf"

    def test_default_values(self):
        p = LlamaBenchParams(model="/m.gguf")
        args = p.to_args()
        assert args[args.index("--n-prompt") + 1] == "512"
        assert args[args.index("--n-gen") + 1] == "128"
        assert args[args.index("--n-gpu-layers") + 1] == "99"
        assert args[args.index("--flash-attn") + 1] == "1"

    def test_batch_sizes_comma_separated(self):
        p = LlamaBenchParams(model="/m.gguf", batch_size=[128, 256, 512])
        args = p.to_args()
        assert args[args.index("--batch-size") + 1] == "128,256,512"

    def test_ubatch_sizes_comma_separated(self):
        p = LlamaBenchParams(model="/m.gguf", ubatch_size=[128, 256])
        args = p.to_args()
        assert args[args.index("--ubatch-size") + 1] == "128,256"

    def test_n_cpu_moe_omitted_by_default(self):
        p = LlamaBenchParams(model="/m.gguf")
        assert "--n-cpu-moe" not in p.to_args()

    def test_n_cpu_moe_included_when_set(self):
        p = LlamaBenchParams(model="/m.gguf", n_cpu_moe=38)
        args = p.to_args()
        assert "--n-cpu-moe" in args
        assert args[args.index("--n-cpu-moe") + 1] == "38"

    def test_flash_attn_false(self):
        p = LlamaBenchParams(model="/m.gguf", flash_attn=False)
        args = p.to_args()
        assert args[args.index("--flash-attn") + 1] == "0"

    def test_output_format(self):
        p = LlamaBenchParams(model="/m.gguf", output="json")
        args = p.to_args()
        assert "--output" in args
        assert args[args.index("--output") + 1] == "json"

    def test_output_omitted_by_default(self):
        p = LlamaBenchParams(model="/m.gguf")
        assert "--output" not in p.to_args()

    def test_extra_args_appended(self):
        p = LlamaBenchParams(model="/m.gguf", extra_args=["--verbose", "--seed", "42"])
        args = p.to_args()
        assert args[-3:] == ["--verbose", "--seed", "42"]


class TestLlamaBenchParamsToCommand:
    def test_binary_path_uses_llama_cpp_dir(self):
        p = LlamaBenchParams(model="/m.gguf")
        cmd = p.to_command("/opt/llama.cpp")
        assert cmd.startswith("/opt/llama.cpp/llama-bench")

    def test_model_quoted_in_command(self):
        p = LlamaBenchParams(model="/path/to/my model.gguf")
        cmd = p.to_command("~/llama.cpp")
        # shlex.quote wraps paths with spaces in single quotes
        assert "'/path/to/my model.gguf'" in cmd

    def test_full_example(self):
        """Reproduce the example from the problem statement."""
        p = LlamaBenchParams(
            model="path/to/Qwen3.5-35B-A3B-MXFP4_MOE.gguf",
            n_prompt=1024,
            n_gen=0,
            batch_size=[128, 256, 512, 1024],
            ubatch_size=[128, 256, 512],
            n_gpu_layers=99,
            n_cpu_moe=38,
            flash_attn=True,
        )
        cmd = p.to_command("~/llama.cpp")
        assert "llama-bench" in cmd
        assert "--n-prompt" in cmd
        assert "1024" in cmd
        assert "--batch-size" in cmd
        assert "128,256,512,1024" in cmd
        assert "--n-cpu-moe" in cmd
        assert "38" in cmd
        assert "--flash-attn" in cmd
        assert "1" in cmd


class TestParamsFromDict:
    def test_basic(self):
        p = params_from_dict({"model": "/m.gguf", "n_prompt": 256})
        assert p.model == "/m.gguf"
        assert p.n_prompt == 256

    def test_batch_size_from_string(self):
        p = params_from_dict({"model": "/m.gguf", "batch_size": "128,256,512"})
        assert p.batch_size == [128, 256, 512]

    def test_ubatch_size_from_list(self):
        p = params_from_dict({"model": "/m.gguf", "ubatch_size": [128, 512]})
        assert p.ubatch_size == [128, 512]
