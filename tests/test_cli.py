"""Tests for home_llm.cli – Click CLI."""

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from home_llm.cli import cli


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        textwrap.dedent(
            """
            default_host = "gpu-server"

            [hosts.gpu-server]
            llama_cpp_dir = "/opt/llama.cpp"
            models_dir = "/mnt/models"
            description = "Test GPU server"

            [hosts.gpu-server.ssh]
            host = "192.168.1.100"
            user = "ubuntu"
            port = 22

            [benchmark]
            n_prompt = 512
            n_gen = 0
            n_gpu_layers = 99
            flash_attn = true
            batch_size = [512]
            ubatch_size = [512]
            """
        )
    )
    return cfg


class TestHostsList:
    def test_lists_hosts(self, config_file):
        runner = CliRunner()
        result = runner.invoke(cli, ["-c", str(config_file), "hosts", "list"])
        assert result.exit_code == 0
        assert "gpu-server" in result.output
        assert "192.168.1.100" in result.output

    def test_no_hosts_message(self, tmp_path):
        empty = tmp_path / "empty.toml"
        empty.write_text("")
        runner = CliRunner()
        result = runner.invoke(cli, ["-c", str(empty), "hosts", "list"])
        assert result.exit_code == 0
        assert "No hosts configured" in result.output


class TestBenchPrint:
    def test_prints_command(self, config_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "-c", str(config_file),
                "bench", "print",
                "--model", "/mnt/models/MyModel.gguf",
                "--batch-size", "128,256",
            ],
        )
        assert result.exit_code == 0
        assert "llama-bench" in result.output
        assert "/mnt/models/MyModel.gguf" in result.output
        assert "128,256" in result.output

    def test_uses_host_llama_cpp_dir(self, config_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "-c", str(config_file),
                "bench", "print",
                "--model", "/m.gguf",
            ],
        )
        assert result.exit_code == 0
        assert "/opt/llama.cpp/llama-bench" in result.output

    def test_n_cpu_moe_included(self, config_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "-c", str(config_file),
                "bench", "print",
                "--model", "/m.gguf",
                "--n-cpu-moe", "38",
            ],
        )
        assert result.exit_code == 0
        assert "--n-cpu-moe" in result.output
        assert "38" in result.output


class TestBenchRunDryRun:
    def test_dry_run_prints_command_without_executing(self, config_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "-c", str(config_file),
                "bench", "run",
                "--dry-run",
                "--model", "/mnt/models/MyModel.gguf",
            ],
        )
        assert result.exit_code == 0
        assert "llama-bench" in result.output
        # Verify host info is shown
        assert "gpu-server" in result.output

    def test_dry_run_full_example(self, config_file):
        """Reproduce the problem-statement benchmark via --dry-run."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "-c", str(config_file),
                "bench", "run",
                "--dry-run",
                "--model", "path/to/Qwen3.5-35B-A3B-MXFP4_MOE.gguf",
                "--n-prompt", "1024",
                "--n-gen", "0",
                "--batch-size", "128,256,512,1024",
                "--ubatch-size", "128,256,512",
                "--n-gpu-layers", "99",
                "--n-cpu-moe", "38",
                "--flash-attn",
            ],
        )
        assert result.exit_code == 0
        assert "1024" in result.output
        assert "128,256,512,1024" in result.output
        assert "--n-cpu-moe" in result.output

    def test_unknown_host_exits_nonzero(self, config_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "-c", str(config_file),
                "bench", "run",
                "--host", "does-not-exist",
                "--model", "/m.gguf",
            ],
        )
        assert result.exit_code != 0
