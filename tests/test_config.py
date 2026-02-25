"""Tests for home_llm.config – configuration loading."""

import textwrap
from pathlib import Path

import pytest
from home_llm.config import (
    AppConfig,
    BenchmarkDefaults,
    HostConfig,
    SSHConfig,
    get_host,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_empty_config_when_file_missing(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert isinstance(cfg, AppConfig)
        assert cfg.hosts == {}

    def test_loads_host(self, tmp_path):
        path = _write_toml(
            tmp_path,
            """
            [hosts.gpu-server]
            llama_cpp_dir = "/opt/llama.cpp"
            models_dir = "/mnt/models"

            [hosts.gpu-server.ssh]
            host = "192.168.1.100"
            user = "ubuntu"
            port = 22
            """,
        )
        cfg = load_config(path)
        assert "gpu-server" in cfg.hosts
        host = cfg.hosts["gpu-server"]
        assert host.ssh.host == "192.168.1.100"
        assert host.ssh.user == "ubuntu"
        assert host.llama_cpp_dir == "/opt/llama.cpp"

    def test_default_host(self, tmp_path):
        path = _write_toml(
            tmp_path,
            """
            default_host = "gpu-server"

            [hosts.gpu-server.ssh]
            host = "10.0.0.1"
            user = "admin"
            """,
        )
        cfg = load_config(path)
        assert cfg.default_host == "gpu-server"

    def test_benchmark_defaults(self, tmp_path):
        path = _write_toml(
            tmp_path,
            """
            [benchmark]
            n_prompt = 1024
            n_gen = 0
            n_gpu_layers = 80
            flash_attn = false
            batch_size = [128, 256]
            ubatch_size = [128]
            """,
        )
        cfg = load_config(path)
        b = cfg.benchmark
        assert b.n_prompt == 1024
        assert b.n_gen == 0
        assert b.n_gpu_layers == 80
        assert b.flash_attn is False
        assert b.batch_size == [128, 256]
        assert b.ubatch_size == [128]

    def test_ssh_identity_file(self, tmp_path):
        path = _write_toml(
            tmp_path,
            """
            [hosts.box.ssh]
            host = "10.0.0.1"
            user = "ubuntu"
            identity_file = "~/.ssh/homelab_ed25519"
            """,
        )
        cfg = load_config(path)
        assert cfg.hosts["box"].ssh.identity_file == "~/.ssh/homelab_ed25519"


# ---------------------------------------------------------------------------
# get_host
# ---------------------------------------------------------------------------


class TestGetHost:
    def _make_config(self, default_host=None):
        host = HostConfig(
            name="box",
            ssh=SSHConfig(host="10.0.0.1"),
        )
        return AppConfig(hosts={"box": host}, default_host=default_host)

    def test_get_named_host(self):
        cfg = self._make_config()
        host = get_host(cfg, "box")
        assert host.name == "box"

    def test_get_default_host(self):
        cfg = self._make_config(default_host="box")
        host = get_host(cfg)
        assert host.name == "box"

    def test_raises_when_no_host_specified(self):
        cfg = self._make_config()
        with pytest.raises(KeyError, match="No host specified"):
            get_host(cfg)

    def test_raises_for_unknown_host(self):
        cfg = self._make_config()
        with pytest.raises(KeyError, match="unknown"):
            get_host(cfg, "unknown")
