"""Tests for home_llm.ssh – SSH utilities."""

import pytest
from home_llm.config import HostConfig, SSHConfig
from home_llm.ssh import build_ssh_command


def _host(
    hostname="10.0.0.1",
    user="ubuntu",
    port=22,
    identity_file=None,
    connect_timeout=10,
) -> HostConfig:
    return HostConfig(
        name="test",
        ssh=SSHConfig(
            host=hostname,
            user=user,
            port=port,
            identity_file=identity_file,
            connect_timeout=connect_timeout,
        ),
    )


class TestBuildSshCommand:
    def test_basic_command(self):
        args = build_ssh_command(_host(), "echo hello")
        assert args[0] == "ssh"
        assert "-p" in args
        assert "22" in args
        assert "ubuntu@10.0.0.1" in args
        assert args[-1] == "echo hello"

    def test_identity_file_included(self):
        args = build_ssh_command(_host(identity_file="~/.ssh/id_ed25519"), "ls")
        assert "-i" in args
        idx = args.index("-i")
        assert args[idx + 1] == "~/.ssh/id_ed25519"

    def test_identity_file_omitted_when_not_set(self):
        args = build_ssh_command(_host(), "ls")
        assert "-i" not in args

    def test_custom_port(self):
        args = build_ssh_command(_host(port=2222), "ls")
        idx = args.index("-p")
        assert args[idx + 1] == "2222"

    def test_connect_timeout_in_options(self):
        args = build_ssh_command(_host(connect_timeout=30), "ls")
        options_str = " ".join(args)
        assert "ConnectTimeout=30" in options_str

    def test_user_at_host_format(self):
        args = build_ssh_command(_host(hostname="192.168.1.5", user="admin"), "ls")
        assert "admin@192.168.1.5" in args
