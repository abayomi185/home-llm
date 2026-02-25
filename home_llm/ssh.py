"""SSH utilities for executing commands on remote homelab hosts."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import paramiko

from .config import HostConfig, SSHConfig


@dataclass
class CommandResult:
    """Result of a remote command execution."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def _build_client(ssh_cfg: SSHConfig) -> paramiko.SSHClient:
    """Create and connect an :class:`paramiko.SSHClient`."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs: dict = {
        "hostname": ssh_cfg.host,
        "port": ssh_cfg.port,
        "username": ssh_cfg.user,
        "timeout": ssh_cfg.connect_timeout,
        "look_for_keys": True,
        "allow_agent": True,
    }
    if ssh_cfg.identity_file:
        connect_kwargs["key_filename"] = ssh_cfg.identity_file

    client.connect(**connect_kwargs)
    return client


def run_remote(
    host: HostConfig,
    command: str,
    *,
    env: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> CommandResult:
    """Execute *command* on the remote *host* and return the result.

    Args:
        host: Target host configuration.
        command: Shell command string to run remotely.
        env: Optional mapping of environment variables to set.
        timeout: Optional execution timeout in seconds.

    Returns:
        :class:`CommandResult` with exit code, stdout and stderr.
    """
    client = _build_client(host.ssh)
    try:
        _, stdout_ch, stderr_ch = client.exec_command(
            command, environment=env, timeout=timeout
        )
        exit_code = stdout_ch.channel.recv_exit_status()
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout_ch.read().decode("utf-8", errors="replace"),
            stderr=stderr_ch.read().decode("utf-8", errors="replace"),
        )
    finally:
        client.close()


def stream_remote(
    host: HostConfig,
    command: str,
    *,
    env: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> Iterator[str]:
    """Execute *command* on the remote *host* and yield output lines.

    Yields stdout lines as they arrive.  The SSH connection is kept open
    for the duration of the generator and closed automatically on exit.
    """
    client = _build_client(host.ssh)
    try:
        _, stdout_ch, _ = client.exec_command(
            command, environment=env, timeout=timeout, get_pty=True
        )
        for line in stdout_ch:
            yield line.rstrip("\n")
        stdout_ch.channel.recv_exit_status()
    finally:
        client.close()


def build_ssh_command(host: HostConfig, remote_command: str) -> List[str]:
    """Return an ``ssh`` CLI argument list to run *remote_command*.

    This is useful when you want to hand off to the system ``ssh`` binary
    (e.g. for interactive sessions) instead of using paramiko.
    """
    ssh_cfg = host.ssh
    args: List[str] = ["ssh"]
    args += ["-p", str(ssh_cfg.port)]
    args += ["-o", "StrictHostKeyChecking=accept-new"]
    args += ["-o", f"ConnectTimeout={ssh_cfg.connect_timeout}"]
    if ssh_cfg.identity_file:
        args += ["-i", ssh_cfg.identity_file]
    args.append(f"{ssh_cfg.user}@{ssh_cfg.host}")
    args.append(remote_command)
    return args
