"""Command-line interface for home-llm."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

from .bench import LlamaBenchParams
from .config import load_config, get_host, AppConfig, _config_path

console = Console()
err_console = Console(stderr=True)


def _load(config_file: Optional[str]) -> AppConfig:
    path = Path(config_file) if config_file else None
    return load_config(path)


@click.group()
@click.version_option()
@click.option(
    "--config",
    "-c",
    "config_file",
    default=None,
    envvar="HOME_LLM_CONFIG",
    metavar="PATH",
    help="Path to config file (default: ~/.config/home-llm/config.toml).",
)
@click.pass_context
def cli(ctx: click.Context, config_file: Optional[str]) -> None:
    """home-llm – homelab LLM tooling CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file


# ---------------------------------------------------------------------------
# hosts sub-command
# ---------------------------------------------------------------------------


@cli.group()
def hosts() -> None:
    """Manage and inspect configured remote hosts."""


@hosts.command("list")
@click.pass_context
def hosts_list(ctx: click.Context) -> None:
    """List all configured hosts."""
    cfg = _load(ctx.obj["config_file"])

    if not cfg.hosts:
        console.print("[yellow]No hosts configured.[/yellow]")
        console.print(f"Config path: {_config_path()}")
        return

    table = Table(title="Configured Hosts")
    table.add_column("Name", style="cyan")
    table.add_column("Host")
    table.add_column("User")
    table.add_column("Port")
    table.add_column("Description")

    for name, host in cfg.hosts.items():
        marker = " [green](default)[/green]" if name == cfg.default_host else ""
        table.add_row(
            name + marker,
            host.ssh.host,
            host.ssh.user,
            str(host.ssh.port),
            host.description or "",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# bench sub-command
# ---------------------------------------------------------------------------


@cli.group()
def bench() -> None:
    """Benchmarking utilities (llama-bench wrapper)."""


@bench.command("run")
@click.option("--host", "-H", "host_name", default=None, help="Target host name.")
@click.option("--model", "-m", required=True, help="Path to the model file (on the remote host).")
@click.option("--n-prompt", default=None, type=int, help="Number of prompt tokens.")
@click.option("--n-gen", default=None, type=int, help="Number of generated tokens.")
@click.option(
    "--batch-size",
    default=None,
    help="Comma-separated batch sizes, e.g. 128,256,512.",
)
@click.option(
    "--ubatch-size",
    default=None,
    help="Comma-separated ubatch sizes, e.g. 128,256,512.",
)
@click.option("--n-gpu-layers", default=None, type=int, help="Number of GPU layers.")
@click.option("--n-cpu-moe", default=None, type=int, help="Number of CPU MoE layers.")
@click.option(
    "--flash-attn/--no-flash-attn",
    default=None,
    help="Enable or disable flash attention.",
)
@click.option("--output", default=None, type=click.Choice(["csv", "json", "md", "sql"]), help="Output format.")
@click.option("--dry-run", is_flag=True, help="Print the command without running it.")
@click.pass_context
def bench_run(
    ctx: click.Context,
    host_name: Optional[str],
    model: str,
    n_prompt: Optional[int],
    n_gen: Optional[int],
    batch_size: Optional[str],
    ubatch_size: Optional[str],
    n_gpu_layers: Optional[int],
    n_cpu_moe: Optional[int],
    flash_attn: Optional[bool],
    output: Optional[str],
    dry_run: bool,
) -> None:
    """Run llama-bench on a remote host."""
    cfg = _load(ctx.obj["config_file"])

    try:
        host = get_host(cfg, host_name)
    except KeyError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # Build params, applying config defaults where CLI flags are not set
    defaults = cfg.benchmark
    params = LlamaBenchParams(
        model=model,
        n_prompt=n_prompt if n_prompt is not None else defaults.n_prompt,
        n_gen=n_gen if n_gen is not None else defaults.n_gen,
        batch_size=(
            [int(x) for x in batch_size.split(",")]
            if batch_size
            else defaults.batch_size
        ),
        ubatch_size=(
            [int(x) for x in ubatch_size.split(",")]
            if ubatch_size
            else defaults.ubatch_size
        ),
        n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else defaults.n_gpu_layers,
        n_cpu_moe=n_cpu_moe,
        flash_attn=flash_attn if flash_attn is not None else defaults.flash_attn,
        output=output,
    )

    command = params.to_command(host.llama_cpp_dir)
    console.print(f"[bold]Host:[/bold] {host.name} ({host.ssh.user}@{host.ssh.host})")
    console.print(f"[bold]Command:[/bold] {command}")

    if dry_run:
        return

    # Stream output via system ssh for interactive experience
    from .ssh import build_ssh_command

    ssh_args = build_ssh_command(host, command)
    console.print(f"\n[dim]Running via: {' '.join(shlex.quote(a) for a in ssh_args[:-1])} ...[/dim]\n")

    try:
        result = subprocess.run(ssh_args)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


@bench.command("print")
@click.option("--host", "-H", "host_name", default=None, help="Target host name.")
@click.option("--model", "-m", required=True, help="Path to the model file.")
@click.option("--n-prompt", default=None, type=int)
@click.option("--n-gen", default=None, type=int)
@click.option("--batch-size", default=None)
@click.option("--ubatch-size", default=None)
@click.option("--n-gpu-layers", default=None, type=int)
@click.option("--n-cpu-moe", default=None, type=int)
@click.option("--flash-attn/--no-flash-attn", default=None)
@click.pass_context
def bench_print(
    ctx: click.Context,
    host_name: Optional[str],
    model: str,
    n_prompt: Optional[int],
    n_gen: Optional[int],
    batch_size: Optional[str],
    ubatch_size: Optional[str],
    n_gpu_layers: Optional[int],
    n_cpu_moe: Optional[int],
    flash_attn: Optional[bool],
) -> None:
    """Print the llama-bench command that would be run (without executing)."""
    cfg = _load(ctx.obj["config_file"])
    defaults = cfg.benchmark
    llama_cpp_dir = "~/llama.cpp"

    if host_name or cfg.default_host:
        try:
            host = get_host(cfg, host_name)
            llama_cpp_dir = host.llama_cpp_dir
        except KeyError:
            pass

    params = LlamaBenchParams(
        model=model,
        n_prompt=n_prompt if n_prompt is not None else defaults.n_prompt,
        n_gen=n_gen if n_gen is not None else defaults.n_gen,
        batch_size=(
            [int(x) for x in batch_size.split(",")]
            if batch_size
            else defaults.batch_size
        ),
        ubatch_size=(
            [int(x) for x in ubatch_size.split(",")]
            if ubatch_size
            else defaults.ubatch_size
        ),
        n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else defaults.n_gpu_layers,
        n_cpu_moe=n_cpu_moe,
        flash_attn=flash_attn if flash_attn is not None else defaults.flash_attn,
    )

    console.print(params.to_command(llama_cpp_dir))


# ---------------------------------------------------------------------------
# ssh sub-command
# ---------------------------------------------------------------------------


@cli.group()
def ssh() -> None:
    """SSH utilities for remote host interaction."""


@ssh.command("run")
@click.option("--host", "-H", "host_name", default=None, help="Target host name.")
@click.argument("command", nargs=-1, required=True)
@click.pass_context
def ssh_run(
    ctx: click.Context,
    host_name: Optional[str],
    command: tuple,
) -> None:
    """Run an arbitrary COMMAND on the remote host via SSH."""
    cfg = _load(ctx.obj["config_file"])

    try:
        host = get_host(cfg, host_name)
    except KeyError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    from .ssh import build_ssh_command

    remote_cmd = " ".join(shlex.quote(c) for c in command)
    ssh_args = build_ssh_command(host, remote_cmd)

    try:
        result = subprocess.run(ssh_args)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)


@ssh.command("shell")
@click.option("--host", "-H", "host_name", default=None, help="Target host name.")
@click.pass_context
def ssh_shell(ctx: click.Context, host_name: Optional[str]) -> None:
    """Open an interactive SSH shell on the remote host."""
    cfg = _load(ctx.obj["config_file"])

    try:
        host = get_host(cfg, host_name)
    except KeyError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    ssh_cfg = host.ssh
    ssh_args = ["ssh"]
    ssh_args += ["-p", str(ssh_cfg.port)]
    if ssh_cfg.identity_file:
        ssh_args += ["-i", ssh_cfg.identity_file]
    ssh_args.append(f"{ssh_cfg.user}@{ssh_cfg.host}")

    try:
        result = subprocess.run(ssh_args)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
