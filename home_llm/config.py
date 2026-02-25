"""Configuration management for home-llm."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


CONFIG_FILE_ENV = "HOME_LLM_CONFIG"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "home-llm" / "config.toml"


class SSHConfig(BaseModel):
    """SSH connection settings for a remote host."""

    host: str
    port: int = 22
    user: str = Field(default_factory=lambda: os.environ.get("USER", "ubuntu"))
    identity_file: Optional[str] = None
    connect_timeout: int = 10


class HostConfig(BaseModel):
    """Configuration for a remote LLM host."""

    name: str
    ssh: SSHConfig
    llama_cpp_dir: str = "~/llama.cpp"
    models_dir: str = "~/models"
    description: Optional[str] = None


class BenchmarkDefaults(BaseModel):
    """Default parameters used when running llama-bench."""

    n_prompt: int = 512
    n_gen: int = 128
    n_gpu_layers: int = 99
    flash_attn: bool = True
    batch_size: List[int] = Field(default_factory=lambda: [512])
    ubatch_size: List[int] = Field(default_factory=lambda: [512])


class AppConfig(BaseModel):
    """Top-level application configuration."""

    hosts: Dict[str, HostConfig] = Field(default_factory=dict)
    benchmark: BenchmarkDefaults = Field(default_factory=BenchmarkDefaults)
    default_host: Optional[str] = None


def _config_path() -> Path:
    """Return the config file path (env override or default)."""
    env = os.environ.get(CONFIG_FILE_ENV)
    return Path(env) if env else DEFAULT_CONFIG_PATH


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from a TOML file.

    Returns an empty :class:`AppConfig` when no config file exists.
    """
    config_path = path or _config_path()
    if not config_path.exists():
        return AppConfig()

    with open(config_path, "rb") as f:
        raw: Dict[str, Any] = tomllib.load(f)

    # Normalise flat host entries from TOML into HostConfig objects
    hosts_raw = raw.pop("hosts", {})
    hosts: Dict[str, HostConfig] = {}
    for name, data in hosts_raw.items():
        ssh_data = data.pop("ssh", {})
        if "host" not in ssh_data:
            ssh_data["host"] = name
        hosts[name] = HostConfig(name=name, ssh=SSHConfig(**ssh_data), **data)

    benchmark_raw = raw.pop("benchmark", {})
    benchmark = BenchmarkDefaults(**benchmark_raw)

    return AppConfig(hosts=hosts, benchmark=benchmark, **raw)


def get_host(config: AppConfig, name: Optional[str] = None) -> HostConfig:
    """Return the named host config, falling back to *default_host*.

    Raises :class:`KeyError` when the host cannot be found.
    """
    target = name or config.default_host
    if target is None:
        raise KeyError(
            "No host specified and no default_host set in configuration."
        )
    if target not in config.hosts:
        available = list(config.hosts.keys())
        raise KeyError(
            f"Host {target!r} not found. Available hosts: {available}"
        )
    return config.hosts[target]
