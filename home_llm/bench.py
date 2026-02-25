"""llama-bench wrapper and benchmark utilities."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class LlamaBenchParams:
    """Parameters for a ``llama-bench`` invocation.

    All list-valued parameters (``batch_size``, ``ubatch_size``) are
    serialised as comma-separated values as expected by ``llama-bench``.
    """

    model: str
    n_prompt: int = 512
    n_gen: int = 128
    batch_size: List[int] = field(default_factory=lambda: [512])
    ubatch_size: List[int] = field(default_factory=lambda: [512])
    n_gpu_layers: int = 99
    n_cpu_moe: Optional[int] = None
    flash_attn: bool = True
    threads: Optional[int] = None
    repetitions: Optional[int] = None
    output: Optional[str] = None  # csv | json | md | sql
    extra_args: List[str] = field(default_factory=list)

    def to_args(self) -> List[str]:
        """Return the argument list for ``llama-bench`` (without the binary name)."""
        args: List[str] = []

        args += ["--model", self.model]
        args += ["--n-prompt", str(self.n_prompt)]
        args += ["--n-gen", str(self.n_gen)]
        args += ["--batch-size", ",".join(str(b) for b in self.batch_size)]
        args += ["--ubatch-size", ",".join(str(u) for u in self.ubatch_size)]
        args += ["--n-gpu-layers", str(self.n_gpu_layers)]

        if self.n_cpu_moe is not None:
            args += ["--n-cpu-moe", str(self.n_cpu_moe)]

        args += ["--flash-attn", "1" if self.flash_attn else "0"]

        if self.threads is not None:
            args += ["--threads", str(self.threads)]

        if self.repetitions is not None:
            args += ["--repetitions", str(self.repetitions)]

        if self.output is not None:
            args += ["--output", self.output]

        args.extend(self.extra_args)
        return args

    def to_command(self, llama_cpp_dir: str = "~/llama.cpp") -> str:
        """Return the full shell command string for ``llama-bench``."""
        binary = f"{llama_cpp_dir}/llama-bench"
        parts = [binary] + self.to_args()
        return " ".join(shlex.quote(p) for p in parts)


def params_from_dict(data: Dict) -> LlamaBenchParams:
    """Construct :class:`LlamaBenchParams` from a plain dictionary.

    List-valued parameters may be supplied as a list or as a
    comma-separated string (e.g. ``"128,256,512"``).
    """
    parsed: Dict = {}
    for key, value in data.items():
        if key in ("batch_size", "ubatch_size"):
            if isinstance(value, str):
                value = [int(x.strip()) for x in value.split(",")]
        parsed[key] = value
    return LlamaBenchParams(**parsed)
