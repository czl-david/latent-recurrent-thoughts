"""Filesystem locations. Everything outside the repository comes from environment variables
(see scripts/env.example.sh), resolved lazily so that importing never fails.

    LRT_DATA_ROOT      datasets root (read-only); task configs name files relative to it
    LRT_DECODER_PATH   directory of the frozen decoder (a Hugging Face Qwen3-8B checkpoint)
    LRT_OUT_ROOT       runs, evaluations and generated data (default: <repo>/outputs)
"""

import os


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def configs_dir() -> str:
    return os.path.join(project_root(), "configs")


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set (see scripts/env.example.sh)")
    return value


def data_root() -> str:
    return _required("LRT_DATA_ROOT")


def decoder_path() -> str:
    return _required("LRT_DECODER_PATH")


def out_root() -> str:
    return os.environ.get("LRT_OUT_ROOT") or os.path.join(project_root(), "outputs")


def generated_data_dir() -> str:
    """Data produced by lrt.distill; task configs refer to it as 'gen:<file>'."""
    return os.path.join(out_root(), "data_gen")
