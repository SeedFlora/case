"""Shared utilities: config loading, environment capture, hashing, logging.

Everything that ends up in a manifest or a report is produced here so that the
provenance of each number in the paper is traceable to one code path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 42, 2026)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_configs() -> Dict[str, Any]:
    """Load data + experiment configs as a single dict."""
    cfg_dir = REPO_ROOT / "configs"
    return {
        "data": load_yaml(cfg_dir / "data.yaml"),
        "experiment": load_yaml(cfg_dir / "experiment.yaml"),
    }


def modality_names(cfg: Dict[str, Any]) -> list[str]:
    return [m["name"] for m in cfg["data"]["modalities"]]


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, logfile: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------- #
# hashing / inventory
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path, chunk_mb: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_mb * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def capture_environment() -> Dict[str, Any]:
    """Snapshot everything needed to reproduce a run."""
    env: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        env["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        env["ram_total_gb"] = None

    pkgs: Dict[str, str] = {}
    for mod in [
        "numpy", "scipy", "pandas", "sklearn", "torch", "lightgbm",
        "pyarrow", "matplotlib", "statsmodels", "yaml",
    ]:
        try:
            m = __import__(mod)
            pkgs[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pkgs[mod] = "not-installed"
    env["packages"] = pkgs

    try:
        import torch

        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_cuda_version"] = torch.version.cuda
        env["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
    except Exception:
        env["torch_cuda_available"] = False

    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        env["git_commit"] = None
    return env


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# json helpers
# --------------------------------------------------------------------------- #
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=NpEncoder)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Timer:
    label: str = ""
    t0: float = field(default_factory=time.time)

    def stop(self) -> float:
        return time.time() - self.t0
