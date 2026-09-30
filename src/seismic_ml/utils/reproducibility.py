"""Reproducibility helpers: seeding, environment capture, experiment IDs.

See PROJECT_CONTEXT.md §13 for what every experiment must record.
"""

from __future__ import annotations

import os
import platform
import random
import re
import sys
import uuid
from datetime import datetime, timezone

import numpy as np
import torch

import seismic_ml

_MAX_SEED = 2**32 - 1  # NumPy's legacy seeding limit


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    Args:
        seed: Non-negative integer below 2**32.
        deterministic: If True, also request deterministic PyTorch algorithms
            (slower; ops without a deterministic implementation emit a warning).
            This changes global PyTorch state for the rest of the process.

    Note:
        ``PYTHONHASHSEED`` is exported for child processes (e.g. DataLoader
        workers on Windows); it cannot change hashing in the running process.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {_MAX_SEED}], got {seed!r}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_environment_info() -> dict[str, str]:
    """Return software/hardware versions to store alongside experiment results."""
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seismic_ml": seismic_ml.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": str(cuda_available),
        "cuda_version": str(torch.version.cuda) if cuda_available else "n/a",
    }


def generate_experiment_id(name: str | None = None, *, now: datetime | None = None) -> str:
    """Create a unique, chronologically sortable experiment identifier.

    Format: ``<UTC timestamp>_<slug>_<6 hex chars>``, e.g.
    ``20260918T101500Z_cnn-baseline_a1b2c3``.

    Args:
        name: Optional human-readable name; non ``[A-Za-z0-9_-]`` characters
            are replaced by ``-``. Defaults to ``"exp"``.
        now: Timestamp to use (defaults to the current UTC time).
    """
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") if name else ""
    return f"{timestamp}_{slug or 'exp'}_{uuid.uuid4().hex[:6]}"
