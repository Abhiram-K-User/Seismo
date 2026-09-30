"""Resolution of data and output locations — never hard-coded to one machine.

The same code must run on a laptop, on Google Colab with Google Drive
mounted, or against any other storage location. Locations are therefore
resolved at run time with this precedence (first match wins):

1. an explicit argument (e.g. a ``--data-root`` command-line flag),
2. an environment variable (``SEISMIC_DATA_ROOT`` / ``SEISMIC_OUTPUT_ROOT``),
3. the value in ``configs/config.yaml`` (``paths.data_root`` / ``paths.output_root``),
4. the defaults ``data`` / ``outputs``.

Relative paths are resolved against the *project root* (the directory that
contains ``configs/``), never against the current working directory, so
scripts and notebooks behave identically wherever they are launched from.
``~`` and ``$VARS`` inside configured values are expanded.

Examples::

    # Local default: <project>/data
    paths = resolve_paths(config, project_root=project_root_from_config(cfg_path))

    # Colab with Drive mounted (either way works):
    #   os.environ["SEISMIC_DATA_ROOT"] = "/content/drive/MyDrive/seismic-data"
    paths = resolve_paths(config, project_root=root, data_root="/content/drive/MyDrive/seismic-data")

Resolved absolute paths are machine-specific: never write them into
configuration snapshots, manifests or committed files.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_DATA_ROOT = "SEISMIC_DATA_ROOT"
ENV_OUTPUT_ROOT = "SEISMIC_OUTPUT_ROOT"
DEFAULT_DATA_ROOT = "data"
DEFAULT_OUTPUT_ROOT = "outputs"


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved, absolute storage locations for one run."""

    project_root: Path
    data_root: Path
    output_root: Path

    @property
    def raw(self) -> Path:
        """Unmodified source data (immutable)."""
        return self.data_root / "raw"

    @property
    def interim(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed(self) -> Path:
        return self.data_root / "processed"

    @property
    def checkpoints(self) -> Path:
        return self.output_root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.output_root / "logs"

    @property
    def figures(self) -> Path:
        return self.output_root / "figures"

    @property
    def results(self) -> Path:
        return self.output_root / "results"

    def dataset_raw_dir(self, data_config: Mapping[str, Any]) -> Path:
        """Raw directory of the configured dataset: ``<raw>/<dataset.raw_subdir>``."""
        subdir = (data_config.get("dataset") or {}).get("raw_subdir")
        if not subdir:
            raise ValueError("data.dataset.raw_subdir is not configured.")
        if Path(subdir).is_absolute():
            raise ValueError("data.dataset.raw_subdir must be relative to the raw data root.")
        return self.raw / subdir


def project_root_from_config(config_path: str | Path) -> Path:
    """Project root = parent of the directory holding the entry config file."""
    return Path(config_path).resolve().parent.parent


def _resolve(value: str | Path, project_root: Path) -> Path:
    expanded = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not expanded.is_absolute():
        expanded = project_root / expanded
    return expanded.resolve()


def _pick(explicit: str | Path | None, env_name: str, configured: Any, default: str,
          environ: Mapping[str, str]) -> str | Path:
    if explicit not in (None, ""):
        return explicit  # type: ignore[return-value]
    if environ.get(env_name):
        return environ[env_name]
    if configured not in (None, ""):
        return configured
    return default


def resolve_paths(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProjectPaths:
    """Resolve data/output roots from arguments, environment and configuration.

    Args:
        config: Resolved configuration (``paths`` section is read if present).
        project_root: Directory containing ``configs/``; base for relative paths.
        data_root: Explicit override (e.g. from ``--data-root``).
        output_root: Explicit override (e.g. from ``--output-root``).
        environ: Environment mapping (defaults to ``os.environ``; injectable for tests).
    """
    env = os.environ if environ is None else environ
    root = Path(project_root).resolve()
    paths_cfg = config.get("paths") or {}
    data = _pick(data_root, ENV_DATA_ROOT, paths_cfg.get("data_root"), DEFAULT_DATA_ROOT, env)
    output = _pick(output_root, ENV_OUTPUT_ROOT, paths_cfg.get("output_root"),
                   DEFAULT_OUTPUT_ROOT, env)
    return ProjectPaths(
        project_root=root,
        data_root=_resolve(data, root),
        output_root=_resolve(output, root),
    )


def relative_to_root(path: str | Path, root: str | Path) -> str:
    """Portable POSIX-style path of ``path`` relative to ``root`` (for manifests).

    Raises:
        ValueError: If ``path`` is not inside ``root``.
    """
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
