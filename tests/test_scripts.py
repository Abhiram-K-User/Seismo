"""Entry-point scripts parse arguments, load the config, and refuse to pretend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PLACEHOLDER_SCRIPTS = ["train.py", "evaluate.py"]
ALL_SCRIPTS = [*PLACEHOLDER_SCRIPTS, "download_data.py", "inspect_data.py", "prepare_data.py"]


def _run(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo_root / "src"), env.get("PYTHONPATH")])
    )
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("script", ALL_SCRIPTS)
def test_script_help(repo_root: Path, script: str) -> None:
    result = _run(repo_root, f"scripts/{script}", "--help")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", PLACEHOLDER_SCRIPTS)
def test_script_reports_not_implemented(repo_root: Path, script: str) -> None:
    result = _run(repo_root, f"scripts/{script}", "--config", "configs/config.yaml")
    assert result.returncode == 1
    assert "not implemented yet" in result.stderr


def test_inspect_data_with_data_root_override(repo_root: Path, toy_data_root: Path) -> None:
    result = _run(repo_root, "scripts/inspect_data.py", "--data-root", str(toy_data_root))
    assert result.returncode == 0, result.stderr
    manifest = json.loads((toy_data_root / "processed" / "dataset_manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["num_events"] == 6 and manifest["num_traces"] == 11
    assert manifest["paths_relative_to"] == "data_root"
    assert toy_data_root.parent.name not in json.dumps(manifest)   # no absolute paths leak
    assert (toy_data_root / "interim" / "pnw_quality_report.json").exists()
    assert not (repo_root / "data" / "processed" / "toy").exists()
