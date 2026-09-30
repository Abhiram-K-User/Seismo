"""Storage locations come from arguments / environment / config — never hard-coded."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from seismic_ml.paths import (
    ENV_DATA_ROOT,
    ENV_OUTPUT_ROOT,
    project_root_from_config,
    relative_to_root,
    resolve_paths,
)

CFG = {"paths": {"data_root": "data", "output_root": "outputs"}}


def test_project_root_is_parent_of_configs(config_path: Path, repo_root: Path) -> None:
    assert project_root_from_config(config_path) == repo_root.resolve()


def test_default_resolves_relative_to_project_root(tmp_path: Path) -> None:
    paths = resolve_paths(CFG, project_root=tmp_path, environ={})
    assert paths.data_root == (tmp_path / "data").resolve()
    assert paths.raw == paths.data_root / "raw"
    assert paths.processed == paths.data_root / "processed"
    assert paths.results == (tmp_path / "outputs" / "results").resolve()


def test_missing_paths_section_uses_defaults(tmp_path: Path) -> None:
    paths = resolve_paths({}, project_root=tmp_path, environ={})
    assert paths.data_root.name == "data" and paths.output_root.name == "outputs"


def test_precedence_argument_over_env_over_config(tmp_path: Path) -> None:
    env = {ENV_DATA_ROOT: str(tmp_path / "from_env"), ENV_OUTPUT_ROOT: str(tmp_path / "out_env")}
    from_env = resolve_paths(CFG, project_root=tmp_path, environ=env)
    assert from_env.data_root == (tmp_path / "from_env").resolve()
    assert from_env.output_root == (tmp_path / "out_env").resolve()
    from_arg = resolve_paths(CFG, project_root=tmp_path, environ=env,
                             data_root=tmp_path / "from_arg")
    assert from_arg.data_root == (tmp_path / "from_arg").resolve()


def test_colab_style_absolute_path(tmp_path: Path) -> None:
    drive = tmp_path / "content" / "drive" / "MyDrive" / "seismic-data"
    paths = resolve_paths(CFG, project_root=tmp_path / "repo", data_root=str(drive), environ={})
    assert paths.data_root == drive.resolve()


def test_env_vars_and_home_are_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_STORAGE", str(tmp_path / "storage"))
    cfg = {"paths": {"data_root": "$MY_STORAGE/seismic"}}
    assert resolve_paths(cfg, project_root=tmp_path, environ={}).data_root == \
        (tmp_path / "storage" / "seismic").resolve()
    home = resolve_paths({"paths": {"data_root": "~/seismic"}}, project_root=tmp_path, environ={})
    assert home.data_root == (Path.home() / "seismic").resolve()


def test_empty_env_var_is_ignored(tmp_path: Path) -> None:
    paths = resolve_paths(CFG, project_root=tmp_path, environ={ENV_DATA_ROOT: ""})
    assert paths.data_root == (tmp_path / "data").resolve()


def test_dataset_raw_dir(tmp_path: Path, data_config: dict) -> None:
    paths = resolve_paths(CFG, project_root=tmp_path, environ={})
    assert paths.dataset_raw_dir(data_config) == paths.raw / "pnw"
    with pytest.raises(ValueError):
        paths.dataset_raw_dir({"dataset": {}})
    with pytest.raises(ValueError):
        paths.dataset_raw_dir({"dataset": {"raw_subdir": str(tmp_path.resolve())}})


def test_relative_to_root_is_portable(tmp_path: Path) -> None:
    assert relative_to_root(tmp_path / "raw" / "pnw" / "a.csv", tmp_path) == "raw/pnw/a.csv"
    with pytest.raises(ValueError):
        relative_to_root(tmp_path.parent, tmp_path)


# --- repository hygiene ------------------------------------------------------------

_MACHINE_PATH = re.compile(r"([A-Za-z]:[\\/]+Users[\\/]+\w|/c/Users/\w|/home/[a-z]\w*/|/Users/[A-Za-z]\w*/)")
_TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".ipynb", ".json", ".txt", ".cfg"}
_SKIP_DIRS = {".venv", "data", "outputs", ".git", "__pycache__", ".pytest_cache", "build", "dist"}


def _text_files(repo_root: Path):
    for path in repo_root.rglob("*"):
        rel = path.relative_to(repo_root)
        if (path.is_file() and path.suffix in _TEXT_SUFFIXES
                and not any(part in _SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts)):
            yield path, rel


def test_no_machine_specific_paths_committed(repo_root: Path) -> None:
    offenders = []
    for path, rel in _text_files(repo_root):
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if _MACHINE_PATH.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    assert not offenders, "machine-specific paths found:\n" + "\n".join(offenders)


# What UTF-8 multi-byte text looks like after being decoded as cp1252 and re-encoded:
# U+00E2 U+20AC (the start of a mangled dash/quote), or U+00C2/U+00C3 followed by a
# Latin-1 continuation character. Written with escapes so this file never matches itself.
_MOJIBAKE = re.compile("\\u00e2\\u20ac|[\\u00c2\\u00c3][\\u0080-\\u00bf]")


def test_text_files_are_clean_utf8(repo_root: Path) -> None:
    """Regression: a cp1252 round trip once double-encoded the notebooks' non-ASCII text."""
    offenders = []
    for path, rel in _text_files(repo_root):
        text = path.read_bytes().decode("utf-8")                 # must be valid UTF-8
        if _MOJIBAKE.search(text):
            offenders.append(str(rel))
    assert not offenders, f"double-encoded (mojibake) text in: {offenders}"


def test_raw_data_is_git_ignored(repo_root: Path) -> None:
    ignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/raw/*", "data/interim/*", "*.hdf5", "*.h5", ".env"):
        assert pattern in ignore.splitlines(), pattern
