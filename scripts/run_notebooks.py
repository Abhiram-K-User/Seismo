"""Execute notebooks in place with explicit UTF-8 I/O.

Why not ``jupyter-execute``? On Windows its CLI opens notebooks with the
system code page (cp1252), which silently double-encodes every non-ASCII
character when the executed notebook is saved (an em dash turns into three
garbage characters). tests/test_paths.py guards against this regression.

Usage:
    python scripts/run_notebooks.py notebooks/01_data_exploration.ipynb [more.ipynb ...]

Each notebook runs with its own directory as the working directory (the setup
cells locate the project root from there). The data root follows the usual
precedence ($SEISMIC_DATA_ROOT > configs/config.yaml). Exits non-zero if any
cell raises.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def run(path: Path, timeout: int) -> None:
    text = path.read_text(encoding="utf-8")
    nb = nbformat.reads(text, as_version=4)
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    path.write_text(nbformat.writes(nb) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--timeout", type=int, default=1800, help="Per-cell timeout in seconds.")
    args = parser.parse_args(argv)
    for path in args.notebooks:
        try:
            run(path, args.timeout)
        except CellExecutionError as exc:
            print(f"{path}: cell failed\n{exc}", file=sys.stderr)
            return 1
        print(f"executed {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
