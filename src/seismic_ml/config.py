"""YAML configuration loading.

The configuration is split across ``configs/*.yaml``. ``config.yaml`` is the
entry point; its ``includes`` mapping names sub-config files (relative to the
entry file) that are loaded and attached under the corresponding top-level
keys::

    config = load_config("configs/config.yaml")
    config["seed"]                         # from config.yaml
    config["data"]["labels"]["num_classes"]  # from data.yaml

``null`` values mean *not decided yet* (usually dataset-dependent). Code must
not silently replace them with guessed defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: Top-level keys every resolved configuration must contain.
REQUIRED_KEYS: tuple[str, ...] = ("seed", "paths", "data", "model", "training")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file whose top level is a mapping.

    An empty file yields an empty dict.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the top level of the file is not a mapping.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(
            f"Top level of {path} must be a mapping, got {type(content).__name__}."
        )
    return content


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the entry configuration and attach its included sub-configs.

    Args:
        path: Path to the entry file, normally ``configs/config.yaml``.

    Returns:
        The resolved configuration. The ``includes`` key itself is consumed.

    Raises:
        ValueError: If ``includes`` is malformed, a section is defined both
            inline and via ``includes``, or a required key is missing.
    """
    path = Path(path)
    config = load_yaml(path)

    includes = config.pop("includes", None) or {}
    if not isinstance(includes, dict):
        raise ValueError(f"'includes' in {path} must be a mapping of section -> file.")
    for section, relative_file in includes.items():
        if section in config:
            raise ValueError(
                f"Section '{section}' is defined both inline and via includes in {path}."
            )
        config[section] = load_yaml(path.parent / relative_file)

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Configuration {path} is missing required keys: {missing}.")
    return config
