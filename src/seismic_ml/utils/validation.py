"""Small argument-validation helpers used by module constructors."""

from __future__ import annotations

from collections.abc import Collection


def check_int(name: str, value: int, *, minimum: int = 1) -> int:
    """Return ``value`` if it is an integer ``>= minimum``.

    Raises:
        TypeError: If ``value`` is not an ``int`` (``bool`` is rejected).
        ValueError: If ``value < minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def check_dropout(name: str, value: float) -> float:
    """Return ``value`` as float if it lies in ``[0, 1)``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}.")
    return float(value)


def check_choice(name: str, value: str, choices: Collection[str]) -> str:
    """Return ``value`` if it is one of ``choices``."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}.")
    return value
