"""Amplitude normalization.

The normalization *scope* is a scientific decision, not a detail: relative
amplitudes are physically informative for source discrimination (for example
P/S amplitude ratios, vertical vs horizontal energy, and amplitude decay
across stations). Per-channel normalization removes inter-component ratios;
per-station normalization removes inter-station amplitude information;
per-event normalization preserves both.

For PNW the amplitudes are raw counts from different instrument types (no
response removal), so inter-station amplitudes are not physically comparable
and the project uses ``scope="station"`` (Decision 021). Statistics are
computed over *live* components only (zero-filled missing components would
otherwise bias them) and within a single sample, so nothing is fitted on
other data (no train/test leakage).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

NORMALIZATION_METHODS: tuple[str, ...] = ("std", "peak", "none")
NORMALIZATION_SCOPES: tuple[str, ...] = ("channel", "station", "event")


def normalize_waveform(
    data: np.ndarray,
    *,
    method: str,
    scope: str,
    channel_mask: Sequence[bool] | np.ndarray | None = None,
    eps: float = 1e-8,
    return_scale: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Normalize waveform amplitudes.

    Args:
        data: ``[C, T]`` (one station) or ``[N, C, T]`` (one event) array.
        method: ``"std"`` (divide by standard deviation), ``"peak"`` (by max
            |amplitude|) or ``"none"``.
        scope: Axes sharing one scale — ``"channel"`` (each component),
            ``"station"`` (all live components of a station) or ``"event"``
            (all live components of all stations; needs ``[N, C, T]``).
        channel_mask: Live components, shape ``[C]`` or ``[N, C]``; masked
            components are excluded from the statistic and returned as zeros.
            Default: components with any non-zero sample.
        eps: Scales below ``eps`` (flat or empty input) are replaced by 1.
        return_scale: Also return the scale (broadcastable to ``data``).

    Returns:
        Normalized array of the same shape (and the scale if requested).
    """
    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"method must be one of {NORMALIZATION_METHODS}, got {method!r}")
    if scope not in NORMALIZATION_SCOPES:
        raise ValueError(f"scope must be one of {NORMALIZATION_SCOPES}, got {scope!r}")
    x = np.asarray(data, dtype=np.float64)
    if x.ndim not in (2, 3):
        raise ValueError(f"data must be [C, T] or [N, C, T], got {x.shape}")
    if scope == "event" and x.ndim != 3:
        raise ValueError("scope='event' needs [N, C, T] input")
    mask = (np.any(x != 0, axis=-1) if channel_mask is None
            else np.broadcast_to(np.asarray(channel_mask, dtype=bool), x.shape[:-1]))
    live = mask[..., None]

    if method == "none":
        scale = np.ones(x.shape[:-2] + (1, 1))
    else:
        masked = np.where(live, x, np.nan)
        axis = {"channel": -1, "station": (-2, -1), "event": (-3, -2, -1)}[scope]
        with warnings.catch_warnings():            # all-NaN slices (no live channel)
            warnings.simplefilter("ignore", RuntimeWarning)
            if method == "std":
                scale = np.nanstd(masked, axis=axis, keepdims=True)
            else:
                scale = np.nanmax(np.abs(masked), axis=axis, keepdims=True)
        scale = np.where(np.isfinite(scale) & (scale > eps), scale, 1.0)

    out = np.where(live, x / scale, 0.0)
    return (out, scale) if return_scale else out
