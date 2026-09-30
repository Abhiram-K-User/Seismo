"""Plotting helpers.

Matplotlib is an optional dependency (``pip install -e ".[notebooks]"``); it
is imported inside the functions so the core package works without it.
Functions return the Matplotlib ``Figure`` and never save files themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from seismic_ml.data.schemas import SeismicEvent, Station
from seismic_ml.placeholders import NotYetImplementedError


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Plotting needs matplotlib: pip install -e \".[notebooks]\"") from exc
    return plt


def plot_event_waveforms(
    event: SeismicEvent,
    *,
    max_stations: int | None = 6,
    time_window_s: tuple[float, float] | None = None,
) -> Any:
    """Plot every component of the stations of one event (raw amplitudes).

    One row per station, one column per component, shared time axis in
    seconds after the trace start. P/S picks are drawn as vertical lines.
    Amplitudes are shown as stored (counts); each panel has its own y-scale,
    so compare shapes, not amplitudes, across panels.

    Args:
        event: Event to plot.
        max_stations: Plot at most this many stations (in stored order);
            ``None`` plots all.
        time_window_s: Optional ``(start, end)`` in seconds after trace start.
    """
    plt = _pyplot()
    observations = event.observations[:max_stations] if max_stations else event.observations
    if not observations:
        raise ValueError(f"event {event.event_id} has no observations")
    n_comp = max(o.waveform.num_channels for o in observations)
    fig, axes = plt.subplots(len(observations), n_comp, sharex=True, squeeze=False,
                             figsize=(3.2 * n_comp, 1.3 * len(observations) + 0.8))
    for r, obs in enumerate(observations):
        wf = obs.waveform
        t = np.arange(wf.num_samples) / wf.sampling_rate
        picks = {}
        for name, when in (("P", obs.p_arrival_time), ("S", obs.s_arrival_time)):
            if when is not None and wf.start_time is not None:
                picks[name] = (when - wf.start_time).total_seconds()
        for c, comp in enumerate(wf.channels):
            ax = axes[r, c]
            x = wf.data[c]
            if np.all(x == 0):
                ax.text(0.5, 0.5, f"{comp}: all zeros (missing)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="0.4")
            else:
                ax.plot(t, x, lw=0.4, color="0.15")
            for name, sec in picks.items():
                ax.axvline(sec, color="tab:blue" if name == "P" else "tab:red", lw=0.8)
            if time_window_s is not None:
                ax.set_xlim(*time_window_s)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(comp, fontsize=9)
            if c == 0:
                band = obs.metadata.get("station_channel_code") or ""
                ax.set_ylabel(f"{obs.station_id}\n{band}", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time after trace start (s)", fontsize=8)
    fig.suptitle(f"{event.event_id} — label: {event.label} — {event.num_stations} stations",
                 fontsize=10)
    fig.tight_layout()
    return fig


def plot_station_map(
    stations: Sequence[Station],
    *,
    events_lat_lon: np.ndarray | None = None,
    ax: Any = None,
) -> Any:
    """Scatter station locations (and optionally event epicentres) in lon/lat.

    Uses an equirectangular plot with the aspect ratio corrected for the mean
    latitude (no basemap dependency); adequate for a regional network.

    Args:
        stations: Stations to plot (one marker per station).
        events_lat_lon: Optional ``[M, 2]`` array of (lat, lon) epicentres.
        ax: Existing axes to draw into.
    """
    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    if events_lat_lon is not None and len(events_lat_lon):
        ax.scatter(events_lat_lon[:, 1], events_lat_lon[:, 0], s=2, alpha=0.3,
                   color="tab:gray", label="events", rasterized=True)
    lats = np.array([s.latitude for s in stations])
    lons = np.array([s.longitude for s in stations])
    ax.scatter(lons, lats, marker="^", s=18, color="tab:red", edgecolor="k", lw=0.3,
               label=f"stations ({len(stations)})")
    all_lats = lats if events_lat_lon is None else np.r_[lats, events_lat_lon[:, 0]]
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.nanmean(all_lats))))
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    ax.legend(loc="lower left", fontsize=8)
    return fig


def plot_attention_weights(
    attention: torch.Tensor,
    station_ids: Sequence[str],
) -> Any:
    """Visualise GAT attention between stations of one event.

    Args:
        attention: ``[N, N]`` (or ``[H, N, N]`` per head) attention matrix.
        station_ids: Station labels for the axes, length ``N``.

    TODO(Phase 6): implement with the GAT.
    """
    raise NotYetImplementedError("plot_attention_weights", phase=6)


def plot_reliability_diagram(
    bin_confidence: np.ndarray,
    bin_accuracy: np.ndarray,
    bin_counts: np.ndarray,
) -> Any:
    """Plot a reliability diagram from binned calibration statistics.

    TODO(Phase 9): implement with calibration evaluation.
    """
    raise NotYetImplementedError("plot_reliability_diagram", phase=9)
