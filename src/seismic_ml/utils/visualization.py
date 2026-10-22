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

    Returns:
        matplotlib Figure object.
    """
    plt = _pyplot()
    attn_np = attention.detach().cpu().numpy() if isinstance(attention, torch.Tensor) else np.array(attention)

    if attn_np.ndim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn_np, cmap="viridis", vmin=0.0, vmax=max(float(attn_np.max()), 1e-3))
        ax.set_xticks(range(len(station_ids)))
        ax.set_yticks(range(len(station_ids)))
        ax.set_xticklabels(station_ids, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(station_ids, fontsize=9)
        ax.set_xlabel("Sender Station", fontsize=10, fontweight="bold")
        ax.set_ylabel("Receiver Station", fontsize=10, fontweight="bold")
        ax.set_title("Station Graph Attention Weights (GATv2)", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight")
        fig.tight_layout()
        return fig

    if attn_np.ndim == 3:
        h, n, _ = attn_np.shape
        fig, axes = plt.subplots(1, h, figsize=(4.5 * h, 4.2), squeeze=False)
        vmax = max(float(attn_np.max()), 1e-3)
        im = None
        for i in range(h):
            ax = axes[0, i]
            im = ax.imshow(attn_np[i], cmap="viridis", vmin=0.0, vmax=vmax)
            ax.set_xticks(range(len(station_ids)))
            ax.set_yticks(range(len(station_ids)))
            ax.set_xticklabels(station_ids, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(station_ids, fontsize=8)
            ax.set_title(f"Head {i + 1}", fontsize=10, fontweight="bold")
            if i == 0:
                ax.set_ylabel("Receiver Station", fontsize=9)
            ax.set_xlabel("Sender Station", fontsize=9)
        fig.tight_layout()
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.03, label="Attention Weight")
        fig.suptitle("Multi-Head Graph Attention Weights (GATv2)", fontsize=12, fontweight="bold")
        return fig

    raise ValueError(f"attention must be 2D [N, N] or 3D [H, N, N], got shape {attn_np.shape}")


def plot_training_history(history: Sequence[dict[str, Any]]) -> Any:
    """Plot training and validation loss and macro F1 curves over epochs."""
    plt = _pyplot()
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    train_f1 = [h["train"].get("macro_f1", 0.0) for h in history]

    has_val = any("val" in h for h in history)
    val_loss = [h["val"]["loss"] for h in history if "val" in h] if has_val else []
    val_f1 = [h["val"].get("macro_f1", 0.0) for h in history if "val" in h] if has_val else []
    val_epochs = [h["epoch"] for h in history if "val" in h] if has_val else []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Loss curve
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", lw=1.8, ms=4)
    if has_val:
        ax1.plot(val_epochs, val_loss, "s-", label="Val Loss", color="#ff7f0e", lw=1.8, ms=4)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.set_title("Training and Validation Loss", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    # Macro F1 curve
    ax2.plot(epochs, train_f1, "o-", label="Train Macro F1", color="#2ca02c", lw=1.8, ms=4)
    if has_val:
        ax2.plot(val_epochs, val_f1, "s-", label="Val Macro F1", color="#d62728", lw=1.8, ms=4)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Macro F1-Score", fontsize=10)
    ax2.set_title("Training and Validation Macro F1", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str]) -> Any:
    """Plot an annotated confusion matrix heatmap."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")

    total = np.sum(cm)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = int(cm[i, j])
            pct = (count / total * 100) if total > 0 else 0.0
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    return fig


def plot_reliability_diagram(
    bin_confidence: np.ndarray,
    bin_accuracy: np.ndarray,
    bin_counts: np.ndarray,
) -> Any:
    """Plot a reliability diagram from binned calibration statistics.

    TODO(Phase 9): implement with calibration evaluation.
    """
    raise NotYetImplementedError("plot_reliability_diagram", phase=9)
