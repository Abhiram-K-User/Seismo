"""Waveform preprocessing (Phase 2).

Operations on :class:`~seismic_ml.data.schemas.WaveformSample` /
:class:`~seismic_ml.data.schemas.StationObservation`. Every function returns
new objects and never modifies its inputs. Parameter values are not defaulted
here: they come from ``configs/data.yaml`` (``preprocessing``) and are
justified in PROJECT_CONTEXT.md (§11H, Decisions 017–022).

Chain applied by :func:`preprocess_observation`::

    full trace:  detrend -> taper -> band-pass            (edge effects stay far from P)
    window:      cut [P - pre - margin, P + post + margin] at the native rate
                 -> resample -> trim margin                (P lands on an exact sample)
    normalise:   one scale per station over its live components

Components that are all zeros in the raw data ("missing", zero-filled by the
dataset curators) stay exactly zero through every step and are reported in a
channel mask; they are never treated as signal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt
from scipy.signal import detrend as _scipy_detrend
from scipy.signal.windows import tukey

from seismic_ml.data.schemas import StationObservation, WaveformSample
from seismic_ml.preprocessing.normalization import normalize_waveform

DETREND_METHODS: tuple[str, ...] = ("demean", "linear")


class WindowOutOfRangeError(ValueError):
    """The requested window does not fit inside the available trace."""


def channel_mask(waveform: WaveformSample) -> tuple[bool, ...]:
    """True for components that contain any non-zero sample (i.e. real data)."""
    return tuple(bool(np.any(row != 0)) for row in waveform.data)


def _with_data(waveform: WaveformSample, data: np.ndarray, **changes: Any) -> WaveformSample:
    return replace(waveform, data=data, **changes)


def detrend(waveform: WaveformSample, *, method: str) -> WaveformSample:
    """Remove the mean (``"demean"``) or a least-squares line (``"linear"``) per channel."""
    if method not in DETREND_METHODS:
        raise ValueError(f"method must be one of {DETREND_METHODS}, got {method!r}")
    kind = "constant" if method == "demean" else "linear"
    data = _scipy_detrend(np.asarray(waveform.data, dtype=np.float64), axis=-1, type=kind)
    return _with_data(waveform, data)


def taper(waveform: WaveformSample, *, taper_s: float) -> WaveformSample:
    """Apply a cosine (Tukey) taper of ``taper_s`` seconds at each end."""
    if taper_s < 0:
        raise ValueError("taper_s must be >= 0")
    n = waveform.num_samples
    alpha = min(1.0, 2.0 * taper_s * waveform.sampling_rate / n)
    return _with_data(waveform, waveform.data * tukey(n, alpha))


def bandpass_filter(
    waveform: WaveformSample,
    *,
    freqmin: float,
    freqmax: float,
    corners: int,
    zerophase: bool,
) -> WaveformSample:
    """Butterworth band-pass (second-order sections) applied to every channel.

    Args:
        freqmin, freqmax: Corner frequencies in Hz (``0 < freqmin < freqmax < Nyquist``).
        corners: Filter order.
        zerophase: Forward-backward filtering (no phase shift, magnitude response
            squared; slightly acausal, i.e. can smear energy before the P onset).
    """
    nyquist = waveform.sampling_rate / 2.0
    if not 0 < freqmin < freqmax < nyquist:
        raise ValueError(f"need 0 < freqmin < freqmax < Nyquist ({nyquist} Hz), "
                         f"got {freqmin}, {freqmax}")
    sos = butter(corners, [freqmin, freqmax], btype="bandpass", fs=waveform.sampling_rate,
                 output="sos")
    run = sosfiltfilt if zerophase else sosfilt
    return _with_data(waveform, run(sos, waveform.data, axis=-1))


def resample(waveform: WaveformSample, *, target_rate: float) -> WaveformSample:
    """Resample with a polyphase anti-aliasing filter (rational rate ratio).

    The start time is unchanged; the number of samples scales with the rate.
    Edge samples are affected by the anti-aliasing filter — resample windows
    with a margin and trim it afterwards (as :func:`preprocess_observation` does).
    """
    if target_rate <= 0:
        raise ValueError("target_rate must be > 0")
    ratio = Fraction(target_rate / waveform.sampling_rate).limit_denominator(1000)
    if not math.isclose(float(ratio), target_rate / waveform.sampling_rate, rel_tol=1e-9):
        raise ValueError(f"rate ratio {target_rate}/{waveform.sampling_rate} is not a simple fraction")
    if ratio == 1:
        return _with_data(waveform, np.array(waveform.data, copy=True))
    data = resample_poly(waveform.data, ratio.numerator, ratio.denominator, axis=-1)
    return _with_data(waveform, data, sampling_rate=float(target_rate))


def extract_window(
    waveform: WaveformSample,
    *,
    start_offset_s: float,
    duration_s: float,
) -> WaveformSample:
    """Cut ``duration_s`` seconds starting ``start_offset_s`` after the trace start.

    Offsets are rounded to the nearest sample. No padding is ever added.

    Raises:
        WindowOutOfRangeError: If the window does not lie inside the trace.
    """
    rate = waveform.sampling_rate
    start = int(round(start_offset_s * rate))
    length = int(round(duration_s * rate))
    if length <= 0:
        raise ValueError("duration_s must be > 0")
    if start < 0 or start + length > waveform.num_samples:
        raise WindowOutOfRangeError(
            f"window [{start}, {start + length}) outside trace of {waveform.num_samples} samples")
    new_start = (waveform.start_time + timedelta(seconds=start / rate)
                 if waveform.start_time is not None else None)
    return _with_data(waveform, np.array(waveform.data[:, start:start + length]),
                      start_time=new_start)


def _p_offset_s(observation: StationObservation) -> float:
    if observation.p_arrival_time is None:
        raise ValueError(f"{observation.station_id}: no P pick")
    if observation.waveform.start_time is None:
        raise ValueError(f"{observation.station_id}: waveform has no start time")
    return (observation.p_arrival_time - observation.waveform.start_time).total_seconds()


def align_to_p_arrival(
    observation: StationObservation,
    *,
    pre_arrival_s: float,
    duration_s: float,
) -> StationObservation:
    """Window the waveform so that it starts ``pre_arrival_s`` before the P pick.

    Raises:
        ValueError: If there is no P pick.
        WindowOutOfRangeError: If the window does not fit inside the trace.
    """
    offset = _p_offset_s(observation) - pre_arrival_s
    return replace(observation, waveform=extract_window(observation.waveform,
                                                        start_offset_s=offset,
                                                        duration_s=duration_s))


def check_signal_quality(
    waveform: WaveformSample,
    *,
    clip_min_samples: int,
) -> dict[str, Any]:
    """Measure quality indicators of a (raw) waveform; decides only hard failures.

    Hard failures (``passed`` False): any NaN/inf sample, or no live component.
    Soft flags (reported, never dropped here): a live component with at least
    ``clip_min_samples`` samples at its peak |value| (clipping indicator), or a
    flat non-zero component.

    Returns:
        ``{"passed", "reasons", "channel_mask", "clipped_components", "flat_components"}``.
    """
    data = np.asarray(waveform.data)
    mask = channel_mask(waveform)
    reasons: list[str] = []
    if not np.all(np.isfinite(data)):
        reasons.append("non_finite_samples")
    if not any(mask):
        reasons.append("no_live_components")
    clipped, flat = [], []
    for name, live, row in zip(waveform.channels, mask, data):
        if not live:
            continue
        peak = np.abs(row).max()
        if np.ptp(row) == 0:
            flat.append(name)
        elif np.count_nonzero(np.abs(row) >= peak) >= clip_min_samples:
            clipped.append(name)
    return {"passed": not reasons, "reasons": reasons, "channel_mask": mask,
            "clipped_components": clipped, "flat_components": flat}


def preprocess_waveform(waveform: WaveformSample, config: Mapping[str, Any]) -> WaveformSample:
    """Full-trace conditioning: detrend -> taper -> band-pass (``data.preprocessing``).

    Resampling and windowing need the P pick and happen in
    :func:`preprocess_observation`.
    """
    band = config["bandpass"]
    out = detrend(waveform, method=config["detrend"])
    out = taper(out, taper_s=config["taper_s"])
    return bandpass_filter(out, freqmin=band["freqmin"], freqmax=band["freqmax"],
                           corners=band["corners"], zerophase=band["zerophase"])


def preprocess_observation(
    observation: StationObservation,
    config: Mapping[str, Any],
) -> tuple[StationObservation, dict[str, Any]]:
    """Apply the complete configured chain to one station recording.

    Returns:
        The processed observation (window starts ``pre_p_s`` before P, rate =
        ``target_sample_rate``, normalised) and an info dict with the channel
        mask, quality flags and the normalisation scale.

    Raises:
        ValueError: No P pick, or failed hard quality checks.
        WindowOutOfRangeError: Not enough data around P (incl. edge margins).
    """
    raw = observation.waveform
    quality = check_signal_quality(raw, clip_min_samples=config["quality_checks"]["clip_min_samples"])
    if not quality["passed"]:
        raise ValueError(f"{observation.station_id}: failed quality checks {quality['reasons']}")
    win = config["window"]
    pre, post = win["pre_p_s"], win["post_p_s"]
    margin, edge = win["resample_margin_s"], win["edge_margin_s"]

    p_offset = _p_offset_s(observation)
    if p_offset - pre - margin < edge or p_offset + post + margin > raw.duration_s - edge:
        raise WindowOutOfRangeError(
            f"{observation.station_id}: P at {p_offset:.2f} s leaves too little data for "
            f"[-{pre}, +{post}] s with {edge} s edge margins in a {raw.duration_s:.2f} s trace")

    conditioned = preprocess_waveform(raw, config)
    windowed = extract_window(conditioned, start_offset_s=p_offset - pre - margin,
                              duration_s=pre + post + 2 * margin)
    resampled = resample(windowed, target_rate=config["target_sample_rate"])
    trimmed = extract_window(resampled, start_offset_s=margin, duration_s=pre + post)

    mask = quality["channel_mask"]
    data = np.where(np.array(mask)[:, None], trimmed.data, 0.0)   # missing stay exactly zero
    norm = config["normalization"]
    normalised, scale = normalize_waveform(data, method=norm["method"], scope=norm["scope"],
                                           channel_mask=mask, return_scale=True)
    processed = replace(observation, waveform=replace(trimmed, data=normalised.astype(np.float32)))
    per_channel = np.broadcast_to(scale, (data.shape[0], 1)).ravel()
    info = {"channel_mask": mask, "clipped_components": quality["clipped_components"],
            "flat_components": quality["flat_components"],
            "norm_scale": tuple(float(v) for v in per_channel)}
    return processed, info
