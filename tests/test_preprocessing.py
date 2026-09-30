"""Preprocessing functions on SYNTHETIC signals (test-only data; never results)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from seismic_ml.data.schemas import Station, StationObservation, WaveformSample
from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.preprocessing import (
    WindowOutOfRangeError,
    align_to_p_arrival,
    bandpass_filter,
    channel_mask,
    check_signal_quality,
    compute_spectrogram,
    detrend,
    extract_window,
    normalize_waveform,
    preprocess_observation,
    preprocess_waveform,
    resample,
    taper,
)

FS = 100.0
T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)

CONFIG = {
    "detrend": "linear", "taper_s": 2.0,
    "bandpass": {"freqmin": 1.0, "freqmax": 16.0, "corners": 4, "zerophase": True},
    "window": {"pre_p_s": 10.0, "post_p_s": 50.0, "edge_margin_s": 5.0, "resample_margin_s": 1.0},
    "target_sample_rate": 50, "normalization": {"method": "std", "scope": "station"},
    "p_wave_alignment": True, "quality_checks": {"require_p_pick": True, "clip_min_samples": 10},
}


def wf(data: np.ndarray, rate: float = FS) -> WaveformSample:
    return WaveformSample(data=np.asarray(data, dtype=np.float64), sampling_rate=rate,
                          channels=("E", "N", "Z"), start_time=T0)


def sines(freq: float, seconds: float = 60.0, rate: float = FS) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return np.stack([np.sin(2 * np.pi * freq * t)] * 3)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


# --- basic operations ----------------------------------------------------------------------

def test_detrend_removes_mean_and_trend() -> None:
    t = np.arange(1000) / FS
    data = np.stack([3.0 + 0.5 * t, np.full(1000, 7.0), np.zeros(1000)])
    lin = detrend(wf(data), method="linear").data
    assert np.allclose(lin, 0.0, atol=1e-9)
    dem = detrend(wf(data), method="demean").data
    assert np.allclose(dem.mean(axis=1), 0.0) and not np.allclose(dem[0], 0.0)
    assert (dem[2] == 0).all()                                      # zero channel stays zero
    with pytest.raises(ValueError):
        detrend(wf(data), method="spline")


def test_taper_zeroes_ends_only() -> None:
    out = taper(wf(np.ones((3, 2000))), taper_s=2.0).data
    assert np.allclose(out[:, 0], 0.0) and np.allclose(out[:, -1], 0.0)
    assert np.allclose(out[:, 300:1700], 1.0)
    with pytest.raises(ValueError):
        taper(wf(np.ones((3, 10))), taper_s=-1.0)


@pytest.mark.parametrize(("freq", "passes"), [(0.2, False), (5.0, True), (10.0, True), (30.0, False)])
def test_bandpass_passes_band_and_rejects_outside(freq: float, passes: bool) -> None:
    out = bandpass_filter(wf(sines(freq)), freqmin=1.0, freqmax=16.0, corners=4, zerophase=True).data
    ratio = rms(out[2, 1000:-1000]) / rms(sines(freq)[2, 1000:-1000])
    assert (ratio > 0.9) if passes else (ratio < 0.05)


def test_zerophase_filter_has_no_delay_causal_filter_does() -> None:
    x = np.zeros((3, 6000))
    x[:, 3000] = 1.0                                                  # impulse at sample 3000
    zero = bandpass_filter(wf(x), freqmin=1, freqmax=16, corners=4, zerophase=True).data[2]
    causal = bandpass_filter(wf(x), freqmin=1, freqmax=16, corners=4, zerophase=False).data[2]
    assert np.argmax(np.abs(zero)) == 3000                            # symmetric, no delay
    assert np.abs(zero[:3000]).max() > 0                              # but acausal (pre-onset smear)
    assert np.abs(causal[:3000]).max() == 0 and np.argmax(np.abs(causal)) > 3000


def test_bandpass_rejects_invalid_corners() -> None:
    with pytest.raises(ValueError):
        bandpass_filter(wf(sines(5)), freqmin=10, freqmax=5, corners=4, zerophase=True)
    with pytest.raises(ValueError):
        bandpass_filter(wf(sines(5)), freqmin=1, freqmax=60, corners=4, zerophase=True)


def test_resample_preserves_frequency_and_zero_channels() -> None:
    x = sines(5.0, seconds=20.0)
    x[0] = 0.0
    out = resample(wf(x), target_rate=50.0)
    assert out.sampling_rate == 50.0 and out.num_samples == 1000 and out.start_time == T0
    spectrum = np.abs(np.fft.rfft(out.data[2]))
    assert np.fft.rfftfreq(1000, 1 / 50.0)[spectrum.argmax()] == pytest.approx(5.0)
    assert (out.data[0] == 0).all()
    assert resample(wf(x), target_rate=100.0).num_samples == 2000   # identity
    with pytest.raises(ValueError):
        resample(wf(x), target_rate=0)


def test_extract_window_and_errors() -> None:
    data = np.tile(np.arange(1000.0), (3, 1))
    out = extract_window(wf(data), start_offset_s=2.0, duration_s=3.0)
    assert out.num_samples == 300 and out.data[0, 0] == 200.0
    assert out.start_time == T0 + timedelta(seconds=2)
    with pytest.raises(WindowOutOfRangeError):
        extract_window(wf(data), start_offset_s=8.0, duration_s=3.0)
    with pytest.raises(WindowOutOfRangeError):
        extract_window(wf(data), start_offset_s=-0.5, duration_s=1.0)


def _observation(data: np.ndarray, p_s: float | None) -> StationObservation:
    return StationObservation(station=Station("XX.A", 47.0, -122.0), waveform=wf(data),
                              p_arrival_time=None if p_s is None else T0 + timedelta(seconds=p_s))


def test_align_to_p_arrival() -> None:
    data = np.tile(np.arange(6000.0), (3, 1))
    out = align_to_p_arrival(_observation(data, 25.0), pre_arrival_s=10.0, duration_s=20.0)
    assert out.waveform.data[0, 0] == 1500.0 and out.waveform.data[0, 1000] == 2500.0  # P at index 1000
    with pytest.raises(ValueError, match="no P pick"):
        align_to_p_arrival(_observation(data, None), pre_arrival_s=10.0, duration_s=20.0)


def test_check_signal_quality() -> None:
    good = sines(5.0, 10)
    assert check_signal_quality(wf(good), clip_min_samples=10)["passed"]
    bad = good.copy()
    bad[1, 5] = np.nan
    assert check_signal_quality(wf(bad), clip_min_samples=10)["reasons"] == ["non_finite_samples"]
    assert not check_signal_quality(wf(np.zeros((3, 100))), clip_min_samples=10)["passed"]
    clipped = np.clip(good * 2, -1, 1)
    report = check_signal_quality(wf(clipped), clip_min_samples=10)
    assert report["passed"] and report["clipped_components"] == ["E", "N", "Z"]
    flat = good.copy()
    flat[0] = 3.0
    assert check_signal_quality(wf(flat), clip_min_samples=10)["flat_components"] == ["E"]
    assert channel_mask(wf(np.vstack([np.zeros(10), np.ones(10), np.ones(10)]))) == (False, True, True)


# --- normalisation ---------------------------------------------------------------------------

def test_station_scope_keeps_component_ratios_and_zero_channels() -> None:
    rng = np.random.default_rng(0)
    x = np.stack([np.zeros(500), 2.0 * rng.standard_normal(500), 6.0 * rng.standard_normal(500)])
    out, scale = normalize_waveform(x, method="std", scope="station", return_scale=True)
    assert np.allclose(out[1:].std(), 1.0) and (out[0] == 0).all()
    assert np.isclose(out[2].std() / out[1].std(), x[2].std() / x[1].std())   # Z/H ratio preserved
    assert scale.shape == (1, 1)


def test_channel_scope_and_peak_and_event_scope() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 400)) * np.array([[1.0], [3.0], [9.0]])
    assert np.allclose(normalize_waveform(x, method="std", scope="channel").std(axis=1), 1.0)
    assert np.isclose(np.abs(normalize_waveform(x, method="peak", scope="station")).max(), 1.0)
    event = np.stack([x, 10 * x])
    out = normalize_waveform(event, method="peak", scope="event")
    assert np.isclose(np.abs(out).max(), 1.0) and np.isclose(np.abs(out[0]).max(), 0.1)
    assert np.array_equal(normalize_waveform(x, method="none", scope="station"), x)


def test_normalization_edge_cases() -> None:
    flat = np.full((3, 50), 4.0)
    assert np.allclose(normalize_waveform(flat, method="std", scope="station"), 4.0)  # scale -> 1
    assert (normalize_waveform(np.zeros((3, 50)), method="std", scope="station") == 0).all()
    x = np.ones((3, 10))
    masked = normalize_waveform(x, method="peak", scope="station", channel_mask=[True, False, True])
    assert (masked[1] == 0).all()
    with pytest.raises(ValueError):
        normalize_waveform(x, method="minmax", scope="station")
    with pytest.raises(ValueError):
        normalize_waveform(x, method="std", scope="event")          # needs [N, C, T]
    with pytest.raises(ValueError):
        normalize_waveform(np.ones(5), method="std", scope="station")


# --- full chain --------------------------------------------------------------------------------

def _synthetic_event_trace(p_s: float, seconds: float = 150.0, vertical_only: bool = False) -> np.ndarray:
    rng = np.random.default_rng(3)
    t = np.arange(int(seconds * FS)) / FS
    x = 0.01 * rng.standard_normal((3, t.size))
    onset = t >= p_s
    x[:, onset] += np.sin(2 * np.pi * 6.0 * t[onset]) * np.exp(-(t[onset] - p_s) / 5.0)
    if vertical_only:
        x[:2] = 0.0
    return x


def test_preprocess_observation_aligns_p_and_normalises() -> None:
    obs = _observation(_synthetic_event_trace(p_s=55.0, vertical_only=True), 55.0)
    out, info = preprocess_observation(obs, CONFIG)
    w = out.waveform
    assert w.data.shape == (3, 3000) and w.data.dtype == np.float32 and w.sampling_rate == 50.0
    assert w.start_time == obs.p_arrival_time - timedelta(seconds=10)
    assert info["channel_mask"] == (False, False, True)
    assert (w.data[:2] == 0).all() and np.isclose(w.data[2].std(), 1.0, atol=1e-5)
    envelope = np.abs(w.data[2])
    assert envelope[:490].max() < 0.2 * envelope[500:600].max()     # onset right after index 500
    assert len(info["norm_scale"]) == 3
    assert obs.waveform.num_samples == 15000                        # input untouched


def test_preprocess_observation_rejects_bad_inputs() -> None:
    with pytest.raises(WindowOutOfRangeError):
        preprocess_observation(_observation(_synthetic_event_trace(p_s=12.0), 12.0), CONFIG)
    with pytest.raises(WindowOutOfRangeError):
        preprocess_observation(_observation(_synthetic_event_trace(p_s=100.0), 100.0), CONFIG)
    with pytest.raises(ValueError, match="no P pick"):
        preprocess_observation(_observation(_synthetic_event_trace(p_s=55.0), None), CONFIG)
    nan_obs = _observation(_synthetic_event_trace(p_s=55.0), 55.0)
    nan_obs.waveform.data[1, 10] = np.nan
    with pytest.raises(ValueError, match="quality"):
        preprocess_observation(nan_obs, CONFIG)


def test_preprocess_waveform_is_full_trace_conditioning() -> None:
    x = sines(0.2, 150) + sines(5.0, 150)
    out = preprocess_waveform(wf(x), CONFIG)
    assert out.num_samples == x.shape[1] and out.sampling_rate == FS
    assert rms(out.data[2, 3000:-3000]) == pytest.approx(rms(sines(5.0, 150)[2]), rel=0.1)


def test_spectrogram_is_still_a_phase4_placeholder() -> None:
    with pytest.raises(NotYetImplementedError) as info:
        compute_spectrogram(torch.zeros(3, 256), sample_rate=100.0, n_fft=64, hop_length=16)
    assert info.value.phase == 4


def test_placeholder_error_is_not_implemented_error() -> None:
    error = NotYetImplementedError("thing", phase=5)
    assert isinstance(error, NotImplementedError)
    assert error.component == "thing" and error.phase == 5
    assert "Phase 5" in str(error)


def test_inputs_are_not_modified() -> None:
    x = sines(5.0, 20)
    original = x.copy()
    sample = wf(x)
    for fn in (lambda s: detrend(s, method="linear"), lambda s: taper(s, taper_s=1.0),
               lambda s: resample(s, target_rate=50.0)):
        fn(sample)
    assert np.array_equal(sample.data, original)
    assert replace(sample).sampling_rate == FS
