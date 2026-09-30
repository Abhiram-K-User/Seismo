"""Time-frequency representation interface (placeholder).

Planned for Phase 4 (time-frequency experiments). Open decisions: STFT
parameters, amplitude scaling (linear/log/dB), window function, and whether
spectrograms are precomputed offline or computed on the fly on the GPU.
"""

from __future__ import annotations

import torch

from seismic_ml.placeholders import NotYetImplementedError


def compute_spectrogram(
    waveforms: torch.Tensor,
    *,
    sample_rate: float,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
) -> torch.Tensor:
    """Compute a magnitude spectrogram for each channel.

    Args:
        waveforms: ``[..., C, T]`` tensor (e.g. ``[B, N, C, T]``).
        sample_rate: Sampling rate in Hz (for frequency-axis bookkeeping).
        n_fft: FFT size.
        hop_length: Hop between frames, in samples.
        win_length: Window length in samples (defaults to ``n_fft``).

    Returns:
        ``[..., C, F, T']`` tensor with ``F = n_fft // 2 + 1`` frequency bins.

    TODO(Phase 4): implement and compare against raw-waveform input.
    """
    raise NotYetImplementedError("compute_spectrogram", phase=4)
