"""Waveform-similarity relationships between stations — Stage B (Phase 5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from seismic_ml.utils.validation import check_choice

SUPPORTED_SIMILARITY_METHODS: tuple[str, ...] = ("cosine", "xcorr")


def compute_waveform_similarity(
    waveforms: torch.Tensor,
    *,
    method: str = "cosine",
    max_lag: int | None = None,
    station_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise similarity between the waveforms of stations of the SAME event.

    Args:
        waveforms: ``[N, C, T]`` (one event) or ``[B, N, C, T]`` (batch).
        method: One of ``"cosine"`` or ``"xcorr"``.
        max_lag: Maximum lag in samples for cross-correlation. Required if
            ``method="xcorr"``.
        station_mask: ``[N]`` / ``[B, N]`` bool mask; padded stations get zero
            similarity.

    Returns:
        ``[N, N]`` or ``[B, N, N]`` symmetric similarity matrix in ``[0, 1]``.
    """
    check_choice("method", method, SUPPORTED_SIMILARITY_METHODS)

    orig_ndim = waveforms.ndim
    if orig_ndim == 3:
        waveforms = waveforms.unsqueeze(0)  # [1, N, C, T]
        if station_mask is not None and station_mask.ndim == 1:
            station_mask = station_mask.unsqueeze(0)

    b, n, c, t = waveforms.shape
    device = waveforms.device

    if station_mask is None:
        station_mask = torch.ones((b, n), dtype=torch.bool, device=device)
    else:
        station_mask = station_mask.to(device=device, dtype=torch.bool)

    valid_pairs = station_mask.unsqueeze(-1) & station_mask.unsqueeze(-2)  # [B, N, N]
    eps = 1e-8

    if method == "cosine":
        # Compute normalized cosine similarity across components
        # Reshape to [B, N, C * T]
        flat_wfs = waveforms.view(b, n, c * t)
        norms = torch.norm(flat_wfs, p=2, dim=-1, keepdim=True).clamp(min=eps)
        normalized = flat_wfs / norms  # [B, N, C*T]
        sim = torch.bmm(normalized, normalized.transpose(1, 2))  # [B, N, N]
        sim = torch.clamp(sim, 0.0, 1.0)

    elif method == "xcorr":
        if max_lag is None or max_lag < 0:
            raise ValueError(f"max_lag must be non-negative for method='xcorr', got {max_lag}.")

        # Compute per-component cross-correlation over lag window
        sim = torch.zeros((b, n, n), dtype=torch.float32, device=device)
        for comp in range(c):
            comp_wfs = waveforms[:, :, comp, :]  # [B, N, T]
            comp_norms = torch.norm(comp_wfs, p=2, dim=-1, keepdim=True).clamp(min=eps)
            comp_normed = comp_wfs / comp_norms  # [B, N, T]

            # Cross-correlation via convolution or direct lag shifts
            comp_sim = torch.zeros((b, n, n), dtype=torch.float32, device=device)
            # Evaluate zero lag
            zero_lag_sim = torch.bmm(comp_normed, comp_normed.transpose(1, 2))
            comp_sim = torch.maximum(comp_sim, zero_lag_sim)

            for lag in range(1, max_lag + 1):
                # Shifted waveforms
                left_shift = comp_normed[:, :, lag:]
                right_pad = comp_normed[:, :, :-lag]
                lag_sim = torch.bmm(left_shift, right_pad.transpose(1, 2))
                comp_sim = torch.maximum(comp_sim, lag_sim)
                comp_sim = torch.maximum(comp_sim, lag_sim.transpose(1, 2))

            sim = sim + comp_sim

        sim = sim / float(c)
        sim = torch.clamp(sim, 0.0, 1.0)

    # Set self-similarity to 1.0
    diag_idx = torch.arange(n, device=device)
    sim[..., diag_idx, diag_idx] = 1.0

    # Symmetry and masking
    sim = 0.5 * (sim + sim.transpose(-1, -2))
    sim = sim * valid_pairs.float()

    return sim.squeeze(0) if orig_ndim == 3 else sim
