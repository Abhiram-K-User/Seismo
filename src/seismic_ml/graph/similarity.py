"""Waveform-similarity relationships between stations — Stage B (placeholder)."""

from __future__ import annotations

import torch

from seismic_ml.placeholders import NotYetImplementedError


def compute_waveform_similarity(
    waveforms: torch.Tensor,
    *,
    method: str,
    max_lag: int | None = None,
    station_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise similarity between the waveforms of stations of the SAME event.

    Args:
        waveforms: ``[N, C, T]`` (one event) or ``[B, N, C, T]`` (batch).
        method: Candidate options (not decided): lag-tolerant normalized
            cross-correlation (``"xcorr"``), spectral coherence
            (``"coherence"``), cosine similarity of learned embeddings.
        max_lag: Maximum lag in samples for cross-correlation (moveout
            between stations makes zero-lag comparison meaningless).
        station_mask: ``[N]`` / ``[B, N]`` bool mask; padded stations get zero
            similarity.

    Returns:
        ``[N, N]`` or ``[B, N, N]`` symmetric similarity, the
        ``waveform_similarity`` term of the conceptual edge formulation.

    Note:
        Similarity must be computed within one event only. Comparing a test
        event with training events is template matching and would leak
        information across splits (PROJECT_CONTEXT.md §12).

    TODO(Phase 5): implement and test whether it adds information beyond
        geographic distance (hypothesis H2).
    """
    raise NotYetImplementedError("compute_waveform_similarity", phase=5)
