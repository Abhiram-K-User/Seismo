"""Static spatial graph construction — Stage B (placeholders).

Adjacency convention (provisional, Decision 003): dense float tensors where
``A[i, j] > 0`` means station ``j`` sends messages to station ``i`` (row =
receiver) with weight ``A[i, j]``; ``0`` means no edge. Batched versions
are ``[B, N, N]``; rows/columns of padded stations are zero. Self-loop policy
is decided with the GAT (Phase 6).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from seismic_ml.data.schemas import Station
from seismic_ml.placeholders import NotYetImplementedError


def pairwise_distances_km(stations: Sequence[Station]) -> torch.Tensor:
    """Great-circle (e.g. haversine) distances between stations.

    Returns:
        Symmetric ``[N, N]`` float tensor in kilometres, zero diagonal.

    TODO(Phase 5): implement.
    """
    raise NotYetImplementedError("pairwise_distances_km", phase=5)


def build_spatial_graph(
    stations: Sequence[Station],
    *,
    method: str,
    k: int | None = None,
    radius_km: float | None = None,
    sigma_km: float | None = None,
) -> torch.Tensor:
    """Build a static geographic adjacency matrix for a set of stations.

    Args:
        stations: The ``N`` stations (graph nodes), in node order.
        method: Candidate options (not decided): ``"knn"`` (``k`` nearest
            stations), ``"radius"`` (stations within ``radius_km``),
            ``"gaussian_kernel"`` (weights ``exp(-d^2 / sigma_km^2)``),
            ``"fully_connected"``.
        k, radius_km, sigma_km: Method-specific parameters.

    Returns:
        ``[N, N]`` non-negative adjacency (see module docstring). This is the
        ``geographic_similarity`` term of the conceptual edge formulation
        ``A_ij = alpha * geographic_similarity + beta * waveform_similarity``.

    TODO(Phase 5): implement; compare methods experimentally.
    """
    raise NotYetImplementedError("build_spatial_graph", phase=5)
