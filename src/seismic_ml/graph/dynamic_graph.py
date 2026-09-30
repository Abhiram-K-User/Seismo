"""Dynamic (per-event, data-dependent) graph construction (placeholder).

Planned with the spatio-temporal model (Phase 7). A "dynamic" graph differs
between events (and possibly over time within an event) instead of being
fixed by station geography alone.
"""

from __future__ import annotations

import torch

from seismic_ml.placeholders import NotYetImplementedError


def build_dynamic_graph(
    node_embeddings: torch.Tensor,
    *,
    spatial_adjacency: torch.Tensor | None = None,
    waveform_similarity: torch.Tensor | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    top_k: int | None = None,
    station_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a per-event adjacency from geography, waveforms and/or embeddings.

    Conceptual starting point (NOT a final mathematical commitment)::

        A_ij = alpha * geographic_similarity(i, j)
             + beta  * waveform_similarity(i, j)
             (+ an optional learned relationship from node embeddings)

    Args:
        node_embeddings: ``[B, N, D]`` station embeddings (Stage A output).
        spatial_adjacency: ``[N, N]`` or ``[B, N, N]`` static geographic term.
        waveform_similarity: ``[B, N, N]`` waveform-similarity term.
        alpha, beta: Mixing weights (fixed or learned — to be decided).
        top_k: Optional sparsification to the ``k`` strongest edges per node.
        station_mask: ``[B, N]`` bool; padded stations get no edges.

    Returns:
        ``[B, N, N]`` non-negative adjacency (convention in ``construction.py``).

    Note:
        A *learned* relationship needs trainable parameters and will then
        live in an ``nn.Module``; this function is the interface for the
        combination step. Compare against the static graph (hypothesis H3).

    TODO(Phase 7): implement.
    """
    raise NotYetImplementedError("build_dynamic_graph", phase=7)
