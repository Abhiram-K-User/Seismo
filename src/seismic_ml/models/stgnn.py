"""Spatio-temporal GNN — Stage D (placeholder).

Planned for Phase 7. The exact architecture is to be determined
experimentally, not hard-coded now. Candidate designs (none chosen):

* temporal model per station (1-D conv / GRU / attention over time),
  followed by spatial graph attention;
* interleaved temporal and spatial blocks (STGCN-style);
* spatial attention at every time step over a time-varying (dynamic) graph.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.utils.validation import check_dropout, check_int


class STGNNOutput(NamedTuple):
    """Result of :meth:`STGNN.forward`.

    Attributes:
        event_embedding: ``[B, out_dim]`` — input to the classifier head.
        node_embeddings: ``[B, N, out_dim]`` final station embeddings.
        attention_weights: Spatial attention maps for inspection, or ``None``.
    """

    event_embedding: torch.Tensor
    node_embeddings: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...] | None


class STGNN(nn.Module):
    """Capture temporal information and spatial station relationships jointly.

    Args:
        in_dim: Per-time-step station feature size.
        hidden_dim: Hidden size.
        out_dim: Output embedding size.
        num_layers: Number of spatio-temporal blocks.
        dropout: Dropout probability in ``[0, 1)``.

    Expected input:
        node_sequences: ``[B, N, T', in_dim]`` — station feature sequences,
            e.g. from ``CNNEncoder(pool_time=False)``.
        adjacency: ``[B, N, N]`` (one graph per event) or ``[B, T', N, N]``
            (time-varying graph); dense convention (Decision 003).
        station_mask: ``[B, N]`` bool or None.

    Expected output:
        :class:`STGNNOutput`.

    TODO(Phase 7): select the design experimentally and implement it.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = check_int("in_dim", in_dim)
        self.hidden_dim = check_int("hidden_dim", hidden_dim)
        self.out_dim = check_int("out_dim", out_dim)
        self.num_layers = check_int("num_layers", num_layers)
        self.dropout = check_dropout("dropout", dropout)

    def forward(
        self,
        node_sequences: torch.Tensor,
        adjacency: torch.Tensor,
        station_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> STGNNOutput:
        raise NotYetImplementedError("STGNN.forward", phase=7)
