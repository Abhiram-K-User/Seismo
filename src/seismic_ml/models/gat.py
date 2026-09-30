"""Graph Attention Network over the stations of an event — Stage C (placeholder).

Planned for Phase 6. Candidate variants (not decided): the original GAT
(Veličković et al., 2018) and GATv2 (Brody et al., 2022), with or without
edge weights from the station graph as attention priors / edge features;
implemented directly or via PyTorch Geometric (dependency decision in Phase 6).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.utils.validation import check_dropout, check_int


class GATOutput(NamedTuple):
    """Result of :meth:`GAT.forward`.

    Attributes:
        node_embeddings: ``[B, N, out_dim]`` updated station embeddings.
        attention_weights: One ``[B, H, N, N]`` tensor per layer
            (``attn[b, h, i, j]`` = weight station ``i`` gives station ``j``),
            or ``None`` unless ``return_attention=True``.
    """

    node_embeddings: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...] | None


class GAT(nn.Module):
    """Learn which stations are relevant to one another via attention.

    Args:
        in_dim: Input node feature size (station embedding size ``D``).
        hidden_dim: Hidden size per layer.
        out_dim: Output node embedding size.
        num_heads: Attention heads ``H`` per layer.
        num_layers: Number of attention layers.
        dropout: Dropout probability in ``[0, 1)``.

    Expected input:
        node_features: ``[B, N, in_dim]`` station embeddings.
        adjacency: ``[B, N, N]``; ``adjacency[b, i, j] > 0`` allows station
            ``i`` to attend to station ``j`` (dense convention, Decision 003).
        station_mask: ``[B, N]`` bool or None; padded stations neither send
            nor receive messages.

    Expected output:
        :class:`GATOutput`.

    Interpretability:
        Attention weights must remain retrievable (``return_attention=True``)
        so that station relevance can be inspected. Attention weights are not
        automatically faithful explanations; analyse them with care.

    TODO(Phase 6): implement; compare with the non-graph baseline (H1).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = check_int("in_dim", in_dim)
        self.hidden_dim = check_int("hidden_dim", hidden_dim)
        self.out_dim = check_int("out_dim", out_dim)
        self.num_heads = check_int("num_heads", num_heads)
        self.num_layers = check_int("num_layers", num_layers)
        self.dropout = check_dropout("dropout", dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        station_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> GATOutput:
        raise NotYetImplementedError("GAT.forward", phase=6)
