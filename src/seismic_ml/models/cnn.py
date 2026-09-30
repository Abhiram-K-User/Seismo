"""Baseline 1 — CNN + softmax event classifier without a graph (placeholder).

Planned for Phase 3. Composes the shared station encoder
(:class:`~seismic_ml.features.cnn_encoder.CNNEncoder`) with masked pooling
over stations and a linear classification head.
"""

from __future__ import annotations

import torch
from torch import nn

from seismic_ml.features.cnn_encoder import CNNEncoder
from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.utils.validation import check_choice, check_int

STATION_POOLING: tuple[str, ...] = ("mean", "max", "attention")


class CNNBaseline(nn.Module):
    """Non-graph event classifier: encoder -> station pooling -> logits.

    Args:
        encoder: Station encoder; must pool over time (``pool_time=True``).
        num_classes: Number of known classes ``K`` (>= 2).
        station_pooling: How station embeddings are aggregated into one event
            embedding. Candidate options: ``"mean"``, ``"max"``, ``"attention"``.
            Pooling must ignore padded stations.

    Expected input:
        waveforms: ``[B, N, C, T]``; station_mask: ``[B, N]`` bool or None.

    Expected output:
        Logits ``[B, K]``. Softmax is applied by the loss / at evaluation.

    TODO(Phase 3): implement pooling + head; train as Baseline 1.
    """

    def __init__(
        self,
        encoder: CNNEncoder,
        num_classes: int = 2,
        *,
        station_pooling: str,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, CNNEncoder):
            raise TypeError(f"encoder must be a CNNEncoder, got {type(encoder).__name__}.")
        if not encoder.pool_time:
            raise ValueError("CNNBaseline requires an encoder with pool_time=True.")
        self.encoder = encoder
        self.num_classes = check_int("num_classes", num_classes, minimum=2)
        self.station_pooling = check_choice("station_pooling", station_pooling, STATION_POOLING)

    def forward(
        self,
        waveforms: torch.Tensor,
        station_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotYetImplementedError("CNNBaseline.forward", phase=3)
