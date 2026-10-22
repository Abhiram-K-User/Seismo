"""Baseline 1 — CNN + softmax event classifier without a graph (Phase 3).

Composes the shared station encoder (:class:`~seismic_ml.features.cnn_encoder.CNNEncoder`)
with masked pooling over stations and a linear classification head.
"""

from __future__ import annotations

import torch
from torch import nn

from seismic_ml.features.cnn_encoder import CNNEncoder
from seismic_ml.models.edl import EvidentialClassifier
from seismic_ml.uncertainty.dirichlet import DirichletOutput
from seismic_ml.utils.validation import check_choice, check_int

STATION_POOLING: tuple[str, ...] = ("mean", "max", "attention")
HEAD_TYPES: tuple[str, ...] = ("softmax", "evidential")


class StationAttentionPooling(nn.Module):
    """Attention-weighted pooling across variable station observations."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, station_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool station embeddings [B, N, D] into event embeddings [B, D]."""
        scores = self.attn(embeddings).squeeze(-1)  # [B, N]
        if station_mask is not None:
            scores = scores.masked_fill(~station_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)  # [B, N]
        if station_mask is not None:
            weights = weights.masked_fill(~station_mask, 0.0)
            norm = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            weights = weights / norm
        weights = weights.unsqueeze(-1)  # [B, N, 1]
        return (embeddings * weights).sum(dim=1)  # [B, D]


class CNNBaseline(nn.Module):
    """Non-graph event classifier: encoder -> station pooling -> logits.

    Args:
        encoder: Station encoder; must pool over time (``pool_time=True``).
        num_classes: Number of known classes ``K`` (>= 2).
        station_pooling: How station embeddings are aggregated into one event
            embedding. Candidate options: ``"mean"``, ``"max"``, ``"attention"``.
            Pooling ignores padded stations.

    Expected input:
        waveforms: ``[B, N, C, T]``; station_mask: ``[B, N]`` bool or None.

    Expected output:
        Logits ``[B, K]``. Softmax is applied by the loss or at evaluation.
    """

    def __init__(
        self,
        encoder: CNNEncoder,
        num_classes: int = 2,
        *,
        station_pooling: str,
        head_type: str = "softmax",
        evidence_activation: str = "softplus",
    ) -> None:
        super().__init__()
        if not isinstance(encoder, CNNEncoder):
            raise TypeError(f"encoder must be a CNNEncoder, got {type(encoder).__name__}.")
        if not encoder.pool_time:
            raise ValueError("CNNBaseline requires an encoder with pool_time=True.")
        self.encoder = encoder
        self.num_classes = check_int("num_classes", num_classes, minimum=2)
        self.station_pooling = check_choice("station_pooling", station_pooling, STATION_POOLING)
        self.head_type = check_choice("head_type", head_type, HEAD_TYPES)
        self.evidence_activation = evidence_activation

        if self.station_pooling == "attention":
            self.pool_module = StationAttentionPooling(self.encoder.embedding_dim)
        else:
            self.pool_module = None

        if self.head_type == "evidential":
            self.classifier = EvidentialClassifier(
                self.encoder.embedding_dim,
                num_classes=self.num_classes,
                evidence_activation=self.evidence_activation,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.encoder.embedding_dim, self.num_classes),
            )

    def pool_stations(
        self,
        embeddings: torch.Tensor,
        station_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool station embeddings [B, N, D] into an event embedding [B, D]."""
        b, n, d = embeddings.shape
        if station_mask is None:
            station_mask = torch.ones((b, n), dtype=torch.bool, device=embeddings.device)

        if self.station_pooling == "mean":
            mask_expanded = station_mask.unsqueeze(-1).to(embeddings.dtype)
            sum_emb = (embeddings * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)
            return sum_emb / count

        if self.station_pooling == "max":
            mask_expanded = station_mask.unsqueeze(-1)
            masked_emb = embeddings.masked_fill(~mask_expanded, -1e9)
            event_emb = masked_emb.max(dim=1).values
            # Handle empty events where all stations are masked
            has_valid = station_mask.any(dim=1, keepdim=True)
            return torch.where(has_valid, event_emb, torch.zeros_like(event_emb))

        if self.station_pooling == "attention":
            assert self.pool_module is not None
            return self.pool_module(embeddings, station_mask)

        raise ValueError(f"Unknown station_pooling {self.station_pooling!r}")

    def forward(
        self,
        waveforms: torch.Tensor,
        station_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | DirichletOutput:
        """Forward pass.

        Args:
            waveforms: [B, N, C, T] float tensor.
            station_mask: [B, N] bool tensor (optional).
            **kwargs: Extra arguments ignored by CNNBaseline (e.g. coordinates).

        Returns:
            Logits [B, num_classes] (if softmax) or DirichletOutput (if evidential).
        """
        station_embeddings = self.encoder(waveforms, station_mask=station_mask)
        event_embedding = self.pool_stations(station_embeddings, station_mask=station_mask)
        return self.classifier(event_embedding)
