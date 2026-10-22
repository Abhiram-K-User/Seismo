"""Graph Attention Network (GATv2) over the stations of an event — Stage C (Phase 6).

Implements dynamic multi-head graph attention (Brody et al., 2022) over
variable-station seismic networks, with station-mask zeroing, residual connections,
and full attention matrix inspection.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from seismic_ml.features.cnn_encoder import CNNEncoder
from seismic_ml.graph.construction import build_spatial_graph
from seismic_ml.models.cnn import HEAD_TYPES, STATION_POOLING, StationAttentionPooling
from seismic_ml.models.edl import EvidentialClassifier
from seismic_ml.uncertainty.dirichlet import DirichletOutput
from seismic_ml.utils.validation import check_choice, check_dropout, check_int


class GATOutput(NamedTuple):
    """Result of :meth:`GAT.forward`.

    Attributes:
        node_embeddings: ``[B, N, out_dim]`` updated station embeddings.
        attention_weights: One ``[B, H, N, N]`` tensor per layer
            (``attn[b, h, i, j]`` = weight receiver ``i`` gives sender ``j``),
            or ``None`` unless ``return_attention=True``.
    """

    node_embeddings: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...] | None


class GATv2Layer(nn.Module):
    """Single-layer multi-head dynamic graph attention (GATv2) with residual connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads}).")
        self.head_dim = out_dim // num_heads

        self.lin_l = nn.Linear(in_dim, out_dim, bias=False)  # Receiver projection
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)  # Sender projection
        self.lin_v = nn.Linear(in_dim, out_dim, bias=False)  # Value projection

        # Attention vector a_h and bias
        self.attn_vec = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.res_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        station_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for GATv2Layer.

        Args:
            x: Node features ``[B, N, in_dim]``.
            adjacency: Adjacency matrix ``[B, N, N]`` (non-negative, >0 indicates edge).
            station_mask: Bool mask ``[B, N]`` where True indicates a real station.
            return_attention: Whether to return the attention weights ``[B, H, N, N]``.

        Returns:
            Tuple of updated features ``[B, N, out_dim]`` and attention weights
            ``[B, H, N, N]`` (or None).
        """
        b, n, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        # Projections: [B, N, H, head_dim]
        ql = self.lin_l(x).view(b, n, h, d)
        kr = self.lin_r(x).view(b, n, h, d)
        val = self.lin_v(x).view(b, n, h, d)

        # GATv2 score: a^T * LeakyReLU(Q_i + K_j + bias)
        # Q_i: [B, N, 1, H, d], K_j: [B, 1, N, H, d]
        sum_qk = ql.unsqueeze(2) + kr.unsqueeze(1) + self.attn_bias.view(1, 1, 1, h, d)
        act = F.leaky_relu(sum_qk, negative_slope=0.2)
        # Dot product with attention vector: [B, N, N, H]
        scores = (act * self.attn_vec.view(1, 1, 1, h, d)).sum(dim=-1)
        # Permute to [B, H, N, N] where row i is receiver, col j is sender
        scores = scores.permute(0, 3, 1, 2)

        # Build connectivity mask
        if station_mask is not None:
            real_recv = station_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            real_send = station_mask.unsqueeze(1).unsqueeze(2)   # [B, 1, 1, N]
            pair_mask = real_recv & real_send                    # [B, 1, N, N]
        else:
            pair_mask = torch.ones((b, 1, n, n), dtype=torch.bool, device=x.device)

        if adjacency is not None:
            adj_mask = (adjacency > 0).unsqueeze(1)  # [B, 1, N, N]
            pair_mask = pair_mask & adj_mask

        # Mask scores before softmax
        masked_scores = scores.masked_fill(~pair_mask, -1e9)
        attn_weights = F.softmax(masked_scores, dim=-1)

        # Zero out invalid receivers or nodes without incoming edges
        has_edges = pair_mask.any(dim=-1, keepdim=True)
        if station_mask is not None:
            has_edges = has_edges & station_mask.unsqueeze(1).unsqueeze(-1)
        attn_weights = torch.where(has_edges, attn_weights, torch.zeros_like(attn_weights))

        attn_dropped = self.dropout(attn_weights)

        # Aggregate values: attn [B, H, N, N] x val [B, H, N, d] -> [B, H, N, d]
        val_perm = val.permute(0, 2, 1, 3)
        out = torch.matmul(attn_dropped, val_perm)
        # [B, N, H, d] -> [B, N, out_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, self.out_dim)

        # Residual connection + LayerNorm + ELU
        res = self.res_proj(x)
        out = self.layer_norm(res + out)
        out = F.elu(out)

        # Enforce exact zero embeddings for padded stations
        if station_mask is not None:
            out = out.masked_fill(~station_mask.unsqueeze(-1), 0.0)

        attn_ret = attn_weights if return_attention else None
        return out, attn_ret


class GAT(nn.Module):
    """Learn which stations are relevant to one another via multi-head dynamic attention.

    Args:
        in_dim: Input node feature size (station embedding size ``D``).
        hidden_dim: Hidden size per intermediate layer.
        out_dim: Output node embedding size.
        num_heads: Attention heads ``H`` per layer.
        num_layers: Number of attention layers.
        dropout: Dropout probability in ``[0, 1)``.
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
        self.num_layers = check_int("num_layers", num_layers, minimum=1)
        self.dropout = check_dropout("dropout", dropout)

        layers: list[GATv2Layer] = []
        for i in range(self.num_layers):
            layer_in = self.in_dim if i == 0 else self.hidden_dim
            layer_out = self.out_dim if i == self.num_layers - 1 else self.hidden_dim
            layers.append(
                GATv2Layer(
                    in_dim=layer_in,
                    out_dim=layer_out,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        station_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> GATOutput:
        """Forward pass through GATv2 layers."""
        x = node_features
        all_attentions: list[torch.Tensor] = []

        for layer in self.layers:
            x, attn = layer(
                x,
                adjacency=adjacency,
                station_mask=station_mask,
                return_attention=return_attention,
            )
            if return_attention and attn is not None:
                all_attentions.append(attn)

        attn_tuple = tuple(all_attentions) if return_attention else None
        return GATOutput(node_embeddings=x, attention_weights=attn_tuple)


class GATClassifier(nn.Module):
    """End-to-end Baseline 3: CNN encoder -> GATv2 -> station pooling -> classification head.

    Args:
        encoder: Station CNN encoder (pool_time=True).
        gat: Multi-layer GATv2 module.
        num_classes: Number of target classes (>= 2).
        station_pooling: Pooling method (``"attention"``, ``"mean"``, ``"max"``).
        graph_method: Spatial graph method (``"knn"``, ``"radius"``, ``"gaussian_kernel"``, ``"fully_connected"``).
        k_neighbors: Number of nearest neighbors for ``"knn"``.
        radius_km: Distance threshold in km for ``"radius"``.
        sigma_km: Bandwidth in km for ``"gaussian_kernel"``.
    """

    def __init__(
        self,
        encoder: CNNEncoder,
        gat: GAT,
        num_classes: int = 2,
        *,
        station_pooling: str = "attention",
        graph_method: str = "knn",
        k_neighbors: int = 5,
        radius_km: float | None = None,
        sigma_km: float | None = None,
        head_type: str = "softmax",
        evidence_activation: str = "softplus",
    ) -> None:
        super().__init__()
        if not isinstance(encoder, CNNEncoder):
            raise TypeError(f"encoder must be a CNNEncoder, got {type(encoder).__name__}.")
        if not isinstance(gat, GAT):
            raise TypeError(f"gat must be a GAT, got {type(gat).__name__}.")
        if encoder.embedding_dim != gat.in_dim:
            raise ValueError(
                f"encoder embedding_dim ({encoder.embedding_dim}) must match GAT in_dim ({gat.in_dim})."
            )

        self.encoder = encoder
        self.gat = gat
        self.num_classes = check_int("num_classes", num_classes, minimum=2)
        self.station_pooling = check_choice("station_pooling", station_pooling, STATION_POOLING)
        self.head_type = check_choice("head_type", head_type, HEAD_TYPES)
        self.evidence_activation = evidence_activation
        self.graph_method = graph_method
        self.k_neighbors = k_neighbors
        self.radius_km = radius_km
        self.sigma_km = sigma_km

        if self.station_pooling == "attention":
            self.pool_module = StationAttentionPooling(self.gat.out_dim)
        else:
            self.pool_module = None

        if self.head_type == "evidential":
            self.classifier = EvidentialClassifier(
                self.gat.out_dim,
                num_classes=self.num_classes,
                evidence_activation=self.evidence_activation,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.gat.out_dim, self.num_classes),
            )

    def pool_stations(
        self,
        embeddings: torch.Tensor,
        station_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool updated station embeddings [B, N, out_dim] into event embedding [B, out_dim]."""
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
        coordinates: torch.Tensor | None = None,
        adjacency: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | DirichletOutput | tuple[torch.Tensor | DirichletOutput, tuple[torch.Tensor, ...] | None]:
        """Forward pass.

        Args:
            waveforms: ``[B, N, C, T]`` raw or preprocessed waveforms.
            station_mask: ``[B, N]`` bool mask of real stations.
            coordinates: ``[B, N, 2]`` station coordinates (lat, lon) in degrees.
            adjacency: Optional precomputed adjacency matrix ``[B, N, N]``.
            return_attention: If True, returns ``(output, attention_weights)``.

        Returns:
            Logits or DirichletOutput, or tuple with attention weights.
        """
        # 1. Encode per-station waveforms into node features
        station_emb = self.encoder(waveforms, station_mask=station_mask)  # [B, N, D]

        # 2. Build spatial graph if not provided
        if adjacency is None and coordinates is not None:
            adjacency = build_spatial_graph(
                coordinates,
                method=self.graph_method,
                k=self.k_neighbors,
                radius_km=self.radius_km,
                sigma_km=self.sigma_km,
                station_mask=station_mask,
            )

        # 3. GATv2 message passing between stations
        gat_out = self.gat(
            station_emb,
            adjacency=adjacency,
            station_mask=station_mask,
            return_attention=return_attention,
        )

        # 4. Attention pooling over stations into event embedding
        event_emb = self.pool_stations(gat_out.node_embeddings, station_mask=station_mask)

        # 5. Classification head (softmax logits or DirichletOutput)
        out = self.classifier(event_emb)

        if return_attention:
            return out, gat_out.attention_weights
        return out
