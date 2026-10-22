"""Static spatial graph construction — Stage B (Phase 5).

Adjacency convention (Decision 003): dense float tensors where ``A[i, j] > 0``
means station ``j`` sends messages to station ``i`` (row = receiver) with
weight ``A[i, j]``; ``0`` means no edge. Batched versions are ``[B, N, N]``;
rows/columns of padded stations are zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from seismic_ml.data.schemas import Station
from seismic_ml.utils.validation import check_choice, check_int

SUPPORTED_METHODS: tuple[str, ...] = ("knn", "radius", "gaussian_kernel", "fully_connected")
R_EARTH_KM: float = 6371.0088  # Mean radius of the Earth in km (IUGG standard)


def _haversine_distances(coords: torch.Tensor) -> torch.Tensor:
    """Vectorized Haversine distance from coordinates in degrees.

    Args:
        coords: Tensor of shape ``[N, 2]`` or ``[B, N, 2]`` containing
            (latitude, longitude) pairs in degrees.

    Returns:
        Symmetric distance tensor of shape ``[N, N]`` or ``[B, N, N]`` in km.
    """
    orig_ndim = coords.ndim
    if orig_ndim == 2:
        coords = coords.unsqueeze(0)  # [1, N, 2]

    b, n, _ = coords.shape
    if n == 0:
        return torch.empty((0, 0) if orig_ndim == 2 else (b, 0, 0), dtype=coords.dtype, device=coords.device)

    # Convert degrees to radians
    deg_to_rad = math.pi / 180.0
    lat = coords[..., 0] * deg_to_rad  # [B, N]
    lon = coords[..., 1] * deg_to_rad  # [B, N]

    lat1 = lat.unsqueeze(-1)  # [B, N, 1]
    lat2 = lat.unsqueeze(-2)  # [B, 1, N]
    lon1 = lon.unsqueeze(-1)  # [B, N, 1]
    lon2 = lon.unsqueeze(-2)  # [B, 1, N]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat * 0.5) ** 2 + torch.cos(lat1) * torch.cos(lat2) * (torch.sin(dlon * 0.5) ** 2)
    a = torch.clamp(a, 0.0, 1.0)
    c = 2.0 * torch.asin(torch.sqrt(a))
    dist = R_EARTH_KM * c

    # Enforce exact symmetry and zero diagonal
    dist = 0.5 * (dist + dist.transpose(-1, -2))
    diag_idx = torch.arange(n, device=coords.device)
    dist[..., diag_idx, diag_idx] = 0.0
    dist = torch.clamp(dist, min=0.0)

    return dist.squeeze(0) if orig_ndim == 2 else dist


def pairwise_distances_km(
    stations: Sequence[Station] | torch.Tensor,
) -> torch.Tensor:
    """Great-circle (haversine) distances between stations in kilometres.

    Args:
        stations: Either a sequence of :class:`~seismic_ml.data.schemas.Station`
            objects, or a coordinate tensor of shape ``[N, 2]`` or ``[B, N, 2]``
            where the last dimension holds ``(latitude, longitude)`` in degrees.

    Returns:
        Symmetric ``[N, N]`` (or ``[B, N, N]``) float tensor in kilometres with
        zero diagonal.
    """
    if isinstance(stations, torch.Tensor):
        return _haversine_distances(stations.float())

    n = len(stations)
    if n == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    coords_list = [[s.latitude, s.longitude] for s in stations]
    coords = torch.tensor(coords_list, dtype=torch.float32)
    return _haversine_distances(coords)


def build_spatial_graph(
    stations: Sequence[Station] | torch.Tensor,
    *,
    method: str = "knn",
    k: int | None = None,
    radius_km: float | None = None,
    sigma_km: float | None = None,
    station_mask: torch.Tensor | None = None,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Build a static geographic adjacency matrix for a set of stations.

    Args:
        stations: The ``N`` stations (as a sequence of Station or a coordinate
            tensor ``[N, 2]`` or ``[B, N, 2]``).
        method: Graph construction method. One of:
            - ``"knn"``: Connect each station to its ``k`` nearest neighbors.
            - ``"radius"``: Connect stations within ``radius_km``.
            - ``"gaussian_kernel"``: Continuous weights ``exp(-d^2 / (2 * sigma_km^2))``.
            - ``"fully_connected"``: Connect all stations.
        k: Number of neighbors for ``"knn"``.
        radius_km: Distance cutoff in km for ``"radius"``.
        sigma_km: Kernel bandwidth in km for ``"gaussian_kernel"``.
        station_mask: Optional bool mask of shape ``[N]`` or ``[B, N]`` marking
            valid stations. Padded slots receive zero incoming and outgoing edges.
        add_self_loops: Whether to include self-loops on valid stations.

    Returns:
        ``[N, N]`` or ``[B, N, N]`` non-negative adjacency matrix.
    """
    check_choice("method", method, SUPPORTED_METHODS)

    dist = pairwise_distances_km(stations)
    orig_ndim = dist.ndim
    if orig_ndim == 2:
        dist = dist.unsqueeze(0)  # [1, N, N]
        if station_mask is not None and station_mask.ndim == 1:
            station_mask = station_mask.unsqueeze(0)  # [1, N]

    b, n, _ = dist.shape
    device = dist.device

    if station_mask is None:
        station_mask = torch.ones((b, n), dtype=torch.bool, device=device)
    else:
        station_mask = station_mask.to(device=device, dtype=torch.bool)

    # 2D mask for valid pairs: both sender and receiver must be real stations
    valid_pair_mask = station_mask.unsqueeze(-1) & station_mask.unsqueeze(-2)  # [B, N, N]

    eps = 1e-4

    if method == "knn":
        if k is None or k <= 0:
            raise ValueError(f"k must be a positive integer for method='knn', got {k}.")
        check_int("k", k, minimum=1)

        adj = torch.zeros((b, n, n), dtype=torch.float32, device=device)
        for i in range(b):
            valid_idx = torch.where(station_mask[i])[0]
            num_valid = len(valid_idx)
            if num_valid <= 1:
                continue

            sub_dist = dist[i][valid_idx][:, valid_idx].clone()
            # Mask diagonal so node is not its own neighbor in kNN search
            sub_dist.fill_diagonal_(float("inf"))

            k_eff = min(k, num_valid - 1)
            vals, indices = torch.topk(sub_dist, k=k_eff, largest=False, dim=-1)

            # Assign inverse distance weights
            weights = 1.0 / torch.clamp(vals, min=eps)
            for row_pos, (nbr_indices, nbr_weights) in enumerate(zip(indices, weights)):
                u = valid_idx[row_pos]
                v = valid_idx[nbr_indices]
                adj[i, u, v] = nbr_weights
                # Make symmetric: if j is neighbor of i, i sends to j as well
                adj[i, v, u] = torch.maximum(adj[i, v, u], nbr_weights)

    elif method == "radius":
        if radius_km is None or radius_km <= 0:
            raise ValueError(f"radius_km must be positive for method='radius', got {radius_km}.")
        in_radius = (dist <= radius_km) & (dist > 0.0)
        inv_dist = 1.0 / torch.clamp(dist, min=eps)
        adj = torch.where(in_radius, inv_dist, torch.zeros_like(dist))

    elif method == "gaussian_kernel":
        if sigma_km is None or sigma_km <= 0:
            raise ValueError(f"sigma_km must be positive for method='gaussian_kernel', got {sigma_km}.")
        adj = torch.exp(- (dist ** 2) / (2.0 * (sigma_km ** 2)))
        if not add_self_loops:
            diag_idx = torch.arange(n, device=device)
            adj[..., diag_idx, diag_idx] = 0.0

    elif method == "fully_connected":
        # All valid stations are connected with inverse distance weights
        inv_dist = 1.0 / torch.clamp(dist, min=1.0)
        adj = torch.where(dist > 0, inv_dist, torch.zeros_like(dist))

    else:
        raise ValueError(f"Unsupported method: {method}")

    # Add self-loops if requested
    if add_self_loops and method != "gaussian_kernel":
        diag_idx = torch.arange(n, device=device)
        adj[..., diag_idx, diag_idx] = 1.0

    # Mask out padded stations
    adj = adj * valid_pair_mask.float()

    return adj.squeeze(0) if orig_ndim == 2 else adj
