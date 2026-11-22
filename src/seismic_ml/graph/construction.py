"""Static spatial graph construction and kinematic edge features — Stage B (Phase 5+).

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

SUPPORTED_METHODS: tuple[str, ...] = ("knn", "quadrant_knn", "radius", "gaussian_kernel", "fully_connected")
R_EARTH_KM: float = 6371.0088  # Mean radius of the Earth in km (IUGG standard)
DEFAULT_VP_KM_S: float = 6.2   # Average PNW upper-crustal P-wave velocity (km/s)


def _to_coords_tensor(stations: Sequence[Station] | torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Normalize stations or coordinate tensors into `[B, N, 2]` plus `was_2d` flag."""
    if isinstance(stations, torch.Tensor):
        coords = stations.float()
    elif len(stations) == 0:
        coords = torch.empty((0, 2), dtype=torch.float32)
    else:
        coords = torch.tensor([[s.latitude, s.longitude] for s in stations], dtype=torch.float32)

    if coords.ndim == 2:
        return coords.unsqueeze(0), True
    if coords.ndim == 3:
        return coords, False
    raise ValueError(f"coordinates must have shape [N, 2] or [B, N, 2], got {tuple(coords.shape)}.")


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


def _pairwise_azimuths_rad(coords: torch.Tensor) -> torch.Tensor:
    """Forward spherical bearing from receiver ``i`` to sender ``j`` in radians ``(-pi, pi]``.

    Args:
        coords: Coordinate tensor ``[B, N, 2]`` in degrees ``(lat, lon)``.

    Returns:
        ``[B, N, N]`` azimuth angle tensor (diagonal zero).
    """
    b, n, _ = coords.shape
    if n == 0:
        return torch.empty((b, 0, 0), dtype=coords.dtype, device=coords.device)

    deg_to_rad = math.pi / 180.0
    lat = coords[..., 0] * deg_to_rad
    lon = coords[..., 1] * deg_to_rad

    lat1 = lat.unsqueeze(-1)  # [B, N, 1] receiver i
    lat2 = lat.unsqueeze(-2)  # [B, 1, N] sender j
    dlon = lon.unsqueeze(-2) - lon.unsqueeze(-1)

    y = torch.sin(dlon) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    az = torch.atan2(y, x)

    diag_idx = torch.arange(n, device=coords.device)
    az[..., diag_idx, diag_idx] = 0.0
    return az


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


def build_kinematic_edge_features(
    stations: Sequence[Station] | torch.Tensor,
    *,
    station_mask: torch.Tensor | None = None,
    vp_km_s: float = DEFAULT_VP_KM_S,
) -> torch.Tensor:
    """Compute 4-D kinematic edge features ``[d_norm, dt_p_norm, cos_az, sin_az]``.

    Features per directed edge ``j -> i`` (row ``i`` = receiver, col ``j`` = sender):
    - ``d_norm``: ``log(1 + d_ij) / log(1 + 300)`` (scale-compressed spherical distance)
    - ``dt_p_norm``: ``(d_ij / vp_km_s) / 50.0`` (theoretical P-wave moveout scaled by 50s window)
    - ``cos_az``: ``cos(theta_ij)`` where ``theta_ij`` is spherical bearing from ``i`` to ``j``
    - ``sin_az``: ``sin(theta_ij)`` (sine/cosine encoding eliminates 0/360° wrap discontinuity)

    Self-loops and padded stations receive strictly zero feature vectors.

    Returns:
        ``[N, N, 4]`` or ``[B, N, N, 4]`` float tensor matching input rank.
    """
    if vp_km_s <= 0.0:
        raise ValueError(f"vp_km_s must be positive, got {vp_km_s}.")

    coords, was_2d = _to_coords_tensor(stations)
    b, n, _ = coords.shape
    device = coords.device

    if n == 0:
        return torch.empty((0, 0, 4) if was_2d else (b, 0, 0, 4), dtype=torch.float32, device=device)

    dist = _haversine_distances(coords)  # [B, N, N]
    az = _pairwise_azimuths_rad(coords)  # [B, N, N]

    dist_norm = torch.log1p(dist) / math.log1p(300.0)
    dt_p_norm = (dist / float(vp_km_s)) / 50.0
    cos_az = torch.cos(az)
    sin_az = torch.sin(az)

    edge_attr = torch.stack([dist_norm, dt_p_norm, cos_az, sin_az], dim=-1)  # [B, N, N, 4]

    # Zero out self-loops cleanly (azimuth cos(0)=1 would otherwise appear on diagonal)
    diag_idx = torch.arange(n, device=device)
    edge_attr[:, diag_idx, diag_idx, :] = 0.0

    if station_mask is not None:
        mask = station_mask.unsqueeze(0) if (was_2d and station_mask.ndim == 1) else station_mask
        mask = mask.to(device=device, dtype=torch.bool)
        valid_pair = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(-1)
        edge_attr = edge_attr * valid_pair.to(edge_attr.dtype)

    return edge_attr.squeeze(0) if was_2d else edge_attr


def _select_quadrant_neighbors(
    sub_dist: torch.Tensor,
    sub_az: torch.Tensor,
    k_eff: int,
) -> list[list[int]]:
    """Pick up to ``k_eff`` neighbors per node balanced across four 90° azimuth quadrants."""
    m = sub_dist.shape[0]
    selected_per_node: list[list[int]] = []

    for row in range(m):
        dists = sub_dist[row]
        azs = sub_az[row]
        # Map angle in (-pi, pi] into quadrant 0..3
        quadrants = torch.floor(((azs + math.pi) % (2.0 * math.pi)) / (0.5 * math.pi)).to(torch.int64)
        chosen: list[int] = []
        chosen_set: set[int] = {row}

        # Pass 1: closest neighbor in each quadrant (round-robin up to k_eff)
        per_q_sorted: list[list[int]] = []
        for q in range(4):
            q_nodes = [
                int(c)
                for c in torch.argsort(dists).tolist()
                if c != row and int(quadrants[c].item()) == q and math.isfinite(float(dists[c].item()))
            ]
            per_q_sorted.append(q_nodes)

        ptr = [0, 0, 0, 0]
        while len(chosen) < k_eff:
            progressed = False
            for q in range(4):
                if len(chosen) >= k_eff:
                    break
                if ptr[q] < len(per_q_sorted[q]):
                    cand = per_q_sorted[q][ptr[q]]
                    ptr[q] += 1
                    if cand not in chosen_set:
                        chosen.append(cand)
                        chosen_set.add(cand)
                        progressed = True
            if not progressed:
                break

        selected_per_node.append(chosen)
    return selected_per_node


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
            - ``"quadrant_knn"``: Connect each station to up to ``k`` nearest
              neighbors partitioned across 4 orthogonal 90° azimuth sectors.
            - ``"radius"``: Connect stations within ``radius_km``.
            - ``"gaussian_kernel"``: Continuous weights ``exp(-d^2 / (2 * sigma_km^2))``.
            - ``"fully_connected"``: Connect all stations.
        k: Number of neighbors for ``"knn"`` and ``"quadrant_knn"``.
        radius_km: Distance cutoff in km for ``"radius"``.
        sigma_km: Kernel bandwidth in km for ``"gaussian_kernel"``.
        station_mask: Optional bool mask of shape ``[N]`` or ``[B, N]`` marking
            valid stations. Padded slots receive zero incoming and outgoing edges.
        add_self_loops: Whether to include self-loops on valid stations.

    Returns:
        ``[N, N]`` or ``[B, N, N]`` non-negative adjacency matrix.
    """
    check_choice("method", method, SUPPORTED_METHODS)

    coords, was_2d = _to_coords_tensor(stations)
    dist = _haversine_distances(coords)
    orig_ndim = 2 if was_2d else 3
    if was_2d and station_mask is not None and station_mask.ndim == 1:
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

    if method in ("knn", "quadrant_knn"):
        if k is None or k <= 0:
            raise ValueError(f"k must be a positive integer for method={method!r}, got {k}.")
        check_int("k", k, minimum=1)

        az = _pairwise_azimuths_rad(coords) if method == "quadrant_knn" else None
        adj = torch.zeros((b, n, n), dtype=torch.float32, device=device)

        for i in range(b):
            valid_idx = torch.where(station_mask[i])[0]
            num_valid = len(valid_idx)
            if num_valid <= 1:
                continue

            sub_dist = dist[i][valid_idx][:, valid_idx].clone()
            sub_dist.fill_diagonal_(float("inf"))
            k_eff = min(k, num_valid - 1)

            if method == "knn":
                vals, indices = torch.topk(sub_dist, k=k_eff, largest=False, dim=-1)
                weights = 1.0 / torch.clamp(vals, min=eps)
                for row_pos, (nbr_indices, nbr_weights) in enumerate(zip(indices, weights)):
                    u = valid_idx[row_pos]
                    v = valid_idx[nbr_indices]
                    adj[i, u, v] = nbr_weights
                    adj[i, v, u] = torch.maximum(adj[i, v, u], nbr_weights)
            else:
                assert az is not None
                sub_az = az[i][valid_idx][:, valid_idx]
                chosen_lists = _select_quadrant_neighbors(sub_dist, sub_az, k_eff)
                for row_pos, nbr_list in enumerate(chosen_lists):
                    if not nbr_list:
                        continue
                    nbr_idx_t = torch.tensor(nbr_list, dtype=torch.long, device=device)
                    u = valid_idx[row_pos]
                    v = valid_idx[nbr_idx_t]
                    w = 1.0 / torch.clamp(sub_dist[row_pos, nbr_idx_t], min=eps)
                    adj[i, u, v] = w
                    adj[i, v, u] = torch.maximum(adj[i, v, u], w)

    elif method == "radius":
        if radius_km is None or radius_km <= 0:
            raise ValueError(f"radius_km must be positive for method='radius', got {radius_km}.")
        in_radius = (dist <= radius_km) & (dist > 0.0)
        inv_dist = 1.0 / torch.clamp(dist, min=eps)
        adj = torch.where(in_radius, inv_dist, torch.zeros_like(dist))

    elif method == "gaussian_kernel":
        if sigma_km is None or sigma_km <= 0:
            raise ValueError(f"sigma_km must be positive for method='gaussian_kernel', got {sigma_km}.")
        adj = torch.exp(-(dist ** 2) / (2.0 * (sigma_km ** 2)))
        if not add_self_loops:
            diag_idx = torch.arange(n, device=device)
            adj[..., diag_idx, diag_idx] = 0.0

    elif method == "fully_connected":
        inv_dist = 1.0 / torch.clamp(dist, min=1.0)
        adj = torch.where(dist > 0, inv_dist, torch.zeros_like(dist))

    else:
        raise ValueError(f"Unsupported method: {method}")

    if add_self_loops and method != "gaussian_kernel":
        diag_idx = torch.arange(n, device=device)
        adj[..., diag_idx, diag_idx] = 1.0

    adj = adj * valid_pair_mask.float()
    return adj.squeeze(0) if orig_ndim == 2 else adj
