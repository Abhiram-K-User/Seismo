"""Unit tests for station graph construction (Phase 5)."""

from __future__ import annotations

import pytest
import torch

from seismic_ml.data.schemas import Station
from seismic_ml.graph import (
    build_dynamic_graph,
    build_kinematic_edge_features,
    build_spatial_graph,
    compute_waveform_similarity,
    pairwise_distances_km,
)
from seismic_ml.placeholders import NotYetImplementedError


@pytest.fixture
def stations() -> list[Station]:
    return [
        Station("XX.A", latitude=0.0, longitude=0.0),
        Station("XX.B", latitude=0.1, longitude=0.1),
        Station("XX.C", latitude=0.2, longitude=0.0),
    ]


def test_pairwise_distances_properties(stations: list[Station]) -> None:
    dist = pairwise_distances_km(stations)
    assert dist.shape == (3, 3)
    assert torch.allclose(dist, dist.T)  # Symmetric
    assert torch.all(dist.diag() == 0.0)  # Zero diagonal
    assert torch.all(dist >= 0.0)  # Non-negative

    # Off-diagonal should be non-zero
    assert dist[0, 1] > 0.0
    assert dist[0, 2] > 0.0

    # 1 degree latitude is approx 111 km
    lat_stations = [
        Station("S1", latitude=0.0, longitude=0.0),
        Station("S2", latitude=1.0, longitude=0.0),
    ]
    dist_1deg = pairwise_distances_km(lat_stations)
    assert pytest.approx(111.19, rel=1e-2) == dist_1deg[0, 1].item()


def test_pairwise_distances_tensor_inputs() -> None:
    # 2D coords tensor [N, 2]
    coords = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0]])
    dist_2d = pairwise_distances_km(coords)
    assert dist_2d.shape == (3, 3)
    assert torch.allclose(dist_2d, dist_2d.T)

    # 3D batched coords [B, N, 2]
    batched_coords = torch.stack([coords, coords], dim=0)
    dist_3d = pairwise_distances_km(batched_coords)
    assert dist_3d.shape == (2, 3, 3)
    assert torch.allclose(dist_3d[0], dist_3d[1])
    assert torch.allclose(dist_3d, dist_3d.transpose(-1, -2))


def test_pairwise_distances_empty() -> None:
    empty_dist = pairwise_distances_km([])
    assert empty_dist.shape == (0, 0)


@pytest.mark.parametrize("method", ["knn", "quadrant_knn", "radius", "gaussian_kernel", "fully_connected"])
def test_build_spatial_graph_methods(stations: list[Station], method: str) -> None:
    kwargs = {}
    if method in ("knn", "quadrant_knn"):
        kwargs["k"] = 2
    elif method == "radius":
        kwargs["radius_km"] = 50.0
    elif method == "gaussian_kernel":
        kwargs["sigma_km"] = 25.0

    adj = build_spatial_graph(stations, method=method, **kwargs)
    assert adj.shape == (3, 3)
    assert torch.all(adj >= 0.0)
    assert torch.allclose(adj, adj.T)
    # Self-loops should be non-zero
    assert torch.all(adj.diag() > 0.0)


def test_quadrant_knn_preserves_azimuthal_coverage() -> None:
    # Center node 0 surrounded by 2 clustered nodes in NE quadrant (0.01, 0.01) and (0.02, 0.02)
    # and 1 slightly farther node in SW quadrant (-0.05, -0.05).
    # Standard k=2 knn only connects NE neighbors; quadrant_knn (k=2) must connect SW too.
    coords = torch.tensor([
        [47.00, -122.00],  # 0: center
        [47.01, -121.99],  # 1: NE very close
        [47.02, -121.98],  # 2: NE second closest
        [46.95, -122.05],  # 3: SW opposite quadrant
    ])
    adj_q = build_spatial_graph(coords, method="quadrant_knn", k=2, add_self_loops=False)
    assert adj_q[0, 1] > 0.0
    assert adj_q[0, 3] > 0.0  # Opposite SW quadrant preserved!


def test_build_kinematic_edge_features() -> None:
    coords = torch.tensor([
        [[47.0, -122.0], [47.5, -122.0], [47.0, -121.5], [48.0, -121.0]],
    ])
    mask = torch.tensor([[True, True, True, False]])  # 4th station padded
    edges = build_kinematic_edge_features(coords, station_mask=mask, vp_km_s=6.2)

    assert edges.shape == (1, 4, 4, 4)
    assert torch.isfinite(edges).all()
    # Self-loops must be zero
    assert torch.all(edges[0, 0, 0] == 0.0)
    # Padded station edges must be strictly zero
    assert torch.all(edges[0, 3, :, :] == 0.0)
    assert torch.all(edges[0, :, 3, :] == 0.0)
    # Station 1 is due north of Station 0 -> cos_az ~ 1.0, sin_az ~ 0.0
    assert edges[0, 0, 1, 0] > 0.0  # norm distance
    assert edges[0, 0, 1, 1] > 0.0  # norm P travel time
    assert edges[0, 0, 1, 2].item() == pytest.approx(1.0, abs=1e-2)
    assert edges[0, 0, 1, 3].item() == pytest.approx(0.0, abs=1e-2)
    # Opposite bearing (1 -> 0 is due south) -> cos_az ~ -1.0
    assert edges[0, 1, 0, 2].item() == pytest.approx(-1.0, abs=1e-2)


def test_build_spatial_graph_station_mask() -> None:
    coords = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0]])
    mask = torch.tensor([True, True, False])  # 3rd station is padded

    adj = build_spatial_graph(coords, method="knn", k=1, station_mask=mask)
    assert adj.shape == (3, 3)
    # Padded station row and col must be strictly zero
    assert torch.all(adj[2, :] == 0.0)
    assert torch.all(adj[:, 2] == 0.0)
    # Real stations have edge and self loops
    assert adj[0, 0] > 0.0
    assert adj[1, 1] > 0.0
    assert adj[0, 1] > 0.0


def test_build_spatial_graph_invalid_inputs(stations: list[Station]) -> None:
    with pytest.raises(ValueError):
        build_spatial_graph(stations, method="unknown")
    with pytest.raises(ValueError):
        build_spatial_graph(stations, method="knn", k=0)
    with pytest.raises(ValueError):
        build_spatial_graph(stations, method="radius", radius_km=-5.0)
    with pytest.raises(ValueError):
        build_spatial_graph(stations, method="gaussian_kernel", sigma_km=0.0)


def test_compute_waveform_similarity_cosine() -> None:
    # 2 stations, 3 components, 100 samples
    x = torch.randn(2, 3, 100)
    sim = compute_waveform_similarity(x, method="cosine")
    assert sim.shape == (2, 2)
    assert torch.allclose(sim, sim.T)
    assert torch.allclose(sim.diag(), torch.tensor([1.0, 1.0]))
    assert torch.all(sim >= 0.0) and torch.all(sim <= 1.0)


def test_compute_waveform_similarity_xcorr() -> None:
    x = torch.randn(2, 4, 3, 100)  # Batched
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    sim = compute_waveform_similarity(x, method="xcorr", max_lag=5, station_mask=mask)
    assert sim.shape == (2, 4, 4)
    assert torch.allclose(sim, sim.transpose(-1, -2))
    # Check that padded stations have zeros
    assert torch.all(sim[0, 3, :] == 0.0)
    assert torch.all(sim[0, :, 3] == 0.0)


def test_build_dynamic_graph_is_placeholder() -> None:
    with pytest.raises(NotYetImplementedError) as info:
        build_dynamic_graph(torch.zeros(2, 3, 16), alpha=0.5, beta=0.5)
    assert info.value.phase == 7
