"""Graph construction interfaces exist and are placeholders."""

from __future__ import annotations

import pytest
import torch

from seismic_ml.data.schemas import Station
from seismic_ml.graph import (
    build_dynamic_graph,
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


def test_pairwise_distances_is_placeholder(stations: list[Station]) -> None:
    with pytest.raises(NotYetImplementedError) as info:
        pairwise_distances_km(stations)
    assert info.value.phase == 5


def test_build_spatial_graph_is_placeholder(stations: list[Station]) -> None:
    with pytest.raises(NotYetImplementedError) as info:
        build_spatial_graph(stations, method="knn", k=2)
    assert info.value.phase == 5


def test_compute_waveform_similarity_is_placeholder() -> None:
    with pytest.raises(NotYetImplementedError) as info:
        compute_waveform_similarity(torch.zeros(3, 3, 100), method="xcorr", max_lag=10)
    assert info.value.phase == 5


def test_build_dynamic_graph_is_placeholder() -> None:
    with pytest.raises(NotYetImplementedError) as info:
        build_dynamic_graph(torch.zeros(2, 3, 16), alpha=0.5, beta=0.5)
    assert info.value.phase == 7
