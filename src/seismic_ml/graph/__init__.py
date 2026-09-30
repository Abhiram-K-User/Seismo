"""Station graph construction (Stage B).

One node = one seismic station. Graphs are represented as dense adjacency
tensors (provisional convention, see ``construction.py`` and Decision 003
in PROJECT_CONTEXT.md).
"""

from seismic_ml.graph.construction import build_spatial_graph, pairwise_distances_km
from seismic_ml.graph.dynamic_graph import build_dynamic_graph
from seismic_ml.graph.similarity import compute_waveform_similarity

__all__ = [
    "build_dynamic_graph",
    "build_spatial_graph",
    "compute_waveform_similarity",
    "pairwise_distances_km",
]
