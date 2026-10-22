"""Model interfaces: CNN baseline, GAT, STGNN and evidential head."""

from seismic_ml.models.cnn import CNNBaseline
from seismic_ml.models.edl import EvidentialClassifier
from seismic_ml.models.gat import GAT, GATClassifier, GATOutput
from seismic_ml.models.stgnn import STGNN, STGNNOutput

__all__ = [
    "CNNBaseline",
    "EvidentialClassifier",
    "GAT",
    "GATClassifier",
    "GATOutput",
    "STGNN",
    "STGNNOutput",
]
