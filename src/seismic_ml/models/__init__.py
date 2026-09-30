"""Model interfaces: CNN baseline, GAT, STGNN and evidential head.

All classes are placeholders in Phase 0: constructors validate
hyperparameters; ``forward`` raises ``NotYetImplementedError``.
"""

from seismic_ml.models.cnn import CNNBaseline
from seismic_ml.models.edl import EvidentialClassifier
from seismic_ml.models.gat import GAT, GATOutput
from seismic_ml.models.stgnn import STGNN, STGNNOutput

__all__ = [
    "CNNBaseline",
    "EvidentialClassifier",
    "GAT",
    "GATOutput",
    "STGNN",
    "STGNNOutput",
]
