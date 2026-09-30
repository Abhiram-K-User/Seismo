"""Training loop and loss interfaces (placeholders)."""

from seismic_ml.training.losses import (
    annealing_coefficient,
    cross_entropy_loss,
    evidential_loss,
    kl_dirichlet_regularizer,
)
from seismic_ml.training.trainer import Trainer

__all__ = [
    "Trainer",
    "annealing_coefficient",
    "cross_entropy_loss",
    "evidential_loss",
    "kl_dirichlet_regularizer",
]
