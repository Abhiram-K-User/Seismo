"""Per-sample uncertainty scores (placeholders, Phase 9).

These scores are candidate OOD / selective-prediction scores. Convention:
**higher score = more uncertain / more OOD-like** (so ``max_probability``
must be negated or inverted before use as an OOD score — be explicit).

Softmax baselines provide ``predictive_entropy`` and ``max_probability``;
EDL models additionally provide Dirichlet-based scores (``u = K / S`` from
``dirichlet.py``, mutual information, dissonance). Which scores are reported
is decided in Phase 9 after a literature review.
"""

from __future__ import annotations

import torch

from seismic_ml.placeholders import NotYetImplementedError


def predictive_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Entropy of the predictive distribution: ``[B, K]`` -> ``[B]``.

    TODO(Phase 9): implement.
    """
    raise NotYetImplementedError("predictive_entropy", phase=9)


def max_probability(probabilities: torch.Tensor) -> torch.Tensor:
    """Maximum class probability (confidence): ``[B, K]`` -> ``[B]``.

    TODO(Phase 9): implement (baseline score of Hendrycks & Gimpel, 2017).
    """
    raise NotYetImplementedError("max_probability", phase=9)


def dirichlet_mutual_information(alpha: torch.Tensor) -> torch.Tensor:
    """Mutual information (epistemic term) of a Dirichlet: ``[B, K]`` -> ``[B]``.

    TODO(Phase 9): implement if adopted after the literature review.
    """
    raise NotYetImplementedError("dirichlet_mutual_information", phase=9)


def dissonance(belief: torch.Tensor) -> torch.Tensor:
    """Belief dissonance (conflicting evidence): ``[B, K]`` -> ``[B]``.

    TODO(Phase 9): implement if adopted after the literature review.
    """
    raise NotYetImplementedError("dissonance", phase=9)
