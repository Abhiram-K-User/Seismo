"""Every module of the package must import cleanly."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "seismic_ml",
    "seismic_ml.config",
    "seismic_ml.paths",
    "seismic_ml.placeholders",
    "seismic_ml.data",
    "seismic_ml.data.schemas",
    "seismic_ml.data.dataset",
    "seismic_ml.data.loader",
    "seismic_ml.data.acquisition",
    "seismic_ml.data.seisbench_format",
    "seismic_ml.data.pnw",
    "seismic_ml.data.inspection",
    "seismic_ml.data.manifest",
    "seismic_ml.preprocessing",
    "seismic_ml.preprocessing.waveform",
    "seismic_ml.preprocessing.spectrogram",
    "seismic_ml.preprocessing.normalization",
    "seismic_ml.features",
    "seismic_ml.features.cnn_encoder",
    "seismic_ml.graph",
    "seismic_ml.graph.construction",
    "seismic_ml.graph.similarity",
    "seismic_ml.graph.dynamic_graph",
    "seismic_ml.models",
    "seismic_ml.models.cnn",
    "seismic_ml.models.gat",
    "seismic_ml.models.stgnn",
    "seismic_ml.models.edl",
    "seismic_ml.uncertainty",
    "seismic_ml.uncertainty.dirichlet",
    "seismic_ml.uncertainty.metrics",
    "seismic_ml.training",
    "seismic_ml.training.trainer",
    "seismic_ml.training.losses",
    "seismic_ml.evaluation",
    "seismic_ml.evaluation.classification",
    "seismic_ml.evaluation.calibration",
    "seismic_ml.evaluation.ood",
    "seismic_ml.utils",
    "seismic_ml.utils.logging",
    "seismic_ml.utils.reproducibility",
    "seismic_ml.utils.validation",
    "seismic_ml.utils.visualization",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_package_version() -> None:
    import seismic_ml

    assert isinstance(seismic_ml.__version__, str) and seismic_ml.__version__


@pytest.mark.parametrize(
    "package",
    [
        "seismic_ml.data",
        "seismic_ml.preprocessing",
        "seismic_ml.features",
        "seismic_ml.graph",
        "seismic_ml.models",
        "seismic_ml.uncertainty",
        "seismic_ml.training",
        "seismic_ml.evaluation",
        "seismic_ml.utils",
    ],
)
def test_public_api_names_exist(package: str) -> None:
    module = importlib.import_module(package)
    for name in module.__all__:
        assert hasattr(module, name), f"{package}.__all__ lists missing name {name!r}"
