"""Implemented infrastructure utilities: seeding, env info, IDs, logging, validation."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from seismic_ml.utils import (
    configure_logging,
    generate_experiment_id,
    get_environment_info,
    get_logger,
    set_seed,
)
from seismic_ml.utils.validation import check_choice, check_dropout, check_int


def test_set_seed_makes_draws_repeatable() -> None:
    set_seed(123)
    first = (np.random.rand(3), torch.rand(3))
    set_seed(123)
    second = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, "42", True])
def test_set_seed_rejects_invalid_seed(seed) -> None:
    with pytest.raises(ValueError):
        set_seed(seed)


def test_environment_info_keys() -> None:
    info = get_environment_info()
    for key in ("python", "platform", "seismic_ml", "numpy", "torch", "cuda_available"):
        assert info[key]


def test_experiment_id_format_and_uniqueness() -> None:
    now = datetime(2026, 9, 18, 10, 15, 0, tzinfo=timezone.utc)
    first = generate_experiment_id("CNN baseline/v1", now=now)
    second = generate_experiment_id("CNN baseline/v1", now=now)
    assert re.fullmatch(r"20260918T101500Z_CNN-baseline-v1_[0-9a-f]{6}", first)
    assert first != second
    assert generate_experiment_id(now=now).startswith("20260918T101500Z_exp_")


def test_configure_logging_is_idempotent(tmp_path) -> None:
    log_file = tmp_path / "logs" / "run.log"
    configure_logging("DEBUG", log_file=log_file)
    logger = configure_logging("INFO", log_file=log_file)
    assert len(logger.handlers) == 2  # one console + one file handler
    get_logger("tests").info("hello")
    for handler in logger.handlers:
        handler.flush()
    assert "hello" in log_file.read_text(encoding="utf-8")
    configure_logging("INFO")  # release the file handler
    assert len(logging.getLogger("seismic_ml").handlers) == 1


def test_get_logger_namespacing() -> None:
    assert get_logger("training") is get_logger("seismic_ml.training")
    assert get_logger("seismic_ml").name == "seismic_ml"


def test_validation_helpers() -> None:
    assert check_int("x", 3) == 3
    assert check_int("x", 0, minimum=0) == 0
    assert check_dropout("p", 0) == 0.0
    assert check_choice("c", "a", ("a", "b")) == "a"
    with pytest.raises(ValueError):
        check_int("x", 1, minimum=2)
    with pytest.raises(TypeError):
        check_dropout("p", "0.1")
    with pytest.raises(ValueError):
        check_choice("c", "z", ("a", "b"))


def test_visualization_later_phase_placeholders() -> None:
    from seismic_ml.placeholders import NotYetImplementedError
    from seismic_ml.utils.visualization import plot_attention_weights, plot_reliability_diagram

    calls = [
        (6, lambda: plot_attention_weights(torch.zeros(2, 2), ["A.B", "C.D"])),
        (9, lambda: plot_reliability_diagram(np.zeros(3), np.zeros(3), np.zeros(3))),
    ]
    for phase, call in calls:
        with pytest.raises(NotYetImplementedError) as info:
            call()
        assert info.value.phase == phase


def test_plot_event_waveforms_and_station_map(toy_raw_dir, data_config) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from seismic_ml.data.loader import load_events
    from seismic_ml.utils.visualization import plot_event_waveforms, plot_station_map

    event = load_events(toy_raw_dir, data_config, event_ids=["uw1"])[0]
    fig = plot_event_waveforms(event, max_stations=2)
    assert len(fig.axes) == 2 * 3                     # stations x components
    fig2 = plot_station_map(event.stations, events_lat_lon=np.array([[47.0, -122.0]]))
    assert fig2.axes[0].get_xlabel().startswith("longitude")
    plt.close("all")
