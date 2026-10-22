"""Model interfaces instantiate, validate hyperparameters, and do not compute yet."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from seismic_ml.features import CNNEncoder
from seismic_ml.models import STGNN, CNNBaseline, EvidentialClassifier, GAT
from seismic_ml.placeholders import NotYetImplementedError

B, N, C, T, D = 2, 4, 3, 100, 16


# --- CNNEncoder --------------------------------------------------------------


def test_cnn_encoder_instantiation() -> None:
    encoder = CNNEncoder(embedding_dim=D)
    assert isinstance(encoder, nn.Module)
    assert (encoder.embedding_dim, encoder.in_channels) == (D, 3)
    assert encoder.input_representation == "waveform" and encoder.pool_time


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"embedding_dim": 0}, ValueError),
        ({"embedding_dim": 16.0}, TypeError),
        ({"embedding_dim": True}, TypeError),
        ({"embedding_dim": 16, "in_channels": 0}, ValueError),
        ({"embedding_dim": 16, "input_representation": "wavelet"}, ValueError),
    ],
)
def test_cnn_encoder_rejects_invalid_hyperparameters(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        CNNEncoder(**kwargs)


def test_cnn_encoder_forward() -> None:
    encoder = CNNEncoder(embedding_dim=D)
    x = torch.randn(B, N, C, T)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -1] = False  # last station is masked

    out = encoder(x, station_mask=mask)
    assert out.shape == (B, N, D)
    assert torch.all(out[:, -1] == 0.0)  # masked station must be exactly zero
    assert not torch.all(out[:, 0] == 0.0)

    # Test backward pass
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())


def test_cnn_encoder_pool_time_false() -> None:
    encoder = CNNEncoder(embedding_dim=D, pool_time=False)
    x = torch.randn(B, N, C, T)
    out = encoder(x)
    assert out.shape == (B, N, T, D)


# --- CNNBaseline -------------------------------------------------------------


def test_cnn_baseline_instantiation() -> None:
    model = CNNBaseline(CNNEncoder(embedding_dim=D), num_classes=2, station_pooling="mean")
    assert isinstance(model.encoder, CNNEncoder)
    assert model.num_classes == 2


def test_cnn_baseline_validation() -> None:
    with pytest.raises(TypeError):
        CNNBaseline(nn.Identity(), station_pooling="mean")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CNNBaseline(CNNEncoder(embedding_dim=D, pool_time=False), station_pooling="mean")
    with pytest.raises(ValueError):
        CNNBaseline(CNNEncoder(embedding_dim=D), num_classes=1, station_pooling="mean")
    with pytest.raises(ValueError):
        CNNBaseline(CNNEncoder(embedding_dim=D), station_pooling="sum")


@pytest.mark.parametrize("pooling", ["mean", "max", "attention"])
def test_cnn_baseline_forward(pooling: str) -> None:
    encoder = CNNEncoder(embedding_dim=D)
    model = CNNBaseline(encoder, num_classes=2, station_pooling=pooling)
    x = torch.randn(B, N, C, T)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -1] = False

    logits = model(x, station_mask=mask)
    assert logits.shape == (B, 2)
    assert torch.isfinite(logits).all()

    # Test gradient flow
    loss = logits.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


# --- GAT ---------------------------------------------------------------------


def test_gat_instantiation_and_placeholder() -> None:
    gat = GAT(D, 32, 8, num_heads=4, num_layers=2, dropout=0.1)
    assert (gat.num_heads, gat.num_layers, gat.dropout) == (4, 2, 0.1)
    with pytest.raises(NotYetImplementedError) as info:
        gat(torch.zeros(B, N, D), torch.ones(B, N, N), return_attention=True)
    assert info.value.phase == 6


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_gat_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError):
        GAT(D, 32, 8, num_heads=4, num_layers=2, dropout=dropout)


# --- STGNN -------------------------------------------------------------------


def test_stgnn_instantiation_and_placeholder() -> None:
    stgnn = STGNN(D, 32, 8, num_layers=2)
    assert (stgnn.in_dim, stgnn.out_dim, stgnn.dropout) == (D, 8, 0.0)
    with pytest.raises(NotYetImplementedError) as info:
        stgnn(torch.zeros(B, N, 10, D), torch.ones(B, N, N))
    assert info.value.phase == 7


def test_stgnn_rejects_invalid_layers() -> None:
    with pytest.raises(ValueError):
        STGNN(D, 32, 8, num_layers=0)


# --- EvidentialClassifier ----------------------------------------------------


def test_evidential_classifier_instantiation_and_placeholder() -> None:
    head = EvidentialClassifier(D, num_classes=2, evidence_activation="softplus")
    assert head.num_classes == 2
    with pytest.raises(NotYetImplementedError) as info:
        head(torch.zeros(B, D))
    assert info.value.phase == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_classes": 1, "evidence_activation": "relu"},
        {"num_classes": 2, "evidence_activation": "sigmoid"},
    ],
)
def test_evidential_classifier_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EvidentialClassifier(D, **kwargs)
