"""Model interfaces instantiate, validate hyperparameters, and do not compute yet."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from seismic_ml.features import CNNEncoder
from seismic_ml.models import STGNN, CNNBaseline, EvidentialClassifier, GAT, GATClassifier
from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.uncertainty.dirichlet import DirichletOutput

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


def test_gat_instantiation() -> None:
    gat = GAT(D, 32, 8, num_heads=4, num_layers=2, dropout=0.1)
    assert (gat.num_heads, gat.num_layers, gat.dropout) == (4, 2, 0.1)


def test_gat_forward() -> None:
    gat = GAT(D, 32, 8, num_heads=4, num_layers=2, dropout=0.1)
    x = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -1] = False
    adj = torch.ones(B, N, N)

    out = gat(x, adjacency=adj, station_mask=mask, return_attention=True)
    assert out.node_embeddings.shape == (B, N, 8)
    assert torch.all(out.node_embeddings[:, -1] == 0.0)  # Padded station is strictly zero
    assert not torch.all(out.node_embeddings[:, 0] == 0.0)
    assert out.attention_weights is not None
    assert len(out.attention_weights) == 2  # 2 layers
    assert out.attention_weights[0].shape == (B, 4, N, N)

    # Test gradient backward
    loss = out.node_embeddings.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gat.parameters())


def test_gat_classifier_forward() -> None:
    encoder = CNNEncoder(embedding_dim=D)
    gat = GAT(D, 32, 32, num_heads=4, num_layers=2, dropout=0.1)
    model = GATClassifier(
        encoder=encoder,
        gat=gat,
        num_classes=2,
        station_pooling="attention",
        graph_method="knn",
        k_neighbors=2,
    )

    waveforms = torch.randn(B, N, C, T)
    coords = torch.tensor([[[47.0, -122.0], [47.1, -122.1], [47.2, -122.2], [47.3, -122.3]],
                           [[47.0, -122.0], [47.1, -122.1], [47.2, -122.2], [47.3, -122.3]]])
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -1] = False

    logits = model(waveforms, station_mask=mask, coordinates=coords)
    assert logits.shape == (B, 2)
    assert torch.isfinite(logits).all()

    # Test return_attention
    logits_with_attn, attn = model(waveforms, station_mask=mask, coordinates=coords, return_attention=True)
    assert logits_with_attn.shape == (B, 2)
    assert attn is not None and len(attn) == 2

    # Test gradient flow
    loss = logits.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


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


def test_evidential_classifier_forward() -> None:
    head = EvidentialClassifier(D, num_classes=2, evidence_activation="softplus")
    assert head.num_classes == 2
    x = torch.randn(B, D, requires_grad=True)
    out = head(x)
    assert isinstance(out, DirichletOutput)
    assert out.evidence.shape == (B, 2)
    assert torch.all(out.evidence >= 0.0)
    assert torch.allclose(out.belief.sum(dim=-1) + out.uncertainty, torch.ones(B))

    # Backward gradient check
    loss = out.expected_probability.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_models_with_evidential_head() -> None:
    encoder = CNNEncoder(embedding_dim=D)
    cnn_edl = CNNBaseline(encoder, num_classes=2, station_pooling="attention", head_type="evidential")
    gat = GAT(D, 32, 32, num_heads=4, num_layers=2)
    gat_edl = GATClassifier(encoder, gat, num_classes=2, head_type="evidential")

    wfs = torch.randn(B, N, C, T)
    coords = torch.randn(B, N, 2)
    mask = torch.ones(B, N, dtype=torch.bool)

    out_cnn = cnn_edl(wfs, station_mask=mask)
    assert isinstance(out_cnn, DirichletOutput)
    assert out_cnn.uncertainty.shape == (B,)

    out_gat = gat_edl(wfs, station_mask=mask, coordinates=coords)
    assert isinstance(out_gat, DirichletOutput)
    assert out_gat.expected_probability.shape == (B, 2)


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
