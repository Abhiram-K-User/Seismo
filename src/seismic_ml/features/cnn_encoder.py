"""Station-level CNN encoder interface — Stage A (placeholder).

Raw-waveform input is planned for Phase 3; time-frequency input for Phase 4.
Nothing in this module is a working model yet: the constructor only records
and validates hyperparameters, and ``forward`` raises.
"""

from __future__ import annotations

import torch
from torch import nn

from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.utils.validation import check_choice, check_int

INPUT_REPRESENTATIONS: tuple[str, ...] = ("waveform", "spectrogram")


class CNNEncoder(nn.Module):
    """Encode each station's 3-component record into one embedding.

    The same encoder (shared weights) is applied to every station of an
    event, so station embeddings are comparable and can serve as graph node
    features in later stages. The encoder is expected to learn temporal /
    time-frequency characteristics relevant to source discrimination.

    Args:
        embedding_dim: Size ``D`` of each station embedding.
        in_channels: Number of components ``C`` (3 for Z/N/E).
        input_representation: ``"waveform"`` for ``[B, N, C, T]`` input or
            ``"spectrogram"`` for ``[B, N, C, F, T']`` input (Phase 4).
        pool_time: If True, pool over time and return one vector per station.
            If False, keep a temporal axis for spatio-temporal models (Phase 7).

    Expected input:
        waveforms: float tensor ``[B, N, C, T]`` (or ``[B, N, C, F, T']``).
        station_mask: optional bool tensor ``[B, N]``; True = station present.

    Expected output:
        ``[B, N, D]`` if ``pool_time`` else ``[B, N, T', D]``.
        Embeddings of padded stations (mask False) must be zero.

    TODO(Phase 3): choose and implement the architecture (depth, kernel sizes,
        normalization layers, temporal pooling), justify it experimentally and
        record it in the Decision Log.
    TODO(Phase 4): support spectrogram input.
    """

    def __init__(
        self,
        embedding_dim: int,
        in_channels: int = 3,
        *,
        input_representation: str = "waveform",
        pool_time: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = check_int("embedding_dim", embedding_dim)
        self.in_channels = check_int("in_channels", in_channels)
        self.input_representation = check_choice(
            "input_representation", input_representation, INPUT_REPRESENTATIONS
        )
        self.pool_time = bool(pool_time)

    def forward(
        self,
        waveforms: torch.Tensor,
        station_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotYetImplementedError("CNNEncoder.forward", phase=3)
