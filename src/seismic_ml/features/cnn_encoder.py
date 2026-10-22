"""Station-level CNN encoder — Stage A (Phase 3).

Encodes each station's 3-component record into a learned feature embedding.
Uses a shared 1D Temporal Convolutional Network (TCN) with dilated residual blocks.
"""

from __future__ import annotations

import torch
from torch import nn

from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.utils.validation import check_choice, check_int

INPUT_REPRESENTATIONS: tuple[str, ...] = ("waveform", "spectrogram")


class DilatedResidualBlock1D(nn.Module):
    """1D residual convolutional block with dilation for temporal sequence modeling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Match exact temporal length if padding creates a 1-sample mismatch
        if out.shape[-1] != residual.shape[-1]:
            out = out[..., :residual.shape[-1]]

        return self.relu(out + residual)


class CNNEncoder(nn.Module):
    """Encode each station's 3-component record into one embedding.

    The same encoder (shared weights) is applied to every station of an
    event, so station embeddings are comparable and can serve as graph node
    features in later stages.

    Args:
        embedding_dim: Size ``D`` of each station embedding.
        in_channels: Number of components ``C`` (3 for Z/N/E).
        input_representation: ``"waveform"`` for ``[B, N, C, T]`` input or
            ``"spectrogram"`` for ``[B, N, C, F, T']`` input (Phase 4).
        pool_time: If True, pool over time and return one vector per station.
            If False, keep a temporal axis for spatio-temporal models (Phase 7).
        hidden_channels: Base channel width for the internal convolutions.
        kernel_size: 1D convolution kernel size (default 7).
        num_blocks: Number of dilated residual blocks (dilations: 1, 2, 4, 8...).
        dropout: Dropout rate inside residual blocks.
    """

    def __init__(
        self,
        embedding_dim: int,
        in_channels: int = 3,
        *,
        input_representation: str = "waveform",
        pool_time: bool = True,
        hidden_channels: int = 64,
        kernel_size: int = 7,
        num_blocks: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding_dim = check_int("embedding_dim", embedding_dim)
        self.in_channels = check_int("in_channels", in_channels)
        self.input_representation = check_choice(
            "input_representation", input_representation, INPUT_REPRESENTATIONS
        )
        self.pool_time = bool(pool_time)

        if self.input_representation == "spectrogram":
            # Phase 4 hook
            self.network = None
            return

        # Initial projection from in_channels (e.g. 3) to hidden_channels
        self.init_conv = nn.Sequential(
            nn.Conv1d(self.in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        # Dilated residual blocks with exponentially increasing receptive fields
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(
                DilatedResidualBlock1D(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_channels, self.embedding_dim)

    def forward(
        self,
        waveforms: torch.Tensor,
        station_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            waveforms: Float tensor ``[B, N, C, T]``.
            station_mask: Optional boolean tensor ``[B, N]`` (True = valid station).

        Returns:
            ``[B, N, D]`` (if ``pool_time=True``) or ``[B, N, T, D]`` (if ``pool_time=False``).
        """
        if self.input_representation == "spectrogram":
            raise NotYetImplementedError("CNNEncoder.forward(spectrogram)", phase=4)

        if waveforms.ndim != 4:
            raise ValueError(f"Expected 4D waveforms [B, N, C, T], got {waveforms.shape}")

        b, n, c, t = waveforms.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")

        # Flatten batch and station dimensions to process stations in parallel with shared weights
        x = waveforms.view(b * n, c, t)

        x = self.init_conv(x)
        x = self.blocks(x)

        if self.pool_time:
            # Pool across temporal dimension T
            x = self.global_pool(x).squeeze(-1)  # [B*N, hidden_channels]
            out = self.proj(x)  # [B*N, embedding_dim]
            out = out.view(b, n, self.embedding_dim)
        else:
            # Keep temporal dimension
            x = x.transpose(1, 2)  # [B*N, T, hidden_channels]
            out = self.proj(x)  # [B*N, T, embedding_dim]
            out = out.view(b, n, t, self.embedding_dim)

        # Zero out embeddings for padded / masked stations
        if station_mask is not None:
            if self.pool_time:
                mask_expanded = station_mask.unsqueeze(-1).to(out.dtype)
                out = out * mask_expanded
            else:
                mask_expanded = station_mask.unsqueeze(-1).unsqueeze(-1).to(out.dtype)
                out = out * mask_expanded

        return out
