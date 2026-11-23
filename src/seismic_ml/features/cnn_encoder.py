"""Station-level CNN encoder — Stage A (Phase 3 & Phase 4 Dual-Stream Upgrade).

Encodes each station's 3-component record into a learned feature embedding:
- ``"waveform"``: Shared 1D Temporal Convolutional Network (TCN) with 4 dilated residual blocks.
- ``"spectrogram"``: Shared 2D-CNN over STFT log-power spectrograms (1.0–16.0 Hz @ 50 Hz)
  to resolve quarry ripple-fire spectral scalloping.
- ``"dual"``: Parallel 1D temporal + 2D STFT spectral streams fused via scaled dot-product
  cross-attention (Monograph §10.2).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from seismic_ml.utils.validation import check_choice, check_int

INPUT_REPRESENTATIONS: tuple[str, ...] = ("waveform", "spectrogram", "dual")


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

        if out.shape[-1] != residual.shape[-1]:
            out = out[..., : residual.shape[-1]]

        return self.relu(out + residual)


class SpectrogramBranch2D(nn.Module):
    """2D-CNN branch operating on 3-component STFT log-power spectrograms."""

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 64,
        *,
        sample_rate_hz: float = 50.0,
        n_fft: int = 128,
        hop_length: int = 64,
        freq_min_hz: float = 1.0,
        freq_max_hz: float = 16.0,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.sample_rate_hz = float(sample_rate_hz)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.freq_min_hz = float(freq_min_hz)
        self.freq_max_hz = float(freq_max_hz)

        self.register_buffer("hann_window", torch.hann_window(self.n_fft), persistent=False)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def compute_stft_log_power(self, x_1d: torch.Tensor) -> torch.Tensor:
        """Convert ``[M, C, T]`` waveforms to ``[M, C, F_band, T']`` log-power spectrograms."""
        m, c, t = x_1d.shape
        n_fft = min(self.n_fft, t)
        hop = min(self.hop_length, max(1, n_fft // 2))
        win = self.hann_window[:n_fft] if n_fft < self.n_fft else self.hann_window

        flat = x_1d.reshape(m * c, t)
        stft = torch.stft(
            flat,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=win.to(device=flat.device, dtype=flat.dtype),
            return_complex=True,
            center=True,
        )
        power = stft.abs().pow(2)
        log_power = torch.log1p(power)  # [M*C, F, T']

        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate_hz, device=flat.device)
        band_mask = (freqs >= self.freq_min_hz) & (freqs <= self.freq_max_hz)
        if band_mask.any():
            log_power = log_power[:, band_mask, :]

        _, f_bins, t_frames = log_power.shape
        return log_power.view(m, c, f_bins, t_frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on ``[M, C, T]`` waveforms or ``[M, C, F, T']`` spectrograms."""
        spec = self.compute_stft_log_power(x) if x.ndim == 3 else x
        feat = self.net(spec)
        return self.pool(feat).flatten(1)  # [M, out_dim]


class CrossAttentionFusion(nn.Module):
    """Scaled dot-product cross-attention fusing 1D temporal and 2D spectral embeddings."""

    def __init__(self, temp_dim: int, spec_dim: int, out_dim: int) -> None:
        super().__init__()
        self.scale = math.sqrt(float(temp_dim))
        self.q_proj = nn.Linear(temp_dim, temp_dim, bias=False)
        self.k_proj = nn.Linear(spec_dim, temp_dim, bias=False)
        self.v_proj = nn.Linear(spec_dim, temp_dim, bias=False)
        self.out_proj = nn.Linear(temp_dim + spec_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h_temp: torch.Tensor, h_spec: torch.Tensor) -> torch.Tensor:
        """Fuse ``[M, temp_dim]`` and ``[M, spec_dim]`` into ``[M, out_dim]``."""
        q = self.q_proj(h_temp)
        k = self.k_proj(h_spec)
        v = self.v_proj(h_spec)
        gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / self.scale)
        h_cross = gate * v + h_temp
        fused = torch.cat([h_cross, h_spec], dim=-1)
        return self.norm(self.out_proj(fused))


class CNNEncoder(nn.Module):
    """Encode each station's 3-component record into one embedding.

    Supports ``"waveform"`` (1D dilated TCN), ``"spectrogram"`` (2D STFT CNN),
    and ``"dual"`` (parallel 1D + 2D STFT streams with cross-attention fusion).
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

        if self.input_representation in ("waveform", "dual"):
            self.init_conv = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            )
            blocks = [
                DilatedResidualBlock1D(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
            self.blocks = nn.Sequential(*blocks)
            self.global_pool = nn.AdaptiveAvgPool1d(1)

            if self.input_representation == "waveform":
                self.proj = nn.Linear(hidden_channels, self.embedding_dim)
            else:
                branch_dim = max(32, self.embedding_dim // 2)
                self.temp_proj = nn.Linear(hidden_channels, branch_dim)
                self.spec_branch = SpectrogramBranch2D(
                    in_channels=self.in_channels,
                    out_dim=branch_dim,
                    dropout=min(dropout, 0.2),
                )
                self.fusion = CrossAttentionFusion(branch_dim, branch_dim, self.embedding_dim)
        else:
            # Pure 2D spectrogram branch
            self.spec_branch = SpectrogramBranch2D(
                in_channels=self.in_channels,
                out_dim=hidden_channels,
                dropout=min(dropout, 0.2),
            )
            self.proj = nn.Linear(hidden_channels, self.embedding_dim)

    def _encode_valid_traces(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``[M, C, T]`` (or ``[M, C, F, T']``) traces into ``[M, D]``."""
        if self.input_representation == "waveform":
            h = self.init_conv(x)
            h = self.blocks(h)
            h = self.global_pool(h).squeeze(-1)
            return self.proj(h)

        if self.input_representation == "spectrogram":
            h_spec = self.spec_branch(x)
            return self.proj(h_spec)

        # Dual-stream hybrid: 1D temporal + 2D STFT spectral + cross-attention fusion
        h_1d = self.global_pool(self.blocks(self.init_conv(x))).squeeze(-1)
        h_temp = F.relu(self.temp_proj(h_1d))
        h_spec = self.spec_branch(x)
        return self.fusion(h_temp, h_spec)

    def forward(
        self,
        waveforms: torch.Tensor,
        station_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            waveforms: Float tensor ``[B, N, C, T]`` (or ``[B, N, C, F, T']`` for
                precomputed spectrograms).
            station_mask: Optional boolean tensor ``[B, N]`` (True = valid station).

        Returns:
            ``[B, N, D]`` (if ``pool_time=True``) or ``[B, N, T, D]`` (if ``pool_time=False``).
        """
        if waveforms.ndim not in (4, 5):
            raise ValueError(f"Expected 4D [B, N, C, T] or 5D [B, N, C, F, T'], got {waveforms.shape}")
        if waveforms.ndim == 5 and self.input_representation != "spectrogram":
            raise ValueError(
                f"5D spectrogram tensor requires input_representation='spectrogram', got {self.input_representation!r}."
            )

        b, n, c = waveforms.shape[:3]
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")

        if not self.pool_time:
            if self.input_representation != "waveform":
                raise ValueError("pool_time=False is only supported for input_representation='waveform'.")
            t = waveforms.shape[3]
            x = waveforms.view(b * n, c, t)
            x = self.blocks(self.init_conv(x)).transpose(1, 2)
            out = self.proj(x).view(b, n, t, self.embedding_dim)
            if station_mask is not None:
                out = out * station_mask.unsqueeze(-1).unsqueeze(-1).to(out.dtype)
            return out

        # Compact to only valid stations when station_mask is given (saves ~5-8x VRAM on padded batches)
        if station_mask is not None:
            valid_x = waveforms[station_mask]  # [N_valid, C, ...]
            out = waveforms.new_zeros((b, n, self.embedding_dim))
            if valid_x.shape[0] > 0:
                valid_emb = self._encode_valid_traces(valid_x)
                out[station_mask] = valid_emb.to(out.dtype)
            return out

        flat_x = waveforms.view(b * n, *waveforms.shape[2:])
        return self._encode_valid_traces(flat_x).view(b, n, self.embedding_dim)
