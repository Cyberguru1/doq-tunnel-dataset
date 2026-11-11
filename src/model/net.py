"""Proposed detector: Time-Distributed Dilated Conv -> BiLSTM -> Residual fusion.

Implements the architecture from the proposal (section 5.3, "Technique
Modification and Enhancement"):

  1. A *time-distributed* dilated convolutional front-end: every time step
     (one flow record / one 10 s window of a persistent session) is a vector of
     F flow statistics. The same stack of 1-D dilated convolutions is applied
     to every step independently (shared weights == Keras TimeDistributed),
     with exponentially growing dilation so the receptive field covers the
     whole feature vector in three layers.
  2. A BiLSTM over the sequence of per-step embeddings to model how the
     session evolves (bidirectional temporal feature extraction).
  3. The *enhanced* variant adds residual connections that fuse the
     intermediate outputs of the earlier conv blocks (multi-scale skip paths)
     and a global skip from the conv embedding straight into the classifier,
     so deep features propagate without being forced through the recurrent
     bottleneck.

The model works for any sequence length T >= 1; with T == 1 it degenerates to a
per-flow classifier, which is how the published CIRA-CIC-DoHBrw-2020 baselines
operate.

Input tensor:  (batch, T, F) float
Output tensor: (batch, n_classes) logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    """Conv1d(k=3, dilation=d) -> BatchNorm -> GELU -> Dropout, 'same' padding."""

    def __init__(self, c_in: int, c_out: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation  # (k-1)/2 * dilation with k=3
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=3, dilation=dilation, padding=pad)
        self.bn = nn.GroupNorm(1, c_out)  # batch-independent (BatchNorm gave train/eval mismatch with PLE tokens)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: (N, C, L)
        return self.drop(F.gelu(self.bn(self.conv(x))))


class PiecewiseLinearEmbedding(nn.Module):
    """Per-feature piecewise-linear encoding on training-set quantile bins
    (Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep
    Learning", NeurIPS 2022) followed by a per-feature linear map to d_token.

    Turns each scalar flow statistic into a d_token vector, so the dilated conv
    stack sees a sequence of F feature *tokens* instead of a raw 1-channel signal.
    Input (N, F) -> output (N, d_token, F).
    """

    def __init__(self, bin_edges: torch.Tensor, d_token: int):
        super().__init__()
        self.register_buffer("edges", bin_edges.float())        # (F, K+1)
        F_, K = bin_edges.shape[0], bin_edges.shape[1] - 1
        self.weight = nn.Parameter(torch.randn(F_, K, d_token) * 0.05)
        self.bias = nn.Parameter(torch.zeros(F_, d_token))

    def forward(self, x):  # (N, F)
        lo, hi = self.edges[:, :-1], self.edges[:, 1:]
        t = ((x.unsqueeze(-1) - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)  # (N, F, K)
        e = torch.einsum("nfk,fkd->nfd", t, self.weight) + self.bias           # (N, F, d)
        return e.transpose(1, 2)                                               # (N, d, F)


class TimeDistributedDilatedConv(nn.Module):
    """Shared dilated-conv stack applied independently to every time step.

    Each step's F features are treated as a 1-D signal with one channel.
    Returns a per-step embedding of size `embed_dim`.

    residual=True  -> enhanced variant: outputs of all blocks are fused by
                      1x1 projections and summed (multi-scale residual fusion)
                      and a skip from block 1 is added to block 3.
    residual=False -> plain sequential stack (base variant for ablation).
    """

    def __init__(self, n_features: int, channels: int = 32, dilations=(1, 2, 4),
                 embed_dim: int = 128, dropout: float = 0.1, residual: bool = True,
                 feat_embed: nn.Module | None = None, token_dim: int = 1):
        super().__init__()
        self.residual = residual
        self.feat_embed = feat_embed
        blocks, fuse = [], []
        c_in = token_dim if feat_embed is not None else 1
        for d in dilations:
            blocks.append(DilatedConvBlock(c_in, channels, d, dropout))
            fuse.append(nn.Conv1d(channels, channels, kernel_size=1))
            c_in = channels
        self.blocks = nn.ModuleList(blocks)
        self.fuse = nn.ModuleList(fuse)
        self.proj = nn.Sequential(
            nn.Flatten(),  # (N, C*F)
            nn.Linear(channels * n_features, embed_dim),
        )
        # enhanced variant: residual input skip (raw feature vector -> embedding) so the
        # per-step embedding is at least as expressive as a linear/MLP encoder.
        self.in_skip = nn.Linear(n_features, embed_dim) if residual else None
        self.post = nn.Sequential(nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):  # x: (B, T, F)
        B, T, Fd = x.shape
        flat = x.reshape(B * T, Fd)          # time-distributed: fold T into batch
        h = self.feat_embed(flat) if self.feat_embed is not None else flat.unsqueeze(1)
        outs = []
        for i, blk in enumerate(self.blocks):
            h_new = blk(h)
            if self.residual and i > 0:
                h_new = h_new + h            # residual link between consecutive blocks
            h = h_new
            outs.append(h)
        if self.residual:
            h = sum(f(o) for f, o in zip(self.fuse, outs))  # multi-scale fusion
        e = self.proj(h)                     # (B*T, embed_dim)
        if self.in_skip is not None:
            e = e + self.in_skip(flat)
        e = self.post(e)
        return e.reshape(B, T, -1)


class AttentionPool(nn.Module):
    """Learned soft attention over time steps (mask-aware)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h, mask=None):  # h: (B, T, D), mask: (B, T) bool, True=valid
        s = self.score(h).squeeze(-1)  # (B, T)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        return (a * h).sum(1)


class DilatedResBiLSTM(nn.Module):
    """TD-DilatedConv -> BiLSTM -> (residual fusion) -> classifier."""

    def __init__(self, n_features: int, n_classes: int, channels: int = 32,
                 dilations=(1, 2, 4), embed_dim: int = 128, lstm_hidden: int = 96,
                 lstm_layers: int = 1, dropout: float = 0.2, residual: bool = True,
                 bin_edges: torch.Tensor | None = None, token_dim: int = 8):
        super().__init__()
        self.residual = residual
        fe = PiecewiseLinearEmbedding(bin_edges, token_dim) if bin_edges is not None else None
        self.frontend = TimeDistributedDilatedConv(
            n_features, channels, dilations, embed_dim, dropout, residual, fe, token_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.pool_rnn = AttentionPool(2 * lstm_hidden)
        self.pool_conv = AttentionPool(embed_dim)
        head_in = 2 * lstm_hidden + (embed_dim if residual else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, mask=None):
        """x: (B, T, F); mask: optional (B, T) bool, True for real steps."""
        e = self.frontend(x)                 # (B, T, E)
        h, _ = self.lstm(e)                  # (B, T, 2H)
        z = self.pool_rnn(h, mask)
        if self.residual:
            z = torch.cat([z, self.pool_conv(e, mask)], dim=1)  # global skip path
        return self.head(z)

    @torch.no_grad()
    def embed(self, x, mask=None):
        e = self.frontend(x)
        h, _ = self.lstm(e)
        z = self.pool_rnn(h, mask)
        if self.residual:
            z = torch.cat([z, self.pool_conv(e, mask)], dim=1)
        return z


def build_model(n_features: int, n_classes: int, variant: str = "enhanced", **kw) -> nn.Module:
    """variant: 'enhanced' (residual fusion, the proposed model) or 'base'."""
    return DilatedResBiLSTM(n_features, n_classes, residual=(variant == "enhanced"), **kw)


def quantile_bin_edges(X: "np.ndarray", n_bins: int = 16) -> torch.Tensor:
    """Per-feature quantile bin edges from (already scaled) training rows X (N, F)."""
    import numpy as np
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(X, qs, axis=0).T  # (F, K+1)
    edges = np.maximum.accumulate(edges, axis=1)  # monotone
    return torch.tensor(edges, dtype=torch.float32)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    import numpy as np
    m = build_model(28, 4, bin_edges=quantile_bin_edges(np.random.randn(500, 28), 16))
    print(m(torch.randn(5, 8, 28)).shape, count_params(m), "params (PLE)")
    m = build_model(28, 4)
    x = torch.randn(5, 8, 28)
    print(m(x).shape, count_params(m), "params")
    m1 = build_model(28, 2, variant="base")
    print(m1(torch.randn(3, 1, 28)).shape, count_params(m1), "params (base, T=1)")
