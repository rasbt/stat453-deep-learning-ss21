"""Building blocks of the UNet, following Ho et al. (2020) / Nichol & Dhariwal (2021).

Every block takes (x, temb) so the UNet can run them uniformly; blocks that do not
use the time embedding simply ignore it."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features of the integer timesteps t, shape (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal features -> 2-layer MLP. Shared by every ResBlock, which adds its
    own learned projection on top."""

    def __init__(self, base_channels: int, time_embed_dim: int):
        super().__init__()
        self.base_channels = base_channels
        self.mlp = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.base_channels))


class ResBlock(nn.Module):
    """GN -> SiLU -> conv, add projected time embedding, GN -> SiLU -> dropout -> conv,
    plus a skip (1x1 conv when the channel count changes)."""

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int,
                 num_groups: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(temb))[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention over spatial positions with a residual."""

    def __init__(self, channels: int, num_heads: int, num_groups: int):
        super().__init__()
        assert channels % num_heads == 0, f'{channels} channels not divisible by {num_heads} heads'
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        b, c, h, w = x.shape
        y = rearrange(self.norm(x), 'b c h w -> b (h w) c')
        q, k, v = rearrange(self.qkv(y), 'b L (K H d) -> K b H L d', K=3, H=self.num_heads)
        y = F.scaled_dot_product_attention(q, k, v)
        y = self.proj(rearrange(y, 'b H L d -> b L (H d)'))
        return x + rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)


class Downsample(nn.Module):
    """Stride-2 3x3 conv; channel count unchanged."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour 2x then 3x3 conv; avoids transposed-conv checkerboarding."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))
