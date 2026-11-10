"""The original UNet from the upstream repo, kept so old checkpoints still load and
so the two architectures can be compared on equal footing. Select with model.arch: legacy.

Shortcuts relative to the paper: the time embedding is sliced and added with no learned
projection, attention has no norm or residual, and the resampling convs change the
channel count."""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LegacyResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


class LegacyAttention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False,
                                           dropout_p=self.dropout_prob if self.training else 0.0)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')


class LegacySinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.register_buffer('embeddings', embeddings)

    def forward(self, t):
        return self.embeddings[t][:, :, None, None]


class LegacyUnetLayer(nn.Module):
    def __init__(self, upscale: bool, attention: bool, num_groups: int,
                 dropout_prob: float, num_heads: int, C: int):
        super().__init__()
        self.ResBlock1 = LegacyResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = LegacyResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C // 2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C * 2, kernel_size=3, stride=2, padding=1)
        self.attention_layer = LegacyAttention(C, num_heads=num_heads, dropout_prob=dropout_prob) if attention else None

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if self.attention_layer is not None:
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x


class LegacyUNET(nn.Module):
    def __init__(self,
                 channels: List[int] = (64, 128, 256, 512, 512, 384),
                 attentions: List[bool] = (False, True, False, False, False, True),
                 upscales: List[bool] = (False, False, False, True, True, True),
                 num_groups: int = 32,
                 dropout_prob: float = 0.0,
                 num_heads: int = 8,
                 input_channels: int = 1,
                 output_channels: int = 1,
                 time_steps: int = 1000):
        super().__init__()
        assert len(channels) == len(attentions) == len(upscales), 'per-layer lists must match'
        assert len(channels) % 2 == 0, 'need an equal number of encoder and decoder layers'
        half = len(channels) // 2

        self.shallow_conv = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1)
        self.embeddings = LegacySinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(channels))

        def make(i):
            return LegacyUnetLayer(upscale=upscales[i], attention=attentions[i], num_groups=num_groups,
                                   dropout_prob=dropout_prob, num_heads=num_heads, C=channels[i])
        self.encoder = nn.ModuleList([make(i) for i in range(half)])
        self.decoder = nn.ModuleList([make(i) for i in range(half, len(channels))])

        out_channels = (channels[-1] // 2) + channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels // 2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        embeddings = self.embeddings(t)
        x = self.shallow_conv(x)
        skips = []
        for layer in self.encoder:
            x, skip = layer(x, embeddings)
            skips.append(skip)
        for layer, skip in zip(self.decoder, reversed(skips)):
            x, _ = layer(x, embeddings)
            x = torch.cat((x, skip), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))
