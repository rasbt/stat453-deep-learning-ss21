"""DDPM UNet, specified by a base width, per-resolution channel multipliers, a number
of ResBlocks per resolution, and the resolutions that get self-attention."""
from typing import Sequence

import torch
import torch.nn as nn

from .blocks import AttentionBlock, Downsample, ResBlock, TimestepEmbedding, Upsample


class UNET(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 image_size: int,
                 base_channels: int = 64,
                 channel_mults: Sequence[int] = (1, 2, 2, 2),
                 num_res_blocks: int = 2,
                 attention_resolutions: Sequence[int] = (16,),
                 num_heads: int = 4,
                 num_groups: int = 32,
                 dropout: float = 0.0,
                 time_embed_dim: int = None):
        super().__init__()
        time_embed_dim = time_embed_dim or base_channels * 4
        attention_resolutions = set(attention_resolutions)

        def res_block(cin, cout):
            return ResBlock(cin, cout, time_embed_dim, num_groups, dropout)

        def attn(c):
            return AttentionBlock(c, num_heads, num_groups)

        self.time_embed = TimestepEmbedding(base_channels, time_embed_dim)
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---- encoder: each entry of self.down is a ModuleList run in order and whose output is a skip
        self.down = nn.ModuleList()
        skip_channels = [base_channels]
        ch, res = base_channels, image_size
        for level, mult in enumerate(channel_mults):
            out = base_channels * mult
            for _ in range(num_res_blocks):
                blocks = [res_block(ch, out)]
                ch = out
                if res in attention_resolutions:
                    blocks.append(attn(ch))
                self.down.append(nn.ModuleList(blocks))
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)
                res //= 2

        self.middle = nn.ModuleList([res_block(ch, ch), attn(ch), res_block(ch, ch)])

        # ---- decoder: one extra ResBlock per level consumes the Downsample skip
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out = base_channels * mult
            for i in range(num_res_blocks + 1):
                blocks = [res_block(ch + skip_channels.pop(), out)]
                ch = out
                if res in attention_resolutions:
                    blocks.append(attn(ch))
                if level != 0 and i == num_res_blocks:
                    blocks.append(Upsample(ch))
                    res *= 2
                self.up.append(nn.ModuleList(blocks))
        assert not skip_channels, 'encoder/decoder skip mismatch'

        self.out = nn.Sequential(nn.GroupNorm(num_groups, ch), nn.SiLU(),
                                 nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))

    @staticmethod
    def _run(blocks: nn.ModuleList, h: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            h = block(h, temb)
        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(t)
        h = self.in_conv(x)
        skips = [h]
        for blocks in self.down:
            h = self._run(blocks, h, temb)
            skips.append(h)
        h = self._run(self.middle, h, temb)
        for blocks in self.up:
            h = self._run(blocks, torch.cat([h, skips.pop()], dim=1), temb)
        return self.out(h)
