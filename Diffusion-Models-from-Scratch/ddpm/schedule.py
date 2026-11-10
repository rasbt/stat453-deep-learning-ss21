"""Noise schedules for the forward diffusion process q(x_t | x_0).

A schedule is defined by its betas. Everything else the training loss and the
sampler need (alpha_bar, its square roots, the posterior variance) is derived
once here and registered as buffers, so a single .to(device) moves it all.
"""
import math

import torch
import torch.nn as nn


def linear_betas(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Ho et al. (2020): betas grow linearly from 1e-4 to 0.02."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_betas(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Nichol & Dhariwal (2021): alpha_bar follows a squared cosine, which adds
    noise more slowly at the start and avoids the near-destroyed late steps of
    the linear schedule on small images."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=max_beta)


SCHEDULES = {
    'linear': linear_betas,
    'cosine': cosine_betas,
}


class NoiseSchedule(nn.Module):
    """Holds the per-timestep constants of the diffusion process as buffers."""

    def __init__(self, betas: torch.Tensor):
        super().__init__()
        betas = betas.double()
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

        f = lambda x: x.float()
        self.register_buffer('betas', f(betas))
        self.register_buffer('alphas', f(alphas))
        self.register_buffer('alpha_bar', f(alpha_bar))
        self.register_buffer('sqrt_alpha_bar', f(alpha_bar.sqrt()))
        self.register_buffer('sqrt_one_minus_alpha_bar', f((1 - alpha_bar).sqrt()))
        self.register_buffer('sqrt_recip_alphas', f(alphas.rsqrt()))
        # Variance of q(x_{t-1} | x_t, x_0), the "lower bound" choice in the paper.
        self.register_buffer('posterior_variance', f(betas * (1 - alpha_bar_prev) / (1 - alpha_bar)))

    @property
    def num_timesteps(self) -> int:
        return self.betas.shape[0]

    def extract(self, name: str, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
        """Gather buffer `name` at timesteps t and reshape to broadcast over an image batch."""
        out = getattr(self, name)[t]
        return out.view(-1, *([1] * (ndim - 1)))


def make_schedule(name: str, timesteps: int, **kwargs) -> NoiseSchedule:
    if name not in SCHEDULES:
        raise ValueError(f"Unknown schedule '{name}'. Choose from {list(SCHEDULES)}")
    return NoiseSchedule(SCHEDULES[name](timesteps, **kwargs))
