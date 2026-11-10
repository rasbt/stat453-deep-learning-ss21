"""The diffusion process itself: forward noising, the training objective and two
samplers (DDPM ancestral, DDIM). The network is treated as a black box epsilon-predictor."""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedule import NoiseSchedule


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, schedule: NoiseSchedule, variance: str = 'beta'):
        """
        variance: which sigma_t^2 to use in the reverse step.
            'beta'      -> beta_t                      (upper bound, what the original code used)
            'posterior' -> beta_t (1-abar_{t-1})/(1-abar_t)  (lower bound)
        Ho et al. report both give similar sample quality.
        """
        super().__init__()
        if variance not in ('beta', 'posterior'):
            raise ValueError("variance must be 'beta' or 'posterior'")
        self.model = model
        self.schedule = schedule
        self.variance = variance

    @property
    def num_timesteps(self) -> int:
        return self.schedule.num_timesteps

    # ---- forward process -------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I) in closed form."""
        if noise is None:
            noise = torch.randn_like(x0)
        s = self.schedule
        return s.extract('sqrt_alpha_bar', t) * x0 + s.extract('sqrt_one_minus_alpha_bar', t) * noise

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """L_simple: MSE between the true noise and the model's prediction at a random t."""
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(self.model(x_t, t), noise)

    # ---- reverse process -------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One step x_t -> x_{t-1} of ancestral sampling (Algorithm 2 in the paper)."""
        s = self.schedule
        tt = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps = self.model(x_t, tt)
        mean = s.sqrt_recip_alphas[t] * (x_t - s.betas[t] / s.sqrt_one_minus_alpha_bar[t] * eps)
        if t == 0:
            return mean
        var = s.betas[t] if self.variance == 'beta' else s.posterior_variance[t]
        return mean + var.sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self,
               shape: Sequence[int],
               device: torch.device,
               snapshot_steps: Sequence[int] = (),
               noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate a batch of images from pure noise (or from the given starting noise).

        Returns the final x_0 and, for each t in snapshot_steps (in descending
        order), the batch as it looked right after the step at that t."""
        snapshot_steps = set(snapshot_steps)
        x = torch.randn(*shape, device=device) if noise is None else noise.to(device).clone()
        snapshots = []
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t)
            if t in snapshot_steps:
                snapshots.append(x.clone())
        return x, snapshots

    # ---- DDIM (Song, Meng & Ermon, 2021) ------------------------------------
    def ddim_timesteps(self, num_steps: int) -> List[int]:
        """num_steps timesteps spread evenly over [0, T-1], descending, always ending at 0."""
        if not 1 <= num_steps <= self.num_timesteps:
            raise ValueError(f'num_steps must be in [1, {self.num_timesteps}]')
        ts = torch.linspace(0, self.num_timesteps - 1, num_steps).round().long().unique()
        return ts.flip(0).tolist()

    @torch.no_grad()
    def ddim_step(self, x_t: torch.Tensor, t: int, t_prev: int, eta: float = 0.0,
                  clip_denoised: bool = True) -> torch.Tensor:
        """One DDIM step from timestep t to t_prev (t_prev = -1 means x_0).

        The model's eps gives a point estimate of x_0; the step then re-noises that
        estimate to level t_prev along a direction set by eps. With eta=0 this is
        deterministic; eta=1 recovers the DDPM posterior variance.

        clip_denoised clamps the x_0 estimate to [-1, 1] and recomputes eps from it.
        Near t = T the estimate divides by sqrt(alpha_bar_t), which is 0.006 for the
        linear schedule but 5e-5 for the cosine one, so without clipping a small eps
        error becomes a huge x_0 and the cosine model samples pure noise."""
        s = self.schedule
        ab_t = s.alpha_bar[t]
        ab_prev = s.alpha_bar[t_prev] if t_prev >= 0 else torch.ones((), device=x_t.device)
        tt = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps = self.model(x_t, tt)

        x0_pred = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clip_denoised:
            x0_pred = x0_pred.clamp(-1, 1)
            eps = (x_t - ab_t.sqrt() * x0_pred) / (1 - ab_t).sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
        direction = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        x_prev = ab_prev.sqrt() * x0_pred + direction
        if eta > 0 and t_prev >= 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
        return x_prev

    @torch.no_grad()
    def ddim_sample(self,
                    shape: Sequence[int],
                    device: torch.device,
                    num_steps: int = 50,
                    eta: float = 0.0,
                    snapshot_steps: Sequence[int] = (),
                    noise: Optional[torch.Tensor] = None,
                    clip_denoised: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate a batch with DDIM in num_steps model evaluations instead of T.

        snapshot_steps are given in full-schedule timesteps; each is mapped to the
        nearest timestep actually visited, and the state right after that step is kept."""
        ts = self.ddim_timesteps(num_steps)
        visited = torch.tensor(ts)
        snap_at = set()
        for s_ in snapshot_steps:
            snap_at.add(ts[int((visited - s_).abs().argmin())])

        x = torch.randn(*shape, device=device) if noise is None else noise.to(device).clone()
        snapshots = []
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            x = self.ddim_step(x, t, t_prev, eta, clip_denoised)
            if t in snap_at:
                snapshots.append(x.clone())
        return x, snapshots
