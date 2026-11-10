"""Fast CPU tests: python -m pytest tests -q"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddpm import GaussianDiffusion, UNET, make_schedule
from ddpm.checkpoint import _remap_legacy_keys
from ddpm.model import LegacyUNET


@pytest.fixture(params=['linear', 'cosine'])
def schedule(request):
    return make_schedule(request.param, 1000)


def test_schedule_values_are_valid(schedule):
    assert schedule.num_timesteps == 1000
    assert torch.all(schedule.betas > 0) and torch.all(schedule.betas < 1)
    ab = schedule.alpha_bar
    assert torch.all(ab[1:] <= ab[:-1]), 'alpha_bar must be non-increasing'
    assert ab[0] > 0.99 and ab[-1] < 0.01, 'alpha_bar should run from ~1 down to ~0'
    assert torch.allclose(schedule.sqrt_alpha_bar ** 2 + schedule.sqrt_one_minus_alpha_bar ** 2, torch.ones(1000), atol=1e-5)


def test_cosine_starts_slower_than_linear():
    lin, cos = make_schedule('linear', 1000), make_schedule('cosine', 1000)
    assert cos.alpha_bar[200] > lin.alpha_bar[200], 'cosine keeps more signal early on'


class ZeroModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_q_sample_endpoints():
    diff = GaussianDiffusion(ZeroModel(), make_schedule('linear', 1000))
    x0 = torch.randn(4, 1, 8, 8)
    noise = torch.randn_like(x0)
    t0 = torch.zeros(4, dtype=torch.long)
    # beta_0 = 1e-4, so sqrt(1 - alpha_bar_0) = 0.01: a 1% noise contribution remains at t=0
    assert torch.allclose(diff.q_sample(x0, t0, noise), x0, atol=0.06)
    tT = torch.full((4,), 999, dtype=torch.long)
    xT = diff.q_sample(x0, tT, noise)
    assert torch.allclose(xT, noise, atol=0.1), 'at t=T the sample should be almost pure noise'


def test_ddim_timesteps():
    diff = GaussianDiffusion(ZeroModel(), make_schedule('linear', 1000))
    ts = diff.ddim_timesteps(50)
    assert ts[0] == 999 and ts[-1] == 0 and len(ts) == 50
    assert all(a > b for a, b in zip(ts, ts[1:]))


def test_ddim_eta1_matches_ddpm_posterior_variance():
    s = make_schedule('linear', 1000)
    t, tp = 500, 499
    ab_t, ab_p = s.alpha_bar[t], s.alpha_bar[tp]
    sigma2 = (((1 - ab_p) / (1 - ab_t)) * (1 - ab_t / ab_p))
    assert torch.allclose(sigma2, s.posterior_variance[t], atol=1e-6)


def test_samplers_run_with_fixed_noise():
    torch.manual_seed(0)
    diff = GaussianDiffusion(ZeroModel(), make_schedule('linear', 20))
    noise = torch.randn(2, 1, 8, 8)
    x_ddpm, snaps = diff.sample((2, 1, 8, 8), torch.device('cpu'), snapshot_steps=[19, 0], noise=noise)
    x_ddim, _ = diff.ddim_sample((2, 1, 8, 8), torch.device('cpu'), num_steps=5, noise=noise)
    assert x_ddpm.shape == x_ddim.shape == (2, 1, 8, 8) and len(snaps) == 2
    # with a zero-noise model, deterministic DDIM should return noise / sqrt(alpha_bar_T) rescaled to x0 estimate: finite
    assert torch.isfinite(x_ddim).all()


@pytest.mark.parametrize('channels', [1, 3])
def test_unet_shapes(channels):
    model = UNET(in_channels=channels, out_channels=channels, image_size=32, base_channels=32,
                 channel_mults=(1, 2, 2), num_res_blocks=1, attention_resolutions=(16,), num_heads=4, num_groups=8)
    x = torch.randn(2, channels, 32, 32)
    t = torch.randint(0, 1000, (2,))
    assert model(x, t).shape == x.shape


def test_legacy_unet_shapes():
    model = LegacyUNET(input_channels=1, output_channels=1)
    x = torch.randn(2, 1, 32, 32)
    t = torch.randint(0, 1000, (2,))
    assert model(x, t).shape == x.shape


def test_unet_depends_on_timestep():
    torch.manual_seed(0)
    model = UNET(in_channels=1, out_channels=1, image_size=32, base_channels=32, channel_mults=(1, 2),
                 num_res_blocks=1, attention_resolutions=(), num_groups=8).eval()
    x = torch.randn(1, 1, 32, 32)
    a = model(x, torch.tensor([10]))
    b = model(x, torch.tensor([900]))
    assert not torch.allclose(a, b), 'output must change with t'


def test_legacy_key_remap():
    sd = {'Layer1.ResBlock1.conv1.weight': 1, 'Layer4.conv.weight': 2, 'module.Layer6.attention_layer.proj1.bias': 3, 'shallow_conv.weight': 4}
    out = _remap_legacy_keys(sd)
    assert set(out) == {'encoder.0.ResBlock1.conv1.weight', 'decoder.0.conv.weight',
                        'module.decoder.2.attention_layer.proj1.bias', 'shallow_conv.weight'}
