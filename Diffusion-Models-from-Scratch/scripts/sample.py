"""Generate images from a trained checkpoint.

    python scripts/sample.py --checkpoint checkpoints/mnist/ddpm_checkpoint.pt                    # DDPM, 1000 steps
    python scripts/sample.py --checkpoint checkpoints/mnist/ddpm_checkpoint.pt --sampler ddim      # DDIM, 50 steps
    python scripts/sample.py --checkpoint checkpoints/cifar10/ddpm_checkpoint.pt --sampler ddim --steps 20 --num-samples 64

Checkpoints written by the old single-file train.py are converted on the fly.

Writes to the results folder from the checkpoint's config (DDIM output goes to a
ddim<steps>_eta<eta>/ subfolder so the two samplers can be compared side by side):
    grid.png        all samples in one image
    sample_<i>.png  each final image
    reverse_<i>.png denoising trajectory of sample i, t = 999 -> 0
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from ddpm import DATASETS, build_diffusion, build_model, set_seed
from ddpm.checkpoint import load_checkpoint, load_weights


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--num-samples', dest='num_samples', type=int, default=None, help='override sample.num_samples')
    p.add_argument('--out', type=str, default=None, help='override the results folder')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--no-clip', dest='no_clip', action='store_true', help='DDIM: do not clamp the x0 estimate to [-1, 1]')
    p.add_argument('--no-ema', action='store_true', help='sample with the raw weights instead of the EMA weights')
    p.add_argument('--sampler', choices=['ddpm', 'ddim'], default=None, help='override sample.sampler (default ddpm)')
    p.add_argument('--steps', type=int, default=None, help='DDIM steps, override sample.ddim_steps (default 50)')
    p.add_argument('--eta', type=float, default=None, help='DDIM stochasticity, 0 = deterministic, 1 = DDPM-like (default 0)')
    p.add_argument('--upscale', type=int, default=4, help='nearest-neighbour upscale factor for the PNGs')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def to_png_range(x: torch.Tensor, factor: int) -> torch.Tensor:
    """[-1, 1] (clamped) -> [0, 1], upscaled for viewing."""
    x = (x.clamp(-1, 1) + 1) / 2
    return F.interpolate(x, scale_factor=factor, mode='nearest') if factor > 1 else x


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device(args.device)

    ck = load_checkpoint(args.checkpoint, device)
    cfg = ck['config']
    ds = DATASETS[cfg['dataset']]
    n = args.num_samples or cfg['sample']['num_samples']
    sc = cfg['sample']
    snapshot_steps = sc['snapshot_steps']
    sampler = args.sampler or sc.get('sampler', 'ddpm')
    steps = args.steps or sc.get('ddim_steps', 50)
    eta = args.eta if args.eta is not None else sc.get('ddim_eta', 0.0)
    out = args.out or cfg['paths']['results']
    if sampler == 'ddim' and args.out is None:
        out = os.path.join(out, f'ddim{steps}_eta{eta:g}')
    os.makedirs(out, exist_ok=True)

    model = build_model(cfg).to(device)
    load_weights(model, ck, use_ema=not args.no_ema)
    model.eval()
    diffusion = build_diffusion(cfg, model).to(device)

    shape = (n, ds.channels, ds.image_size, ds.image_size)
    how = f'DDIM {steps} steps, eta={eta:g}' if sampler == 'ddim' else f'DDPM {diffusion.num_timesteps} steps'
    print(f"Sampling {n} {cfg['dataset']} images | {how} | {cfg['diffusion']['schedule']} schedule | "
          f"epoch {ck.get('epoch', '?')} weights | {device}")
    if sampler == 'ddim':
        final, snapshots = diffusion.ddim_sample(shape, device=device, num_steps=steps, eta=eta,
                                                 snapshot_steps=snapshot_steps, clip_denoised=not args.no_clip)
    else:
        final, snapshots = diffusion.sample(shape, device=device, snapshot_steps=snapshot_steps)

    final = final.cpu()
    snapshots = [s.cpu() for s in snapshots]
    save_image(to_png_range(final, args.upscale), os.path.join(out, 'grid.png'), nrow=min(n, 10), padding=2)
    for i in range(n):
        save_image(to_png_range(final[i:i + 1], args.upscale), os.path.join(out, f'sample_{i}.png'))
        strip = torch.stack([s[i] for s in snapshots])  # (num_snapshots, C, H, W), t descending
        save_image(make_grid(to_png_range(strip, args.upscale), nrow=len(snapshots), padding=2),
                   os.path.join(out, f'reverse_{i}.png'))
    print(f'Saved {n} samples to {out}/')


if __name__ == '__main__':
    main()
