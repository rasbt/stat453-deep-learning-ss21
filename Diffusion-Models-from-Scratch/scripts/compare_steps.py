"""Same starting noise, different samplers / step counts, one figure.

    python scripts/compare_steps.py --checkpoint checkpoints/mnist/ddpm_checkpoint.pt
    python scripts/compare_steps.py --checkpoint checkpoints/cifar10/ddpm_checkpoint.pt --steps 5 10 20 50 100 --ddpm

Writes results/<dataset>/steps_comparison.png: one row per setting, the same 8
noise vectors down each column, so differences are due to the sampler alone.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

from ddpm import DATASETS, build_diffusion, build_model, set_seed
from ddpm.checkpoint import load_checkpoint, load_weights


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[5, 10, 20, 50, 100], help='DDIM step counts')
    p.add_argument('--ddpm', action='store_true', default=True, help='also include the full DDPM sampler (default on)')
    p.add_argument('--no-ddpm', dest='ddpm', action='store_false')
    p.add_argument('-n', type=int, default=8, help='images per row')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-clip', dest='no_clip', action='store_true', help='DDIM: do not clamp the x0 estimate to [-1, 1]')
    p.add_argument('--no-ema', dest='no_ema', action='store_true', help='use raw weights instead of the EMA weights')
    p.add_argument('--scale', type=int, default=3)
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    ck = load_checkpoint(args.checkpoint, device)
    cfg = ck['config']
    ds = DATASETS[cfg['dataset']]
    model = build_model(cfg).to(device)
    load_weights(model, ck, use_ema=not args.no_ema)
    model.eval()
    diffusion = build_diffusion(cfg, model).to(device)

    shape = (args.n, ds.channels, ds.image_size, ds.image_size)
    noise = torch.randn(*shape, device=device)
    rows = []
    for s in args.steps:
        t0 = time.time()
        x, _ = diffusion.ddim_sample(shape, device, num_steps=s, eta=0.0, noise=noise, clip_denoised=not args.no_clip)
        rows.append((f'DDIM {s} steps', x, time.time() - t0))
    if args.ddpm:
        t0 = time.time()
        x, _ = diffusion.sample(shape, device, noise=noise)
        rows.append((f'DDPM {diffusion.num_timesteps} steps', x, time.time() - t0))

    # render: label column on the left, image row on the right
    try:
        font = ImageFont.truetype('/home/atwumasi/miniconda3/envs/mscs635/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf', 18)
        small = ImageFont.truetype('/home/atwumasi/miniconda3/envs/mscs635/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf', 14)
    except OSError:
        font = small = ImageFont.load_default()
    label_w, pad, cell = 230, 2, ds.image_size * args.scale
    grids = []
    for name, x, secs in rows:
        g = make_grid(torch.nn.functional.interpolate((x.clamp(-1, 1) + 1) / 2, scale_factor=args.scale, mode='nearest').cpu(),
                      nrow=args.n, padding=pad, pad_value=1.0)
        grids.append((name, secs, Image.fromarray((g.permute(1, 2, 0).numpy() * 255).astype('uint8'))))
    row_h = grids[0][2].height
    canvas = Image.new('RGB', (label_w + grids[0][2].width, row_h * len(grids)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (name, secs, im) in enumerate(grids):
        canvas.paste(im, (label_w, i * row_h))
        draw.text((12, i * row_h + row_h // 2 - 22), name, fill='black', font=font)
        draw.text((12, i * row_h + row_h // 2 + 2), f'{secs:.1f}s for {args.n} images', fill=(110, 110, 110), font=small)

    out = args.out or os.path.join(cfg['paths']['results'], 'steps_comparison.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    canvas.save(out)
    for name, _, secs in [(r[0], None, r[2]) for r in rows]:
        print(f'{name:18s} {secs:6.1f}s')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
