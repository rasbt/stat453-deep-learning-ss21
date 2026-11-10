"""Compute FID for a checkpoint: generate N images and compare Inception features
against the dataset's test split.

    python scripts/evaluate.py --checkpoint checkpoints/mnist/ddpm_checkpoint.pt --sampler ddim --steps 50 -n 5000
    python scripts/evaluate.py --checkpoint checkpoints/cifar10/ddpm_checkpoint.pt --sampler ddpm -n 5000

Appends one row to results/<dataset>/fid.csv. Real-image statistics are cached in
results/fid_stats/ so later runs only pay for generation.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm import DATASETS, build_diffusion, build_model, get_dataset, set_seed
from ddpm.checkpoint import load_checkpoint, load_weights


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddim')
    p.add_argument('--steps', type=int, default=50, help='DDIM steps (ignored for ddpm)')
    p.add_argument('--eta', type=float, default=0.0)
    p.add_argument('-n', '--num-samples', dest='n', type=int, default=5000)
    p.add_argument('--batch-size', dest='batch_size', type=int, default=250)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-clip', dest='no_clip', action='store_true', help='DDIM: do not clamp the x0 estimate to [-1, 1]')
    p.add_argument('--no-ema', dest='no_ema', action='store_true', help='use raw weights instead of the EMA weights')
    p.add_argument('--label', default='', help='free text stored in the csv row, e.g. the config name')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


@torch.no_grad()
def inception_features(inception, images, device):
    """images: (B, C, H, W) in [0, 1]. Grayscale is repeated to 3 channels."""
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return inception(images.to(device))[0].squeeze(-1).squeeze(-1).cpu().numpy()


def real_stats(dataset_name, inception, device, batch_size, cache_dir='results/fid_stats'):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{dataset_name}_test.npz')
    if os.path.exists(path):
        d = np.load(path)
        return d['mu'], d['sigma'], int(d['n'])
    ds, _ = get_dataset(dataset_name, train=False)
    feats = []
    for x, _ in tqdm(DataLoader(ds, batch_size=batch_size, num_workers=4), desc=f'Inception on {dataset_name} test'):
        feats.append(inception_features(inception, (x + 1) / 2, device))  # dataset is in [-1, 1]
    feats = np.concatenate(feats)
    mu, sigma = feats.mean(0), np.cov(feats, rowvar=False)
    np.savez(path, mu=mu, sigma=sigma, n=len(feats))
    return mu, sigma, len(feats)


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

    inception = InceptionV3([3], resize_input=True, normalize_input=True).to(device).eval()
    mu_r, sig_r, n_real = real_stats(cfg['dataset'], inception, device, args.batch_size)

    how = f'ddim{args.steps}_eta{args.eta:g}' if args.sampler == 'ddim' else 'ddpm'
    print(f"FID: {args.n} samples of {cfg['dataset']} | {how} | epoch {ck.get('epoch', '?')} | vs {n_real} test images")
    feats, t0 = [], time.time()
    shape = (args.batch_size, ds.channels, ds.image_size, ds.image_size)
    for _ in tqdm(range(0, args.n, args.batch_size), desc='Generating'):
        if args.sampler == 'ddim':
            x, _ = diffusion.ddim_sample(shape, device, num_steps=args.steps, eta=args.eta, clip_denoised=not args.no_clip)
        else:
            x, _ = diffusion.sample(shape, device)
        feats.append(inception_features(inception, (x.clamp(-1, 1) + 1) / 2, device))
    gen_seconds = time.time() - t0
    feats = np.concatenate(feats)[:args.n]
    fid = calculate_frechet_distance(feats.mean(0), np.cov(feats, rowvar=False), mu_r, sig_r)

    out = os.path.join(cfg['paths']['results'], 'fid.csv')
    new = not os.path.exists(out)
    with open(out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['label', 'dataset', 'arch', 'schedule', 'epoch', 'weights', 'sampler', 'steps', 'eta',
                        'num_samples', 'fid', 'gen_seconds', 'sec_per_image'])
        steps = args.steps if args.sampler == 'ddim' else diffusion.num_timesteps
        w.writerow([args.label or Path(args.checkpoint).parent.name, cfg['dataset'],
                    cfg['model'].get('arch', 'unet'), cfg['diffusion']['schedule'], ck.get('epoch', ''),
                    'raw' if args.no_ema else 'ema', args.sampler, steps, args.eta if args.sampler == 'ddim' else '',
                    args.n, f'{fid:.2f}', f'{gen_seconds:.1f}', f'{gen_seconds / args.n:.4f}'])
    print(f'FID = {fid:.2f}   ({gen_seconds:.0f}s to generate {args.n} images)  -> {out}')


if __name__ == '__main__':
    main()
