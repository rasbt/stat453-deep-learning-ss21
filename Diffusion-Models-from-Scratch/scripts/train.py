"""Train a DDPM from a YAML config.

    python scripts/train.py --config configs/mnist.yaml
    python scripts/train.py --config configs/cifar10.yaml --epochs 50 --resume checkpoints/cifar10/ddpm_checkpoint.pt
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm import apply_overrides, build_diffusion, build_model, get_dataset, load_config, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='YAML experiment config')
    p.add_argument('--epochs', type=int, default=None, help='override train.epochs')
    p.add_argument('--batch-size', dest='batch_size', type=int, default=None, help='override train.batch_size')
    p.add_argument('--lr', type=float, default=None, help='override train.lr')
    p.add_argument('--seed', type=int, default=None, help='override seed')
    p.add_argument('--resume', type=str, default=None, help='checkpoint to continue training from')
    p.add_argument('--reset-ema', dest='reset_ema', action='store_true',
                   help='on resume, start a fresh EMA from the loaded weights instead of loading the saved EMA')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    if args.seed is not None:
        cfg['seed'] = args.seed
    seed = cfg['seed'] if cfg['seed'] != -1 else random.randint(0, 2**32 - 1)
    set_seed(seed)

    tr = cfg['train']
    device = torch.device(args.device)
    ckpt_path = cfg['paths']['checkpoint']
    results = cfg['paths']['results']
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(results, exist_ok=True)

    dataset, ds_cfg = get_dataset(cfg['dataset'])
    loader = DataLoader(dataset, batch_size=tr['batch_size'], shuffle=True, drop_last=True,
                        num_workers=tr['num_workers'], pin_memory=device.type == 'cuda')

    model = build_model(cfg).to(device)
    diffusion = build_diffusion(cfg, model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=tr['lr'])
    # use_warmup ramps the decay up from 0 so the EMA tracks the model early on. Without it,
    # decay 0.9999 keeps 0.9999**steps of the random init: 25% after 14k steps, enough to make
    # every sample pure noise on a 15-epoch MNIST run.
    ema = ModelEmaV3(model, decay=tr['ema_decay'], use_warmup=tr.get('ema_warmup', True))

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        if args.reset_ema:
            ema = ModelEmaV3(model, decay=tr['ema_decay'], use_warmup=tr.get('ema_warmup', True))
            print('EMA reset to the loaded model weights')
        else:
            ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck.get('epoch', 0)
        print(f'Resumed from {args.resume} at epoch {start_epoch}')
    # timm applies EMA warmup only when the global step is passed to update(); count it ourselves
    global_step = start_epoch * len(loader)

    log_path = os.path.join(results, 'train_loss.csv')
    if start_epoch == 0 or not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'loss'])

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{cfg['dataset']}: {len(dataset)} images, {ds_cfg.channels} ch | "
          f"{cfg['diffusion']['schedule']} schedule, T={diffusion.num_timesteps} | "
          f"UNet {n_params:.1f}M params | {tr['epochs']} epochs, batch {tr['batch_size']}, lr {tr['lr']} | "
          f"seed {seed} | {device}")

    for epoch in range(start_epoch, tr['epochs']):
        model.train()
        total = 0.0
        for x, _ in tqdm(loader, desc=f'Epoch {epoch + 1}/{tr["epochs"]}'):
            x = x.to(device, non_blocking=True)
            loss = diffusion.training_loss(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tr.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr['grad_clip'])
            optimizer.step()
            global_step += 1
            ema.update(model, step=global_step)
            total += loss.item()
        epoch_loss = total / len(loader)
        print(f'Epoch {epoch + 1} | loss {epoch_loss:.5f}')
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{epoch_loss:.6f}'])

        torch.save({
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': cfg,
            'epoch': epoch + 1,
            'seed': seed,
        }, ckpt_path)

    print(f'Saved checkpoint to {ckpt_path}')
    print(f'Next: python scripts/sample.py --checkpoint {ckpt_path}')


if __name__ == '__main__':
    main()
