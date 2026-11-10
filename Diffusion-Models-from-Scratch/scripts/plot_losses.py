"""Plot training-loss curves from results/*/train_loss.csv.

    python scripts/plot_losses.py                      # one figure per run + comparison figures
Writes results/figures/loss_<run>.png and results/figures/loss_compare_<name>.png.
"""
import csv
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GRAY, ACCENT, INK, MUTED = '#9AA0A6', '#2F6FED', '#1F1F1F', '#6B7280'

COMPARISONS = {
    'mnist_arch': (['mnist', 'mnist_legacy'], {'mnist': 'paper-style UNet (9M params)', 'mnist_legacy': 'legacy UNet (35M params)'},
                   'MNIST: rebuilt 9M UNet reaches a lower loss than the 35M upstream network'),
    'mnist_schedule': (['mnist', 'mnist_cosine'], {'mnist': 'linear schedule', 'mnist_cosine': 'cosine schedule'},
                       'MNIST: cosine schedule trains to a higher loss (different noise levels, not comparable)'),
    'fashion_arch': (['fashion_mnist', 'fashion_mnist_legacy'], {'fashion_mnist': 'paper-style UNet', 'fashion_mnist_legacy': 'legacy UNet'},
                     'Fashion-MNIST: rebuilt UNet (lr 2e-4) vs upstream network (lr 2e-5)'),
}


def read(run):
    p = ROOT / 'results' / run / 'train_loss.csv'
    if not p.exists():
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return [int(r['epoch']) for r in rows], [float(r['loss']) for r in rows]


def style(ax, title):
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=12)
    ax.set_xlabel('epoch', color=MUTED)
    ax.set_ylabel('training loss (MSE on noise)', color=MUTED)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#D0D4DA')
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', color='#E5E7EB', linewidth=0.8)
    ax.set_axisbelow(True)


def single(run, out_dir):
    d = read(run)
    if d is None:
        return
    ep, loss = d
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.plot(ep, loss, color=ACCENT, linewidth=2)
    ax.scatter([ep[-1]], [loss[-1]], color=ACCENT, s=24, zorder=3)
    ax.annotate(f'{loss[-1]:.4f}', (ep[-1], loss[-1]), textcoords='offset points', xytext=(6, 0), va='center', color=INK, fontsize=9)
    style(ax, f'{run}: loss fell from {loss[0]:.3f} to {loss[-1]:.4f} over {ep[-1]} epochs')
    fig.tight_layout()
    fig.savefig(out_dir / f'loss_{run}.png')
    plt.close(fig)


def compare(name, runs, labels, title, out_dir):
    data = {r: read(r) for r in runs}
    if any(v is None for v in data.values()):
        return
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    colors = [ACCENT, GRAY]
    for (run, (ep, loss)), c in zip(data.items(), colors):
        ax.plot(ep, loss, color=c, linewidth=2, label=labels[run])
        ax.annotate(f'{loss[-1]:.4f}', (ep[-1], loss[-1]), textcoords='offset points', xytext=(6, 0), va='center', color=INK, fontsize=9)
    style(ax, title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f'loss_compare_{name}.png')
    plt.close(fig)


def main():
    out_dir = ROOT / 'results' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(Path(p).parent.name for p in glob.glob(str(ROOT / 'results' / '*' / 'train_loss.csv')))
    for r in runs:
        single(r, out_dir)
    for name, (rs, labels, title) in COMPARISONS.items():
        compare(name, rs, labels, title, out_dir)
    print('runs:', ', '.join(runs))
    print('figures ->', out_dir)


if __name__ == '__main__':
    main()
