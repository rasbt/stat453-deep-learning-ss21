# Results index

All runs: seed 0, Adam, EMA with warmup (decay 0.9999), grad clip 1.0, T = 1000, linear schedule unless noted.
FID: 5000 generated images vs the 10k test split (pytorch-fid Inception), lower is better. Full tables in `<run>/fid.csv`.

| Run | Dataset | Architecture | Params | Epochs | Final loss | FID DDPM 1000 | FID DDIM 50 | FID DDIM 100 |
|---|---|---|---|---|---|---|---|---|
| `mnist` | MNIST | paper-style UNet | 9.0 M | 15 | 0.0167 | **4.78** | 5.13 | 4.97 |
| `mnist_legacy` | MNIST | upstream UNet | 35.2 M | 15 | 0.0194 | 7.09 | 11.17 | 11.38 |
| `mnist_cosine` | MNIST | paper-style, cosine schedule | 9.0 M | 15 | 0.0285 | 7.85 | 7.12 | 5.85 |
| `fashion_mnist` | Fashion-MNIST | paper-style UNet | 9.0 M | 15 | 0.0271 | 8.13 | **8.02** | 7.99 |
| `fashion_mnist_legacy` | Fashion-MNIST | upstream UNet, lr 2e-5, raw weights | 35.2 M | 15 | 0.0373 | 72.4 | 78.2 | 77.6 |
| `cifar10` | CIFAR-10 | paper-style, base 128, dropout 0.1 | 35.8 M | 105 | 0.0298 | **16.30** | 24.6 | 22.9 |
| `mnist_noWarmup` | MNIST | paper-style, EMA without warmup | 9.0 M | 15 | 0.0167 | (EMA samples are noise) | | |

## Per-run folders

```
<run>/train_loss.csv          per-epoch training loss
<run>/fid.csv                 FID rows: sampler, steps, weights, FID, generation time
<run>/grid.png                100 DDPM samples, seed 0
<run>/sample_<i>.png          individual DDPM samples
<run>/reverse_<i>.png         denoising trajectory of sample i, t = 999 ... 0
<run>/ddim50_eta0/            the same three outputs from 50-step DDIM
<run>/steps_comparison.png    same 8 noises through DDIM 5/10/20/50/100 and DDPM 1000
```

## Figures (`figures/`)

| File | Shows |
|---|---|
| `mnist_arch_grids.png`, `loss_compare_mnist_arch.png` | paper-style vs legacy UNet on MNIST |
| `fashion_arch_grids.png`, `loss_compare_fashion_arch.png` | same on Fashion-MNIST |
| `schedule_grids.png`, `loss_compare_mnist_schedule.png` | linear vs cosine schedule |
| `ema_warmup_lesson.png` | raw weights vs EMA without warmup vs EMA with warmup |
| `*_trajectories.png` | stacked denoising strips |
| `loss_<run>.png` | individual loss curves |

## Two bugs worth remembering

1. **EMA without warmup.** timm's `ModelEmaV3` applies warmup only when `update(model, step=...)` receives the step. Without it, decay 0.9999 leaves 0.9999^steps of the random init in the average: 25% after 14k steps, and every sample is noise while the training loss looks fine.
2. **DDIM on the cosine schedule.** alpha_bar_T is 2e-9, so the x0 estimate divides by 5e-5. Clamp x0 to [-1, 1] at each step (now the default; `--no-clip` disables it). Unclamped, the cosine model scored FID 294 to 338 with DDIM and 7.9 with DDPM from the same checkpoint.
