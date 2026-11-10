# Diffusion-Models-from-Scratch

A from-scratch PyTorch implementation of DDPM (Ho et al., 2020) with a UNet
noise predictor, trained on MNIST, Fashion-MNIST and CIFAR-10.

Originally based on [nickd16/Diffusion-Models-from-Scratch](https://github.com/nickd16/Diffusion-Models-from-Scratch).
Restructured into a package with a separate noise schedule, diffusion process,
model and data module, YAML experiment configs, linear or cosine schedules,
batched sampling and per-dataset result folders. The UNet was rebuilt to follow
the paper (learned time-embedding MLP, pre-norm residual attention, fixed-width
resampling, multiplier-based widths, dropout); the upstream network is kept as
`model.arch: legacy` for comparison.

## Setup

```bash
pip install -r requirements.txt
```

Datasets download automatically into `./data` on first use. A CUDA GPU is strongly recommended.

## Usage

Train:

```bash
python scripts/train.py --config configs/mnist.yaml
python scripts/train.py --config configs/fashion_mnist.yaml
python scripts/train.py --config configs/cifar10.yaml
```

Sample from the saved checkpoint:

```bash
python scripts/sample.py --checkpoint checkpoints/mnist/ddpm_checkpoint.pt                 # DDPM, 1000 steps
python scripts/sample.py --checkpoint checkpoints/mnist/ddpm_checkpoint.pt --sampler ddim   # DDIM, 50 steps
python scripts/sample.py --checkpoint checkpoints/cifar10/ddpm_checkpoint.pt --sampler ddim --steps 20 --num-samples 64
```

The checkpoint is written after every epoch, so you can sample from a run that is still training.
Checkpoints produced by the earlier single-file `train.py` are converted automatically by `sample.py`.

Command-line overrides:

| Script | Flag | Meaning |
|---|---|---|
| train | `--epochs N`, `--batch-size N`, `--lr F`, `--seed N` | override the config |
| train | `--resume PATH` | continue training from a checkpoint |
| train | `--reset-ema` | with `--resume`, start a fresh EMA from the loaded weights |
| sample | `--num-samples N` | how many images to generate |
| sample | `--sampler {ddpm,ddim}` | ancestral sampling over all T steps, or DDIM |
| sample | `--steps N`, `--eta F` | DDIM step count (default 50) and stochasticity (default 0, deterministic) |
| sample, evaluate | `--no-clip` | DDIM without clamping the x_0 estimate (breaks the cosine schedule) |
| sample | `--no-ema` | use raw weights instead of the EMA weights |
| sample | `--out DIR` | write somewhere other than the config's results folder |
| both | `--device cuda:1` | pick a GPU |

Everything else lives in the config. To try a variant, copy a config, change a
setting, and point `paths` at new folders so runs do not overwrite each other.
`configs/mnist_legacy.yaml` trains the original upstream network on identical
settings for a before/after comparison.

Model settings:

| Key | Meaning |
|---|---|
| `arch` | `unet` (paper-style, default) or `legacy` (upstream network) |
| `base_channels` | width at full resolution |
| `channel_mults` | width multiplier per resolution, one entry per downsampling level |
| `num_res_blocks` | ResBlocks per resolution |
| `attention_resolutions` | feature-map sizes that get self-attention |
| `dropout` | dropout inside ResBlocks; the paper uses 0.1 on CIFAR-10 |

## Outputs

```
checkpoints/<dataset>/ddpm_checkpoint.pt   # model, EMA, optimizer, config, epoch (git-ignored)
results/<dataset>/train_loss.csv           # per-epoch training loss
results/<dataset>/grid.png                 # all generated samples in one image (DDPM)
results/<dataset>/sample_<i>.png           # generated image i
results/<dataset>/reverse_<i>.png          # denoising trajectory of image i, t = 999 -> 0
results/<dataset>/ddim50_eta0/...          # same three outputs from the DDIM sampler
results/<dataset>/steps_comparison.png     # same noise through DDIM 5/10/20/50/100 steps and DDPM 1000
results/<dataset>/fid.csv                  # one row per FID evaluation (scripts/evaluate.py)
results/figures/                           # loss curves (scripts/plot_losses.py)
```

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/mnist/ddpm_checkpoint.pt --sampler ddim --steps 50 -n 5000
python scripts/compare_steps.py --checkpoint checkpoints/mnist/ddpm_checkpoint.pt
python scripts/plot_losses.py
PY=python scripts/run_eval.sh mnist cuda:0        # everything above for one run
PY=python scripts/reeval_ddim.sh mnist cuda:0     # redo only the DDIM part (samples, step comparison, DDIM FIDs)
python -m pytest tests -q                         # 12 CPU tests
```

FID uses pytorch-fid's Inception network against the dataset's 10k test split;
grayscale images are repeated to three channels. Real-image statistics are cached
in `results/fid_stats/`.

## Layout

```
configs/            one YAML per experiment (dataset, schedule, model, training, paths)
scripts/train.py    training loop, EMA, checkpointing, loss log
scripts/sample.py   batched ancestral sampling, writes PNGs
ddpm/
  schedule.py       linear / cosine beta schedules; NoiseSchedule holds alpha_bar etc. as buffers
  diffusion.py      GaussianDiffusion: q_sample, training_loss, p_sample, sample
  model/blocks.py   TimestepEmbedding, ResBlock, AttentionBlock, Downsample, Upsample
  model/unet.py     UNET built from base width, channel multipliers and attention resolutions
  model/legacy.py   the original upstream UNet (model.arch: legacy)
  data.py           MNIST / Fashion-MNIST / CIFAR-10, all 32x32 scaled to [-1, 1]
  config.py         load YAML, build model and diffusion from it
  checkpoint.py     load checkpoints, converting the old single-file format
  utils.py          seeding
```

## How it works

**Forward process.** `NoiseSchedule` precomputes beta_t and alpha_bar_t for all
T steps. `GaussianDiffusion.q_sample` noises an image in closed form:
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps.

**Model.** A UNet with two ResBlocks per resolution, self-attention at 16x16 and
in the bottleneck, and skip connections from encoder to decoder. The timestep
enters through a sinusoidal embedding, a shared two-layer MLP, and a learned
per-block projection added inside every ResBlock.

**Training.** For every batch, pick a random t per image, noise it, and train the
UNet to predict eps with MSE (`training_loss`). Gradients are clipped to norm 1
and an EMA copy of the weights is kept for sampling.

**EMA warmup matters.** With decay 0.9999 and no warmup, the EMA still contains
0.9999^steps of the random initialisation: 25% after the 14k steps of a 15-epoch
MNIST run, which is enough to turn every sample into noise. `ema_warmup: true`
(the default) ramps the decay up from zero so short runs sample correctly. Use
`--no-ema` in `sample.py` / `evaluate.py` to sample from the raw weights instead.

**Sampling (DDPM).** `sample` starts from Gaussian noise and applies `p_sample` for
t = T-1 down to 0. Each step subtracts the predicted noise scaled to the current
noise level, rescales, and adds fresh noise with variance sigma_t^2 (beta_t by
default, or the posterior variance). No noise is added at t = 0.

**Sampling (DDIM).** `ddim_sample` visits only a subsequence of timesteps (50 by
default). At each one the predicted noise gives an estimate of x_0, which is then
re-noised to the next timestep in the subsequence: x_prev = sqrt(abar_prev) x0_hat
+ sqrt(1 - abar_prev - sigma^2) eps + sigma z. With eta = 0, sigma = 0 and sampling
is deterministic, so the same starting noise always gives the same image. Needs
no retraining: the same eps-predictor serves both samplers. The x_0 estimate is
clamped to [-1, 1] at every step (`--no-clip` disables this). That matters for
the cosine schedule, whose alpha_bar_T is 2e-9: without clipping the first DDIM
step divides a small prediction error by sqrt(alpha_bar_T) and the samples are
pure noise, while DDPM sampling is unaffected.

## Results

Results for each dataset go in `results/<dataset>/`.
