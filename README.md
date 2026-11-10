# MSCS435: Deep Learning

Course materials for MSCS435, Fall 2026, adapted by atwumasi from Sebastian
Raschka's [STAT 453: Intro to Deep Learning](https://github.com/rasbt/stat453-deep-learning-ss21)
(UW-Madison, Spring 2021). See [LICENSE](LICENSE) for attribution and reuse terms.

## Structure

Each `LXX/` folder is a self-contained lecture: notebooks plus a local copy of
`helper_*.py` (dataset loading, training loop, evaluation, plotting). Folders
don't import from each other, so they can be reordered, dropped, or renamed
independently.

| Folder | Topic |
|---|---|
| L01 | Intro / a simple CNN in plain PyTorch |
| L03 | Perceptron (NumPy and PyTorch) |
| L05 | Linear regression, Adaline (gradient descent) |
| L06 | PyTorch autograd |
| L08 | Logistic/softmax regression, cross-entropy |
| L09 | Multilayer perceptrons, custom `Dataset`/`DataLoader` |
| L10 | Regularization: L2, dropout, data augmentation |
| L11 | Weight initialization, batch normalization |
| L12 | Optimizers and schedulers (Adam, AdamW, AdaBelief, batch size) |
| L13 | CNNs: LeNet-5, AlexNet |
| L14 | Deeper CNNs: VGG16, ResNet34, fully-conv nets, transfer learning |
| L15 | RNNs for text: rule-based (VADER) baseline, LSTM, packed sequences — **LSTM notebooks currently broken, see Known issues** |
| L16 | Autoencoders |
| L17 | Variational autoencoders (VAE) |
| L18 | GANs: DCGAN, WGAN, WGAN-GP (`optional_wgan/`) |
| L19 | Character-RNN, DistilBERT text classification |

## Setup

No environment file existed in the upstream repo; `requirements.txt` and
`environment.yml` here were added for this course.

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate mscs435-dl
```

**pip:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python **3.10** is what's pinned in `environment.yml` — see *Known issues*
below for why 3.12+ needs a code fix first.

NLTK's VADER lexicon (used in `L15/0_rule-based-baseline.ipynb`) needs a
one-time download:
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Running a lecture

```bash
cd L13/code   # or whichever LXX folder
jupyter lab
```

Open the notebook and run top to bottom. Most datasets (MNIST, CIFAR-10)
auto-download via `torchvision.datasets` on first run. Exceptions:
- `L15/movie_data.csv` is committed directly (no download step).
- CelebA (used in several `L17`/`L18` notebooks) is large and not bundled —
  check the notebook's data-loading cell for the expected path and download
  it manually first.

## Known issues

Found doing a pass over the code before adapting it for this course.

### Fixed

- **`distutils` import broke on Python 3.12+.** `helper_evaluation.py` in
  `L09`–`L14` did `from distutils.version import LooseVersion`; `distutils`
  was removed from the standard library in 3.12. The version-gated branch
  (for torch <1.7) was dropped — `set_deterministic()` now always calls
  `torch.use_deterministic_algorithms(True)`.
- **`torch.set_deterministic()` no longer exists.** Removed pre-1.7 API,
  still called unconditionally in `L01/code/helper.py`,
  `L09/code/mlp-softmax-pyscripts/helper.py`, and `L16`–`L18/helper_utils.py`.
  Replaced with `torch.use_deterministic_algorithms(True)` everywhere.
- **Deterministic mode crashed on GPU.** `torch.use_deterministic_algorithms(True)`
  requires `CUBLAS_WORKSPACE_CONFIG` to be set *before* any CUDA op, or CuBLAS
  ops (e.g. `backward()` through a `Linear` layer) raise `RuntimeError`. Added
  `os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'` to the top of every
  `set_deterministic()` (11 copies).
- **`train_vae_v1` raised `KeyError` on its default arguments.** In
  `L16/L17/L18/helper_train.py`, the epoch-loss log line wrote to
  `log_dict['train_combined_per_epoch']`, but the dict was initialized with
  the key `'train_combined_loss_per_epoch'`. Fixed the mismatched key.
- **`train_wgan_v1`'s generator/discriminator update ratio was inverted.** In
  `L18/helper_train.py`, with `discr_iter_per_generator_iter=5` the code
  trained the *generator* on 5 of every 6 batches instead of training the
  *discriminator* 5x per generator update — the opposite of the WGAN
  algorithm the lecture describes (and of what the notebook name
  `04_02_wgan-mnist-5xdiscr.ipynb` promises). Fixed the loop condition.
- **Dead `torchtext` imports crashed unrelated notebooks.** `torchtext`'s
  compiled extension is ABI-locked to a specific torch minor version and has
  no build compatible with torch≥2.4 (needed here for GPU support — see
  below). Several notebooks imported or `%watermark`'d it purely for a
  version banner, never actually using it: all three `L19/distilbert-classifier`
  notebooks and both `L19/character-rnn` notebooks. Removed the dead
  references; the notebooks don't need torchtext at all.
- **`transformers.AdamW` was removed.** `L19/distilbert-classifier/03_distilbert-with-trainer.ipynb`
  imported it directly; newer `transformers` dropped the re-export. Switched
  to `torch.optim.AdamW`.
- **Hardcoded `cuda:3` (or higher) crashed on machines with fewer GPUs.**
  Several notebooks pin a specific GPU index left over from the original
  author's multi-GPU workstation. Fixed to `cuda:0` in `L13/code/3-cnn-cifar10.ipynb`,
  `L14/2-resnet34.ipynb`, `L16/conv-autoencoder_mnist.ipynb`,
  `L17/5_VAE_celeba_latent-arithmetic.ipynb`, and both DistilBERT notebooks
  that had it (`01`, `03`). If your machine has a different number of GPUs,
  grep for `cuda:` and adjust the index to one that exists (`nvidia-smi`
  lists valid indices).
- **No dependency manifest existed upstream.** Added `requirements.txt` and
  `environment.yml`, with `torch`/`torchvision` pinned to a CUDA 12.4 build
  (matching driver 550.x+) rather than left to resolve to whatever the
  default PyPI/conda build is — an unpinned install can silently resolve to
  a newer CUDA build than your driver supports, which fails at *runtime*
  (`CUDA initialization: driver too old`) rather than at install time. Check
  `nvidia-smi` (top-right "CUDA Version") and adjust the `cu124` tag in both
  files if your driver supports a different version.

### Not yet fixed

- **L15's LSTM notebooks (`1_lstm.ipynb`, `2_packed-lstm.ipynb`) are broken
  and need a rewrite, not a patch.** They depend on `torchtext.legacy.data`
  (`Field`, `LabelField`, `TabularDataset`, `BucketIterator`) two ways over:
  (1) `torchtext`'s compiled extension can't even be imported against the
  torch version this repo now needs for GPU support (same ABI issue as
  above), and (2) even on a torch/torchtext pairing where import succeeds,
  the `.legacy` submodule was removed from torchtext entirely in later
  releases — it doesn't exist in the `torchtext` version compatible with any
  currently-supported torch. Fixing this means replacing the data pipeline
  with a plain `torch.utils.data.Dataset`/`DataLoader` + a modern tokenizer
  (e.g. `torchtext.data.utils.get_tokenizer` plus a hand-rolled vocab, or
  swap to `transformers`' tokenizer as `L19` does), not swapping API calls
  1:1. **Recommendation: exclude L15's `1_` and `2_` notebooks from student
  materials until this rewrite happens** — `0_rule-based-baseline.ipynb`
  (VADER, no torchtext) is unaffected and works fine.
- **Large binaries committed to git** (`MNIST.zip` at 33MB, `.pt` checkpoints
  under `L18/optional_wgan` up to 25MB) bloat every clone. Not blocking, but
  worth pruning from history (e.g. with `git filter-repo`) at some point.

## Attribution

Original course design, notebooks, and helper code © 2021 Sebastian
Raschka, released under the MIT License. Modifications for MSCS435 © 2026
atwumasi, also under the MIT License — see [LICENSE](LICENSE).
