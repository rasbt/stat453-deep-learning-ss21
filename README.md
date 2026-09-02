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
| L15 | RNNs for text: rule-based (VADER) baseline, LSTM, packed sequences |
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

## Known issues (inherited from upstream)

These were found doing a pass over the code before adapting it for this
course. Worth fixing before assigning the affected lectures:

- **`distutils` import breaks on Python 3.12+.** `helper_evaluation.py` in
  `L09`–`L14` does `from distutils.version import LooseVersion`, and
  `distutils` was removed from the standard library in 3.12. Either pin the
  course environment to Python ≤3.11 (as `environment.yml` does here), or
  patch the version check to use `packaging.version.parse` instead.
- **`train_vae_v1` raises `KeyError` on its default arguments.** In
  `L16/L17/L18/helper_train.py`, the epoch-loss log line writes to
  `log_dict['train_combined_per_epoch']`, but the dict is initialized with
  the key `'train_combined_loss_per_epoch'`. Any VAE training run with
  `skip_epoch_stats=False` (the default) crashes at the end of epoch 1.
- **`train_wgan_v1`'s generator/discriminator update ratio is inverted.** In
  `L18/helper_train.py`, with `discr_iter_per_generator_iter=5` the code
  trains the *generator* on 5 of every 6 batches instead of training the
  *discriminator* 5x per generator update — the opposite of the WGAN
  algorithm the lecture describes (and of what the notebook name
  `04_02_wgan-mnist-5xdiscr.ipynb` promises).

## Attribution

Original course design, notebooks, and helper code © 2021 Sebastian
Raschka, released under the MIT License. Modifications for MSCS435 © 2026
atwumasi, also under the MIT License — see [LICENSE](LICENSE).
