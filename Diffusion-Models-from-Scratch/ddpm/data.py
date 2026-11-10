"""Dataset registry for the diffusion experiments.

Every dataset is delivered as 32x32 images scaled to [-1, 1], so the UNet and
the sampler never need to know which dataset they are working with.
"""
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    channels: int
    image_size: int
    default_epochs: int


DATASETS = {
    'mnist':         DatasetConfig('mnist',         channels=1, image_size=32, default_epochs=15),
    'fashion_mnist': DatasetConfig('fashion_mnist', channels=1, image_size=32, default_epochs=15),
    'cifar10':       DatasetConfig('cifar10',       channels=3, image_size=32, default_epochs=100),
}


def get_dataset(name: str, root: str = './data', train: bool = True) -> Tuple[Dataset, DatasetConfig]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}")
    cfg = DATASETS[name]
    # ToTensor gives [0, 1]; Normalize with mean 0.5 / std 0.5 maps that to [-1, 1].
    to_unit_range = transforms.Normalize([0.5] * cfg.channels, [0.5] * cfg.channels)

    if name == 'mnist':
        # 28x28 -> 32x32 so three stride-2 downsamples divide cleanly.
        tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), to_unit_range])
        ds = datasets.MNIST(root=root, train=train, download=True, transform=tf)
    elif name == 'fashion_mnist':
        tf = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), to_unit_range])
        ds = datasets.FashionMNIST(root=root, train=train, download=True, transform=tf)
    else:  # cifar10, already 32x32 RGB
        tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), to_unit_range])
        ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tf)
    return ds, cfg
