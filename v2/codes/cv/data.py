"""CV data loading — v1 transforms, plus the train/val split the paper states.

v1 code evaluated on the test set every epoch; the paper's Methods section
states 10% of training data is held out for validation. v2 implements what the
paper states: a seeded 90/10 train/val split (fixed across optimizers within a
seed), test set touched only for the final evaluation and the reported curves.
Transforms are identical to v1: ToTensor + Normalize(0.5), no augmentation.
"""

from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

DATASETS = {
    "cifar10": (torchvision.datasets.CIFAR10, 10, 3, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))),
    "cifar100": (torchvision.datasets.CIFAR100, 100, 3, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))),
    "fashion_mnist": (torchvision.datasets.FashionMNIST, 10, 1, ((0.5,), (0.5,))),
}

HF_CIFAR = {"cifar10": ("uoft-cs/cifar10", "label"),
            "cifar100": ("uoft-cs/cifar100", "fine_label")}


class _MaterializedCIFAR(Dataset):
    """CIFAR served from in-memory uint8 arrays (same profile as torchvision)."""

    def __init__(self, images_u8, labels, transform):
        self.images = images_u8            # (N, 32, 32, 3) uint8
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.images[idx])
        return self.transform(img), int(self.labels[idx])


def _load_cifar_hf(dataset_name, split, transform, cache_dir):
    """CIFAR via the HF parquet mirror (uoft-cs/*): identical images/labels to
    the original tarballs (lossless PNG), used because cs.toronto.edu throttles
    downloads to ~15 kB/s. Materialized to RAM once at construction."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_npy = cache_dir / f"{dataset_name}_{split}_images.npy"
    lab_npy = cache_dir / f"{dataset_name}_{split}_labels.npy"
    if img_npy.exists() and lab_npy.exists():
        return _MaterializedCIFAR(np.load(img_npy, mmap_mode="r"),
                                  np.load(lab_npy), transform)
    from datasets import load_dataset
    repo, label_key = HF_CIFAR[dataset_name]
    ds = load_dataset(repo, cache_dir=str(cache_dir), split=split)
    imgs = np.stack([np.asarray(im.convert("RGB")) for im in ds["img"]])
    labels = np.asarray(ds[label_key])
    np.save(img_npy, imgs)
    np.save(lab_npy, labels)
    return _MaterializedCIFAR(imgs, labels, transform)


def load_cv_data(dataset_name: str, data_dir: str, batch_size: int, seed: int,
                 val_fraction: float = 0.1, num_workers: int = 2,
                 loader_seed: int | None = None):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown CV dataset: {dataset_name}")
    cls, num_classes, channels, (mean, std) = DATASETS[dataset_name]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    root = Path(data_dir) / dataset_name
    if dataset_name in HF_CIFAR:
        try:  # local torchvision files if already present; never re-download
            train_full = cls(root=root, train=True, download=False, transform=transform)
            test_set = cls(root=root, train=False, download=False, transform=transform)
        except RuntimeError:
            hf_cache = Path(data_dir) / "hf_cache"
            train_full = _load_cifar_hf(dataset_name, "train", transform, hf_cache)
            test_set = _load_cifar_hf(dataset_name, "test", transform, hf_cache)
    else:
        train_full = cls(root=root, train=True, download=True, transform=transform)
        test_set = cls(root=root, train=False, download=True, transform=transform)

    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val
    # split depends only on the seed -> identical across optimizer arms
    train_set, val_set = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed))

    loader_generator = None
    if loader_seed is not None:
        loader_generator = torch.Generator().manual_seed(loader_seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              generator=loader_generator)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, num_classes, channels
