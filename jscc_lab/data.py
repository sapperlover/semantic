"""Dataset loading utilities for CIFAR-10 array and batch formats.

The homework data format stores each image as 3072 uint8 values:
the first 1024 entries are the red channel, then green, then blue.
This module converts that layout into PyTorch-friendly NCHW float32
images in the range [0, 1].
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


ArrayPair = Tuple[np.ndarray, np.ndarray]


_CIFAR_BATCH_NAMES = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
]


def _get_mapping_value(mapping: Mapping, names: Sequence[str]):
    """Read a value from CIFAR dictionaries with str or bytes keys."""

    for name in names:
        if name in mapping:
            return mapping[name]
        key = name.encode("utf-8")
        if key in mapping:
            return mapping[key]
    raise KeyError(f"None of the keys {names!r} were found.")


def _array_to_nchw_float(data: np.ndarray) -> np.ndarray:
    """Convert raw CIFAR rows or image arrays into NCHW float32 [0, 1]."""

    array = np.asarray(data)

    if array.ndim == 2:
        if array.shape[1] != 3072:
            raise ValueError(f"Expected raw CIFAR data with shape (N, 3072), got {array.shape}.")
        if array.dtype != np.uint8:
            raise TypeError(f"Expected raw CIFAR data dtype uint8, got {array.dtype}.")
        array = array.reshape(-1, 3, 32, 32)
    elif array.ndim == 4:
        # Accept already-shaped arrays in either NCHW or NHWC to make future
        # generated fixtures and cached data convenient to reuse.
        if array.shape[1:] == (3, 32, 32):
            pass
        elif array.shape[1:] == (32, 32, 3):
            array = np.transpose(array, (0, 3, 1, 2))
        else:
            raise ValueError(
                "Expected image arrays with shape (N, 3, 32, 32) or (N, 32, 32, 3), "
                f"got {array.shape}."
            )
    else:
        raise ValueError(f"Expected 2D raw rows or 4D images, got shape {array.shape}.")

    array = array.astype(np.float32, copy=False)
    if array.max(initial=0.0) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _labels_to_int64(labels: Sequence[int], expected_len: int) -> np.ndarray:
    """Validate labels and return a compact int64 vector."""

    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {label_array.shape}.")
    if len(label_array) != expected_len:
        raise ValueError(f"Expected {expected_len} labels, got {len(label_array)}.")
    if len(label_array) and (label_array.min() < 0 or label_array.max() > 9):
        raise ValueError("CIFAR-10 labels must be integers in [0, 9].")
    return label_array


def _load_npz(path: Path) -> ArrayPair:
    """Load an .npz file containing `data` and `labels` arrays."""

    with np.load(path, allow_pickle=False) as npz:
        if "data" not in npz or "labels" not in npz:
            raise KeyError(f"{path} must contain arrays named 'data' and 'labels'.")
        images = _array_to_nchw_float(npz["data"])
        labels = _labels_to_int64(npz["labels"], expected_len=len(images))
    return images, labels


def _load_pickle_batch(path: Path) -> ArrayPair:
    """Load a CIFAR-style pickle batch dictionary."""

    with path.open("rb") as f:
        batch = pickle.load(f, encoding="latin1")
    if not isinstance(batch, MutableMapping):
        raise TypeError(f"Expected a pickle dictionary in {path}, got {type(batch)!r}.")

    raw_data = _get_mapping_value(batch, ["data"])
    raw_labels = _get_mapping_value(batch, ["labels", "fine_labels"])
    images = _array_to_nchw_float(raw_data)
    labels = _labels_to_int64(raw_labels, expected_len=len(images))
    return images, labels


def _scan_directory(path: Path) -> List[Path]:
    """Find loadable CIFAR batches in a directory in a stable order."""

    ordered = [path / name for name in _CIFAR_BATCH_NAMES if (path / name).is_file()]
    npz_files = sorted(path.glob("*.npz"))

    if ordered:
        return ordered + [p for p in npz_files if p not in ordered]

    candidates: List[Path] = []
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        if child.name in {"batches.meta", "readme.html"}:
            continue
        if child.suffix.lower() in {".npz", ".pkl", ".pickle"} or "batch" in child.name:
            candidates.append(child)
    return candidates


def load_cifar_array_dataset(data_path: str | Path) -> ArrayPair:
    """Load CIFAR-10 arrays from .npz, a pickle batch, or a directory.

    Parameters
    ----------
    data_path:
        A path to one `.npz` file, one CIFAR pickle batch, or a directory
        containing CIFAR files such as `data_batch_1` and `test_batch`.

    Returns
    -------
    images, labels:
        `images` has shape `(N, 3, 32, 32)`, dtype `float32`, and values in
        `[0, 1]`; `labels` has shape `(N,)` and dtype `int64`.
    """

    path = Path(data_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_dir():
        files = _scan_directory(path)
        if not files:
            raise FileNotFoundError(f"No loadable CIFAR files found in directory: {path}")
        parts = [load_cifar_array_dataset(file_path) for file_path in files]
        images = np.concatenate([part[0] for part in parts], axis=0)
        labels = np.concatenate([part[1] for part in parts], axis=0)
        return images, labels

    if path.suffix.lower() == ".npz":
        return _load_npz(path)
    return _load_pickle_batch(path)


class CIFARArrayDataset(Dataset):
    """A minimal PyTorch Dataset wrapping CIFAR image arrays and labels."""

    def __init__(self, data: np.ndarray, labels: Sequence[int]):
        self.images = torch.from_numpy(_array_to_nchw_float(data))
        self.labels = torch.from_numpy(_labels_to_int64(labels, expected_len=len(self.images)))

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def _split_lengths(total: int, splits: Sequence[float | int]) -> List[int]:
    """Convert ratio or count splits into integer lengths that sum to total."""

    if total <= 0:
        raise ValueError("Cannot split an empty dataset.")
    if len(splits) != 3:
        raise ValueError("Expected exactly three splits: train, val, test.")

    if all(isinstance(value, int) for value in splits):
        lengths = [int(value) for value in splits]
        if sum(lengths) != total:
            raise ValueError(f"Integer split lengths must sum to {total}, got {lengths}.")
        return lengths

    ratios = np.asarray(splits, dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError(f"Split ratios must be non-negative, got {splits}.")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {splits}.")

    raw_lengths = ratios * total
    lengths = np.floor(raw_lengths).astype(int)
    remainder = total - int(lengths.sum())
    if remainder > 0:
        order = np.argsort(-(raw_lengths - lengths))
        for idx in order[:remainder]:
            lengths[idx] += 1
    return lengths.tolist()


def make_splits(
    dataset: Dataset,
    train: float | int = 0.8,
    val: float | int = 0.1,
    test: float | int = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Create reproducible train/val/test subsets."""

    lengths = _split_lengths(len(dataset), [train, val, test])
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, lengths, generator=generator)
    return train_set, val_set, test_set


def make_dataloaders(
    data_path: str | Path | None = None,
    dataset: Dataset | None = None,
    batch_size: int = 128,
    train: float | int = 0.8,
    val: float | int = 0.1,
    test: float | int = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """Build reproducible DataLoaders for later training and evaluation scripts."""

    if dataset is None:
        if data_path is None:
            raise ValueError("Either data_path or dataset must be provided.")
        images, labels = load_cifar_array_dataset(data_path)
        dataset = CIFARArrayDataset(images, labels)

    train_set, val_set, test_set = make_splits(dataset, train=train, val=val, test=test, seed=seed)
    train_generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return {
        "train": DataLoader(train_set, shuffle=True, generator=train_generator, **common),
        "val": DataLoader(val_set, shuffle=False, **common),
        "test": DataLoader(test_set, shuffle=False, **common),
    }
