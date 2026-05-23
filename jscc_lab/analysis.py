"""Shared helpers for AE latent-space analysis scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Subset

from .data import CIFARArrayDataset, load_cifar_array_dataset, make_splits
from .models import AutoEncoder


@dataclass
class SampleBatch:
    """Images sampled from the reproducible test split."""

    images: torch.Tensor
    labels: torch.Tensor
    original_indices: List[int]
    test_positions: List[int]


def load_autoencoder(checkpoint_path: str | Path, device: torch.device) -> AutoEncoder:
    """Load the AE architecture and restore weights from `train_ae.py` checkpoints."""

    model = AutoEncoder()
    path = Path(checkpoint_path).expanduser()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_test_split(
    data_path: str | Path,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> Tuple[CIFARArrayDataset, Subset]:
    """Load data and rebuild the same train/val/test split used during AE training."""

    images, labels = load_cifar_array_dataset(data_path)
    dataset = CIFARArrayDataset(images, labels)
    _, _, test_set = make_splits(
        dataset,
        train=train_split,
        val=val_split,
        test=test_split,
        seed=seed,
    )
    return dataset, test_set


def test_original_indices(test_set: Subset) -> List[int]:
    """Return original dataset indices represented by a random_split Subset."""

    if not hasattr(test_set, "indices"):
        return list(range(len(test_set)))
    return [int(index) for index in test_set.indices]


def read_selected_indices(path: str | Path) -> List[int]:
    """Read original dataset indices from `selected_indices.txt`.

    The parser accepts the tabular format written by `save_selected_indices`
    and also a plain file containing one integer index per line.
    """

    selected: List[int] = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        try:
            selected.append(int(parts[2] if len(parts) >= 3 else parts[0]))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Could not parse selected index line: {line!r}") from exc
    if not selected:
        raise ValueError(f"No selected indices found in {path}.")
    return selected


def save_selected_indices(path: str | Path, samples: SampleBatch) -> Path:
    """Save sampled original indices together with labels and test positions."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# order test_position original_index label"]
    for order, (test_pos, original_idx, label) in enumerate(
        zip(samples.test_positions, samples.original_indices, samples.labels.tolist())
    ):
        lines.append(f"{order} {test_pos} {original_idx} {int(label)}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def sample_test_items(
    test_set: Subset,
    num_samples: int,
    seed: int,
    selected_original_indices: Sequence[int] | None = None,
) -> SampleBatch:
    """Sample images from the test split, optionally by original dataset index."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    originals = test_original_indices(test_set)
    if selected_original_indices is None:
        generator = torch.Generator().manual_seed(seed)
        count = min(num_samples, len(test_set))
        positions = torch.randperm(len(test_set), generator=generator)[:count].tolist()
    else:
        index_to_position = {original_idx: pos for pos, original_idx in enumerate(originals)}
        missing = [idx for idx in selected_original_indices if idx not in index_to_position]
        if missing:
            raise ValueError(
                "Selected indices are not in the current test split. "
                f"First missing indices: {missing[:10]}"
            )
        positions = [index_to_position[int(idx)] for idx in selected_original_indices[:num_samples]]

    images = []
    labels = []
    sampled_originals = []
    for position in positions:
        image, label = test_set[int(position)]
        images.append(image)
        labels.append(int(label))
        sampled_originals.append(originals[int(position)])

    return SampleBatch(
        images=torch.stack(images, dim=0),
        labels=torch.tensor(labels, dtype=torch.long),
        original_indices=sampled_originals,
        test_positions=[int(pos) for pos in positions],
    )
