"""
Simple TSP data generator (x, y only) — DDP-friendly version
----------------------------------------------------------------
- Keeps original APIs: generate/save/load + SimpleTSPDataset
- Adds `build_dataloader(...)` that returns a standard torch DataLoader (supports `sampler=`)
- Makes `SimpleTSPDataLoader` backward compatible and DDP-aware:
  * If `sampler` is provided, it wraps a torch DataLoader internally
  * Otherwise, it behaves like the original lightweight Python generator

Recommended: in DDP use `build_dataloader(...)` with a DistributedSampler.
"""

from __future__ import annotations
import os
from typing import Optional, Tuple, Dict, Any, Iterator

import torch
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader, DistributedSampler, RandomSampler

__all__ = [
    'generate_tsp_data', 'coords_to_distance_matrix', 'save_tsp_dataset', 'load_tsp_dataset',
    'SimpleTSPDataset', 'SimpleTSPDataLoader', 'build_dataloader'
]

# ------------------------- data io & transforms ------------------------- #

def generate_tsp_data(num_instances: int, num_cities: int, coord_range: float = 100.0, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    coordinates = torch.rand(num_instances, num_cities, 2) * coord_range
    return coordinates


def coords_to_distance_matrix(coordinates: torch.Tensor) -> torch.Tensor:
    return torch.cdist(coordinates, coordinates)


def save_tsp_dataset(coordinates: torch.Tensor, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    save_data: Dict[str, Any] = {
        'coordinates': coordinates,
        'num_instances': coordinates.shape[0],
        'num_cities': coordinates.shape[1],
        'coord_range': float(coordinates.max().item())
    }
    if metadata:
        save_data.update(metadata)
    torch.save(save_data, filepath)
    print(f"Dataset saved to: {filepath}")
    print(f"Shape: {coordinates.shape}")
    print(f"Instances: {coordinates.shape[0]}, Cities: {coordinates.shape[1]}")


def load_tsp_dataset(filepath: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    data: Dict[str, Any] = torch.load(filepath, weights_only=False)
    coordinates: torch.Tensor = data['coordinates']
    print(f"Dataset loaded from: {filepath}")
    print(f"Shape: {coordinates.shape}")
    print(f"Instances: {data['num_instances']}, Cities: {data['num_cities']}")
    return coordinates, data


# ------------------------------- dataset -------------------------------- #

class SimpleTSPDataset(torch.utils.data.Dataset):
    """Dataset that returns only coordinate tensors of shape (N, 2)."""
    def __init__(self, coordinates: torch.Tensor):
        self.coordinates = coordinates
        print("Creating simple TSP dataset...")
        print(f"Coordinates: {coordinates.shape}")
    def __len__(self) -> int:
        return int(self.coordinates.shape[0])
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.coordinates[idx]
    def get_batch(self, indices: torch.Tensor) -> torch.Tensor:
        return self.coordinates[indices]


# ---------------------------- dataloaders -------------------------------- #

def _default_collate(batch: list[torch.Tensor]) -> torch.Tensor:
    # batch of tensors with shape (N, 2) -> stack to (B, N, 2)
    return torch.stack(batch, dim=0)


def build_dataloader(
    dataset: SimpleTSPDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    sampler: Optional[torch.utils.data.Sampler] = None,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> TorchDataLoader:
    """DDP-friendly factory returning torch DataLoader (supports `sampler=`)."""
    if sampler is not None and shuffle:
        # When a sampler is supplied, DataLoader requires shuffle=False
        shuffle = False
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_default_collate,
    )


class SimpleTSPDataLoader:
    """
    Backward-compatible lightweight DataLoader.
    - If `sampler` is provided, it wraps a standard torch DataLoader internally (DDP-safe).
    - Otherwise, it behaves like the original generator-style loader.
    """
    def __init__(
        self,
        dataset: SimpleTSPDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        sampler: Optional[torch.utils.data.Sampler] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self._torch_loader: Optional[TorchDataLoader] = None

        # If sampler is given (e.g., DistributedSampler in DDP), switch to torch DataLoader
        if sampler is not None:
            if shuffle:
                print("[SimpleTSPDataLoader] `sampler` provided -> forcing shuffle=False (sampler controls order)")
            self._torch_loader = build_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers if num_workers is not None else 0,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self._torch_loader is not None:
            for batch in self._torch_loader:
                yield batch
            return
        # Original generator behavior
        indices = torch.randperm(self.num_samples) if self.shuffle else torch.arange(self.num_samples)
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.dataset.get_batch(batch_indices)

    def __len__(self) -> int:
        if self._torch_loader is not None:
            return len(self._torch_loader)
        return self.num_batches


# ----------------------- convenience creators/tests ---------------------- #

def create_standard_datasets() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("=== Creating Standard TSP Datasets (2 features only) ===")
    num_cities = 20; coord_range = 100.0
    print("\n1. Generating training set...")
    train_coords = generate_tsp_data(200_000, num_cities, coord_range, seed=42)
    save_tsp_dataset(train_coords, f'data/tsp_{num_cities}_uniform_train.pt',
                     {'split': 'train', 'description': 'Training set with 2 features (x,y)'})
    print("\n2. Generating validation set...")
    val_coords = generate_tsp_data(1_000, num_cities, coord_range, seed=123)
    save_tsp_dataset(val_coords, f'data/tsp_{num_cities}_uniform_val.pt',
                     {'split': 'val', 'description': 'Validation set with 2 features (x,y)'})
    print("\n3. Generating test set...")
    test_coords = generate_tsp_data(1_000, num_cities, coord_range, seed=456)
    save_tsp_dataset(test_coords, f'data/tsp_{num_cities}_uniform_test.pt',
                     {'split': 'test', 'description': 'Test set with 2 features (x,y)'})
    print("\n=== Dataset Creation Complete ===")
    return train_coords, val_coords, test_coords


def test_dataset_loading() -> None:
    print("\n=== Testing Dataset Loading ===")
    train_coords, _ = load_tsp_dataset('data/tsp_train.pt')
    dataset = SimpleTSPDataset(train_coords[:100])
    # Non-DDP path (old behavior)
    d1 = SimpleTSPDataLoader(dataset, batch_size=16, shuffle=True)
    # DDP-like path (uses torch DataLoader inside)
    d2 = SimpleTSPDataLoader(dataset, batch_size=16, shuffle=True, sampler=RandomSampler(dataset))

    for name, dl in [('orig', d1), ('torch', d2)]:
        print(f"\n[name={name}] num_batches={len(dl)}")
        for bi, batch in enumerate(dl):
            print(f"  batch{bi}: {tuple(batch.shape)}")
            assert batch.dim() == 3 and batch.shape[2] == 2
            break

    print("\n✓ Dataset loading test passed!")


if __name__ == '__main__':
    # quick self-test
    coords = generate_tsp_data(32, 20, 100.0, seed=7)
    ds = SimpleTSPDataset(coords)
    dl = build_dataloader(ds, batch_size=8, shuffle=True)
    for b in dl:
        print(b.shape)
        break

