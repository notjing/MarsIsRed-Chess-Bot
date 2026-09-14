
from __future__ import annotations

import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from model.shard_io import ShardReader, shard_len


def list_shard_files(directory, exclude_self_gen: bool = False) -> list[str]:
    if not directory or not os.path.isdir(directory):
        return []
    files = sorted(glob.glob(os.path.join(directory, "*.bin")))
    if exclude_self_gen:
        files = [f for f in files if "self_gen" not in Path(f).parts]
    return [f for f in files if shard_len(f) > 0]


class ShardDataset(Dataset):

    def __init__(self, files):
        self.files = [str(f) for f in files]
        if not self.files:
            raise FileNotFoundError("ShardDataset got no shard files")
        self.lengths = [shard_len(f) for f in self.files]
        if sum(self.lengths) == 0:
            raise FileNotFoundError("ShardDataset files contain 0 records")
        self.sample_psa = np.cumsum(self.lengths)
        self._readers: list[ShardReader] | None = None

    def __len__(self) -> int:
        return int(self.sample_psa[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_readers"] = None
        return state

    def close(self) -> None:
        if self._readers is None:
            return
        for reader in self._readers:
            reader.close()
        self._readers = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._readers is None:
            self._readers = [ShardReader(f) for f in self.files]

    # get a specific data point out of the dataset
    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        self._ensure_open()
        shard_i = int(np.searchsorted(self.sample_psa, idx, side="right"))
        local = idx if shard_i == 0 else idx - int(self.sample_psa
                                                   [shard_i - 1])
        board_nhwc, extra, wdl, policy = self._readers[shard_i][local]
        board = torch.from_numpy(board_nhwc).permute(2, 0, 1).contiguous()
        return (
            board,
            torch.from_numpy(extra),
            torch.from_numpy(wdl),
            torch.from_numpy(policy),
        )


class MixedShardDataset(Dataset):
    """Per-sample mix: supervised with probability `supervised_weight`, else self-play[i]."""

    def __init__(self, self_play: ShardDataset, supervised: ShardDataset | None, supervised_weight: float, seed: int = 0):
        self.self_play = self_play
        self.supervised = supervised
        self.supervised_weight = float(supervised_weight)
        self._base_seed = seed
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.self_play)

    def reseed(self, worker_id: int) -> None:
        self._rng = random.Random(self._base_seed + (worker_id + 1) * 997)

    def close(self) -> None:
        self.self_play.close()
        if self.supervised is not None:
            self.supervised.close()

    def __getitem__(self, idx: int):
        if self.supervised is not None and self.supervised_weight > 0.0 and self._rng.random() < self.supervised_weight:
            j = self._rng.randrange(len(self.supervised))
            return self.supervised[j]
        return self.self_play[idx]


class _InfiniteRandomSampler(Sampler):
    def __init__(self, n: int):
        if n <= 0:
            raise ValueError("dataset is empty")
        self.n = n

    def __iter__(self):
        while True:
            yield from torch.randperm(self.n).tolist()

    def __len__(self) -> int:
        return self.n


def _worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if hasattr(ds, "reseed"):
        ds.reseed(worker_id)


def make_dataloader(
    buffer_dir,
    batch_size,
    supervised_dir=None,
    supervised_weight=0.0,
    num_workers=0,
):
    sp_files = list_shard_files(buffer_dir)
    if not sp_files:
        raise FileNotFoundError(f"No .bin shards in replay buffer {buffer_dir}")

    self_play = ShardDataset(sp_files)
    sup_files = list_shard_files(supervised_dir, exclude_self_gen=True) if supervised_dir else []
    if supervised_weight > 0.0 and sup_files:
        ds: Dataset = MixedShardDataset(self_play, ShardDataset(sup_files), supervised_weight)
    else:
        ds = self_play

    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=_InfiniteRandomSampler(len(ds)),
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )
