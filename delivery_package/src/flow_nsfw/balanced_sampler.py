"""Balanced batch sampler — ensures each batch has NSFW + SFW pairs.

Reads labels from manifest JSON (fast), not from dataset iteration (slow).
"""

from __future__ import annotations

import json
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[int]):
    """Each batch contains exactly one NSFW and one SFW sample."""

    def __init__(self, manifest_path: str, split: str = "train",
                 batch_size: int = 2, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Read labels from manifest (fast — no file I/O)
        manifest = json.load(open(manifest_path))
        entries = [v for v in manifest if v.get("split") == split]
        self.nsfw_idx: list[int] = []
        self.sfw_idx: list[int] = []
        for i, v in enumerate(entries):
            if v.get("label") == 1:
                self.nsfw_idx.append(i)
            else:
                self.sfw_idx.append(i)

        self.num_batches = min(len(self.nsfw_idx), len(self.sfw_idx))
        print(f"[balanced] {len(self.nsfw_idx)} NSFW + {len(self.sfw_idx)} SFW -> {self.num_batches} batches")

    def __iter__(self) -> Iterator[list[int]]:
        nsfw = self.nsfw_idx.copy()
        sfw = self.sfw_idx.copy()

        if self.shuffle:
            np.random.shuffle(nsfw)
            np.random.shuffle(sfw)

        batches = []
        for i in range(self.num_batches):
            batches.append([nsfw[i % len(nsfw)], sfw[i % len(sfw)]])

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return self.num_batches
