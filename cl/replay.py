"""Replay buffers: fixed 600-frame windows of earlier subjects, with per-sample weights.

A buffer file is JSON (clip references, not tensors):
    {"subject": "PXB184", "method": "tmr", "entries": [
        {"subject": "PXB184", "scene": "scene01", "start": 600, "weight": 1.0}, ...], ...}
Each window is normalized with its own subject's data_stats.pth, like the training loader.
"""

import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from cl.common import load_stats, load_window
from data_loaders.tensors import social_collate


def write_buffer(path: str, subject: str, method: str, clips, idx, weights, **meta) -> None:
    """Save the selected clips (scene, start) with weights rescaled to mean 1."""
    weights = np.asarray(weights, dtype=float)
    weights = weights * len(weights) / weights.sum()
    entries = [
        {"subject": subject, "scene": str(clips[i][0]), "start": int(clips[i][1]), "weight": float(w)}
        for i, w in zip(idx, weights)
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"subject": subject, "method": method, **meta, "entries": entries}, f, indent=1)


def read_buffers(paths: List[str]) -> List[Dict]:
    entries = []
    for path in paths:
        with open(path) as f:
            entries.extend(json.load(f)["entries"])
    return entries


class ShuffledStream(Sampler):
    """num_samples indices from back-to-back random permutations: the shuffled union of the buffers,
    every clip once per pass (also when the union is smaller than a batch)."""

    def __init__(self, n: int, num_samples: int, seed: int = 0):
        self.n, self.num_samples, self.seed = n, num_samples, seed

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        produced = 0
        while produced < self.num_samples:
            for i in torch.randperm(self.n, generator=g).tolist()[: self.num_samples - produced]:
                yield i
            produced = min(self.num_samples, produced + self.n)


class ReplayWindows(Dataset):
    def __init__(
        self, entries: List[Dict], dataset_root: str, window: int = 600, keyframe_step: int = 30, preload: bool = False
    ):
        self.entries = entries
        self.dataset_root = dataset_root
        self.window = window
        self.keyframe_step = keyframe_step
        subjects = sorted({e["subject"] for e in entries})
        self.stats = {s: load_stats(os.path.join(dataset_root, s)) for s in subjects}
        # buffers are small (~8 MB per window): keep them normalized in memory instead of reading every step
        self.cache = [self._load(e) for e in entries] if preload else None

    def _load(self, e: Dict):
        stats = self.stats[e["subject"]]
        pose, audio = load_window(self.dataset_root, e["subject"], e["scene"], int(e["start"]), self.window)
        motion = (pose - stats["pose_mean"]) / stats["pose_std"]
        audio = ((audio - stats["audio_mean"]) / stats["audio_std"]).astype(np.float32)
        return motion, audio

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Dict:
        e = self.entries[i]
        motion, audio = self.cache[i] if self.cache is not None else self._load(e)
        keyframes = motion[:: self.keyframe_step]
        return {
            "motion": motion,
            "m_length": len(motion),
            "audio": audio,
            "a_length": len(audio),
            "keyframes": keyframes,
            "k_length": len(keyframes),
            "missing": np.ones_like(motion),
            "weight": float(e.get("weight", 1.0)),
        }


def replay_collate(batch):
    motion, cond = social_collate(batch)
    weights = torch.tensor([b["weight"] for b in batch], dtype=torch.float32)
    return motion, cond, weights
