"""Replay buffers: fixed 600-frame windows of earlier subjects, with per-sample weights.

A buffer file is JSON:
    {"subject": "PXB184", "method": "er", "entries": [
        {"subject": "PXB184", "scene": "scene01", "start": 300, "weight": 1.0}, ...]}
Each window is normalized with its own subject's data_stats.pth, like the training loader.
"""

import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from cl.common import load_stats, load_window
from data_loaders.tensors import social_collate


def read_buffers(paths: List[str]) -> List[Dict]:
    entries = []
    for path in paths:
        with open(path) as f:
            entries.extend(json.load(f)["entries"])
    return entries


class ReplayWindows(Dataset):
    def __init__(self, entries: List[Dict], dataset_root: str, window: int = 600, keyframe_step: int = 30):
        self.entries = entries
        self.dataset_root = dataset_root
        self.window = window
        self.keyframe_step = keyframe_step
        subjects = sorted({e["subject"] for e in entries})
        self.stats = {s: load_stats(os.path.join(dataset_root, s)) for s in subjects}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Dict:
        e = self.entries[i]
        stats = self.stats[e["subject"]]
        pose, audio = load_window(self.dataset_root, e["subject"], e["scene"], int(e["start"]), self.window)
        motion = (pose - stats["pose_mean"]) / stats["pose_std"]
        audio = ((audio - stats["audio_mean"]) / stats["audio_std"]).astype(np.float32)
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
