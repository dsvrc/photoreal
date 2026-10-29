"""Shared helpers: checkpoint args, model loading, per-subject stats and fixed windows."""

import argparse
import json
import os

import numpy as np
import torch
import torchaudio

from utils.model_util import create_model_and_diffusion, load_model

AUDIO_PER_FRAME = 1600  # 48 kHz audio / 30 fps motion
WRAP_SUBJECTS = ("PXB184", "RLW104")  # get_data.py fixes a wrap-around in pose dim 3 for these


def load_ckpt_args(ckpt_path: str) -> argparse.Namespace:
    """Model/data/diffusion args saved next to a checkpoint (args.json)."""
    with open(os.path.join(os.path.dirname(ckpt_path), "args.json")) as f:
        args = argparse.Namespace(**json.load(f))
    args.model_path = ckpt_path  # create_gaussian_diffusion needs save_dir or model_path
    return args


def build_model(args: argparse.Namespace, ckpt_path: str, split_type: str = "train"):
    model, diffusion = create_model_and_diffusion(args, split_type=split_type)
    load_model(model, torch.load(ckpt_path, map_location="cpu"))
    return model, diffusion


def load_stats(subject_root: str) -> dict:
    """The normalization the repo uses for this subject (dataset/<subject>/data_stats.pth)."""
    stats = torch.load(os.path.join(subject_root, "data_stats.pth"), map_location="cpu", weights_only=False)
    return {
        "pose_mean": np.asarray(stats["pose_mean"]).reshape(-1),
        "pose_std": np.asarray(stats["pose_std"]).reshape(-1),
        "audio_mean": np.asarray(stats["audio_mean"]),
        "audio_std": np.asarray(stats["audio_std_flat"]),
    }


def load_window(dataset_root: str, subject: str, scene: str, start: int, length: int):
    """Raw (un-normalized) pose [length x 104] and audio [length*1600 x 2] of one window."""
    subject_root = os.path.join(dataset_root, subject)
    pose = np.array(
        np.load(os.path.join(subject_root, scene + "_body_pose.npy"), mmap_mode="r")[start : start + length],
        dtype=float,
    )
    if subject in WRAP_SUBJECTS:
        pose[:, 3] = (pose[:, 3] + np.pi) % (2 * np.pi)
        pose[:, 3] = (pose[:, 3] + np.pi) % (2 * np.pi)
    audio, _ = torchaudio.load(
        os.path.join(subject_root, scene + "_audio.wav"),
        frame_offset=start * AUDIO_PER_FRAME,
        num_frames=length * AUDIO_PER_FRAME,
    )
    audio = audio.T.numpy()
    assert len(pose) == length and len(audio) == length * AUDIO_PER_FRAME, (
        f"{subject}/{scene}@{start}: got {len(pose)} frames / {len(audio)} audio samples"
    )
    return pose, audio


def subject_of(data_root: str) -> str:
    return os.path.basename(os.path.normpath(data_root))
