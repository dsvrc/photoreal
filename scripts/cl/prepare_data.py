"""
Inventory, sanity checks and a continual-learning manifest for the audio2photoreal
body-motion benchmark (one session per subject).

It reproduces the scene filtering of data_loaders/get_data.py and the train/val/test
split of data_loaders/data.py (last 6 scenes: 2 val, 4 test), so the manifest matches
exactly what the training code sees.

    python scripts/cl/prepare_data.py --data_root dataset --out cl_data

Writes <out>/manifest.json with, per subject: scene splits, scene lengths, fixed
training windows (the replay-candidate pool), and the val/test chunks used for evaluation.
"""

import argparse
import json
import os

import numpy as np
import torch
import torchaudio

FPS = 30
AUDIO_PER_FRAME = 1600  # 48 kHz audio / 30 fps motion
# partner mapping used by flip_person in data_loaders/get_data.py
PAIRS = [("PXB184", "RLW104"), ("TXB805", "GQS883")]


def list_scenes(person_dir):
    """Scene prefixes in the loader's order, skipping the scenes the loader skips."""
    scenes = []
    for name in sorted(os.listdir(person_dir)):
        if not name.endswith("_body_pose.npy"):
            continue
        base = os.path.join(person_dir, name[: -len("_body_pose.npy")])
        face_len = len(np.load(base + "_face_expression.npy", mmap_mode="r"))
        missing = np.load(base + "_missing_face_frames.npy")
        if len(missing) == face_len:
            print(f"  skip {os.path.basename(base)}: all face frames missing (the loader skips it too)")
            continue
        scenes.append(os.path.basename(base))
    return scenes


def load_pose(person, person_dir, scene):
    pose = np.load(os.path.join(person_dir, scene + "_body_pose.npy"))
    if person in ("PXB184", "RLW104"):  # same wrap-around fix as get_data.py
        pose[:, 3] = (pose[:, 3] + np.pi) % (2 * np.pi)
        pose[:, 3] = (pose[:, 3] + np.pi) % (2 * np.pi)
    return pose


def chunk_starts(length, window):
    """Non-overlapping chunks exactly as Social._chunk_data builds the test set."""
    return list(range(0, length - window, window))


def window_starts(length, window, stride):
    return list(range(0, length - window + 1, stride))


def inspect_subject(person, root, window, stride):
    person_dir = os.path.join(root, person)
    if not os.path.isdir(person_dir) or not any(
        f.endswith("_body_pose.npy") for f in os.listdir(person_dir)
    ):
        print(f"[{person}] NOT DOWNLOADED (only data_stats.pth or nothing in {person_dir})")
        return None

    print(f"[{person}]")
    scenes = list_scenes(person_dir)
    n = len(scenes)
    if n < 7:
        print(f"  only {n} usable scenes; the repo split needs at least 7")
        return None
    split = {"train": scenes[: n - 6], "val": scenes[n - 6 : n - 4], "test": scenes[n - 4 :]}

    frames, problems = {}, []
    for scene in scenes:
        pose = np.load(os.path.join(person_dir, scene + "_body_pose.npy"), mmap_mode="r")
        info = torchaudio.info(os.path.join(person_dir, scene + "_audio.wav"))
        frames[scene] = int(pose.shape[0])
        if pose.shape[1] != 104:
            problems.append(f"{scene}: pose dim {pose.shape[1]} != 104")
        if info.sample_rate != 48000 or info.num_channels != 2:
            problems.append(f"{scene}: audio {info.sample_rate} Hz x {info.num_channels} ch (expected 48000 x 2)")
        if info.num_frames != pose.shape[0] * AUDIO_PER_FRAME:
            problems.append(
                f"{scene}: audio {info.num_frames} samples != {pose.shape[0]} frames x {AUDIO_PER_FRAME}"
                " (the loader asserts on this)"
            )
        if not np.isfinite(np.asarray(pose)).all():
            problems.append(f"{scene}: NaN/inf in body pose")

    def hours(names):
        return sum(frames[s] for s in names) / FPS / 3600

    entry = {
        "scenes": split,
        "frames": frames,
        "hours": {k: round(hours(v), 3) for k, v in split.items()},
        "train_windows": [[s, st] for s in split["train"] for st in window_starts(frames[s], window, stride)],
        "val_chunks": [[s, st] for s in split["val"] for st in chunk_starts(frames[s], window)],
        "test_chunks": [[s, st] for s in split["test"] for st in chunk_starts(frames[s], window)],
        "problems": problems,
    }
    print(
        f"  scenes train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}"
        f" | hours {entry['hours']}"
        f" | train windows {len(entry['train_windows'])} (10% replay = {len(entry['train_windows']) // 10})"
        f" | val chunks {len(entry['val_chunks'])} | test chunks {len(entry['test_chunks'])}"
    )
    for p in problems:
        print(f"  PROBLEM {p}")
    return entry


def normalization_check(root, subjects, manifest, ref):
    """How far each subject's train poses fall outside the session-0 subject's normalizer."""
    ref_stats = torch.load(os.path.join(root, ref, "data_stats.pth"), map_location="cpu", weights_only=False)
    ref_mean = np.asarray(ref_stats["pose_mean"]).reshape(-1)
    ref_std = np.asarray(ref_stats["pose_std"]).reshape(-1)
    print(f"\n[normalization] reference = {ref} stats; smallest pose std = {ref_std.min():.2e}")
    report = {}
    for person in subjects:
        own = torch.load(os.path.join(root, person, "data_stats.pth"), map_location="cpu", weights_only=False)
        own_mean = np.asarray(own["pose_mean"]).reshape(-1)
        own_std = np.asarray(own["pose_std"]).reshape(-1)
        max_ref = np.zeros(104)
        max_own = np.zeros(104)
        over10, total = 0, 0
        for scene in manifest[person]["scenes"]["train"]:
            pose = load_pose(person, os.path.join(root, person), scene)
            z_ref = np.abs((pose - ref_mean) / ref_std)
            z_own = np.abs((pose - own_mean) / own_std)
            max_ref = np.maximum(max_ref, z_ref.max(0))
            max_own = np.maximum(max_own, z_own.max(0))
            over10 += int((z_ref > 10).sum())
            total += z_ref.size
        worst = np.argsort(-max_ref)[:5]
        report[person] = {
            "max_abs_z_ref": float(max_ref.max()),
            "max_abs_z_own": float(max_own.max()),
            "frac_abs_z_ref_over_10": over10 / max(total, 1),
            "worst_dims_ref": [[int(d), float(max_ref[d])] for d in worst],
            "audio_stats_own": {
                "audio_mean": np.asarray(own["audio_mean"]).reshape(-1).round(5).tolist(),
                "audio_std_flat": np.asarray(own["audio_std_flat"]).reshape(-1).round(5).tolist(),
            },
        }
        print(
            f"  {person}: max|z| with {ref} stats = {max_ref.max():8.2f} (own stats {max_own.max():8.2f}),"
            f" entries with |z|>10: {100 * over10 / max(total, 1):.3f}%, worst dims {report[person]['worst_dims_ref']}"
        )
    return report


def pairing_check(root, manifest):
    """Do dyadic partners share recordings (A's partner channel == B's own channel)?"""
    print("\n[pairing] correlation of A.channel1 with B.channel0 on shared scene names")
    report = {}
    for a, b in PAIRS:
        if a not in manifest or b not in manifest:
            continue
        shared = sorted(set(manifest[a]["frames"]) & set(manifest[b]["frames"]))
        rows = []
        for scene in shared[:3]:
            length = min(manifest[a]["frames"][scene], manifest[b]["frames"][scene]) * AUDIO_PER_FRAME
            offset = 60 * 48000 if length > 75 * 48000 else 0
            num = min(10 * 48000, length - offset)
            wa, _ = torchaudio.load(os.path.join(root, a, scene + "_audio.wav"), frame_offset=offset, num_frames=num)
            wb, _ = torchaudio.load(os.path.join(root, b, scene + "_audio.wav"), frame_offset=offset, num_frames=num)
            r = float(np.corrcoef(wa[1].numpy(), wb[0].numpy())[0, 1])
            same_len = manifest[a]["frames"][scene] == manifest[b]["frames"][scene]
            rows.append({"scene": scene, "corr_a1_b0": r, "same_length": same_len})
            print(f"  {a}<->{b} {scene}: corr = {r:.3f}, same number of frames = {same_len}")
        if not shared:
            print(f"  {a}<->{b}: no scene names in common")
        report[f"{a}-{b}"] = {"shared_scene_names": len(shared), "checked": rows}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--out", default="cl_data")
    parser.add_argument("--order", default="PXB184,RLW104,TXB805,GQS883", help="session order")
    parser.add_argument("--window", type=int, default=600, help="frames per replay candidate / eval chunk")
    parser.add_argument("--stride", type=int, default=300, help="stride between replay-candidate windows")
    args = parser.parse_args()

    order = args.order.split(",")
    manifest = {}
    for person in order:
        entry = inspect_subject(person, args.data_root, args.window, args.stride)
        if entry is not None:
            manifest[person] = entry
    missing = [p for p in order if p not in manifest]
    if missing:
        print(f"\nMissing subjects {missing}: download them before building the benchmark.")
    available = [p for p in order if p in manifest]

    out = {
        "order": order,
        "window": args.window,
        "stride": args.stride,
        "fps": FPS,
        "subjects": manifest,
    }
    if available:
        out["normalization"] = normalization_check(args.data_root, available, manifest, ref=available[0])
        out["pairing"] = pairing_check(args.data_root, manifest)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "manifest.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {path}")
    if any(manifest[p]["problems"] for p in manifest):
        print("Some scenes have PROBLEM lines above; fix or exclude them before training.")


if __name__ == "__main__":
    main()
