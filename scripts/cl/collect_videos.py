"""
Generate and pack everything needed to render the fine-tuning vs TMR conversation videos on another machine.
One GPU job, from the repo root in a2p_env:

    python scripts/cl/collect_videos.py

Step 1 runs cl.evaluate --save_samples (audio only, eval seed 0: the same noise as the paper's numbers) on the
released Session-0 model and on fine-tuning and TMR after Sessions 1-3, for the subjects seen so far. The
generated and ground-truth body poses of every test chunk go to cl_data/video_samples/<method>_s<k>/; the
runs' own eval files are not touched, and a checkpoint whose samples exist is skipped. Step 2 (CPU) packs them.
Copy cl_data/video_pack.tar.gz to the rendering machine. The pack holds:

  samples/<method>_s<k>/<subject>.npz  pred, gt: [chunks x 600 x 104] joint angles (raw, as the renderer
                                       takes them), from the evaluation of the model after session k
  chunks/<subject>/cNN_audio.wav       the chunk's two-channel 48 kHz audio (participant, partner)
  chunks/<subject>/cNN_face.npy        ground-truth face codes [600 x 256]: the body model does not
                                       generate faces, so every video of a chunk uses the same face
  chunks/<subject>/cNN_partner_*.npy   the conversation partner's real body and face codes in the same
                                       scene and frames, if the partner was recorded there (two-person view)
  chunks/<subject>/chunks.json         scene and start frame of every chunk
  scores.csv                           per chunk and checkpoint: mean squared error to the ground truth and
                                       ground-truth motion energy, to pick clips by a fixed rule
  ca_body/<subject>/                   the photoreal renderer of the subject (body_dec.ckpt, config.yml,
                                       static_assets.pt), unless --no_renderer
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ORDER = ["PXB184", "RLW104", "TXB805", "GQS883"]
PARTNER = {"PXB184": "RLW104", "RLW104": "PXB184", "TXB805": "GQS883", "GQS883": "TXB805"}
AUDIO_PER_FRAME = 1600
WRAP_SUBJECTS = ("PXB184", "RLW104")  # same pose fix as data_loaders/get_data.py and cl.common.load_window


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples_dir", default="cl_data/video_samples")
    p.add_argument("--manifest", default="cl_data/manifest.json")
    p.add_argument("--dataset_root", default="dataset")
    p.add_argument("--renderer_root", default="checkpoints/ca_body/data")
    p.add_argument("--out", default="cl_data/video_pack")
    p.add_argument("--subjects", nargs="*", default=None, help="default: every subject found in the samples")
    p.add_argument("--chunks", type=int, nargs="*", default=None, help="chunk indices to pack (default: all)")
    p.add_argument("--no_renderer", action="store_true", help="skip ca_body (already on the rendering machine)")
    p.add_argument("--no_tar", action="store_true")
    p.add_argument("--no_generate", action="store_true", help="only pack existing samples")
    p.add_argument("--methods", nargs="+", default=["vanilla", "tmr"])
    p.add_argument("--runs", default="runs")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--s0", default="checkpoints/diffusion/c1_pose/model000340000.pt")
    return p.parse_args()


def load_pose(path, subject, start, length):
    pose = np.array(np.load(path, mmap_mode="r")[start : start + length], dtype=np.float32)
    if subject in WRAP_SUBJECTS:
        pose[:, 3] = (pose[:, 3] + np.pi) % (2 * np.pi)
        pose[:, 3] = (pose[:, 3] + np.pi) % (2 * np.pi)
    return pose


def generate(a):
    """cl.evaluate --save_samples on every checkpoint, for the subjects seen so far (skips what exists)."""
    import subprocess

    for method in a.methods:
        for k in range(len(ORDER)):
            if k == 0 and method != a.methods[0]:
                continue  # session 0 is the shared released model: once
            ckpt = a.s0 if k == 0 else os.path.join(a.runs, method, f"s{k}_{ORDER[k]}", f"model{a.steps:09d}.pt")
            tag = "s0_s0" if k == 0 else f"{method}_s{k}"
            if not os.path.exists(ckpt):
                print(f"skip {tag}: no {ckpt}")
                continue
            print(f"\n== generating {tag}: {ckpt}", flush=True)
            subprocess.run(
                [sys.executable, "-m", "cl.evaluate", "--ckpt", ckpt, "--subjects", *ORDER[: k + 1],
                 "--modes", "audio_only", "--save_samples", "--manifest", a.manifest,
                 "--out_dir", os.path.join(a.samples_dir, tag)],
                check=True,
            )


def main():
    a = parse_args()
    if not a.no_generate:
        generate(a)
    import torchaudio  # server env

    with open(a.manifest) as f:
        manifest = json.load(f)
    window = manifest["window"]
    os.makedirs(a.out, exist_ok=True)

    # 1) generated samples
    runs = {}
    for d in sorted(glob.glob(os.path.join(a.samples_dir, "*_s[0-9]"))):
        tag = os.path.basename(d)
        for npz in sorted(glob.glob(os.path.join(d, "*_test_audio_only_seed*.npz"))):
            subject = next((s for s in ORDER if f"_{s}_test_" in os.path.basename(npz)), None)
            if subject is None:
                continue
            dst = os.path.join(a.out, "samples", tag, f"{subject}.npz")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(npz, dst)
            runs.setdefault(subject, {})[tag] = dst
            print(f"samples {tag} {subject}: {npz}")
    if not runs:
        sys.exit(f"no samples in {a.samples_dir}: run the cl.evaluate --save_samples lines of video_commands.txt first")
    subjects = a.subjects or [s for s in ORDER if s in runs]

    # 2) audio, face codes and the partner of every chunk
    rows = []
    for subject in subjects:
        chunks = manifest["subjects"][subject]["test_chunks"]
        keep = a.chunks if a.chunks is not None else list(range(len(chunks)))
        root = os.path.join(a.dataset_root, subject)
        proot = os.path.join(a.dataset_root, PARTNER[subject])
        out = os.path.join(a.out, "chunks", subject)
        os.makedirs(out, exist_ok=True)
        meta = []
        for c in keep:
            scene, start = chunks[c][0], int(chunks[c][1])
            audio, sr = torchaudio.load(os.path.join(root, scene + "_audio.wav"),
                                        frame_offset=start * AUDIO_PER_FRAME, num_frames=window * AUDIO_PER_FRAME)
            torchaudio.save(os.path.join(out, f"c{c:02d}_audio.wav"), audio, sr, encoding="PCM_S", bits_per_sample=16)
            face = np.load(os.path.join(root, scene + "_face_expression.npy"), mmap_mode="r")[start : start + window]
            np.save(os.path.join(out, f"c{c:02d}_face.npy"), np.asarray(face, dtype=np.float32))
            partner = False
            pb, pf = os.path.join(proot, scene + "_body_pose.npy"), os.path.join(proot, scene + "_face_expression.npy")
            if os.path.exists(pb) and os.path.exists(pf) and len(np.load(pb, mmap_mode="r")) >= start + window:
                np.save(os.path.join(out, f"c{c:02d}_partner_body.npy"), load_pose(pb, PARTNER[subject], start, window))
                pface = np.load(pf, mmap_mode="r")[start : start + window]
                np.save(os.path.join(out, f"c{c:02d}_partner_face.npy"), np.asarray(pface, dtype=np.float32))
                partner = True
            meta.append({"chunk": c, "scene": scene, "start": start, "partner": PARTNER[subject] if partner else None})
        with open(os.path.join(out, "chunks.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"chunks {subject}: {len(meta)} of {len(chunks)}, partner body for {sum(m['partner'] is not None for m in meta)}")

        # 3) per-chunk scores to choose clips by a fixed rule
        for tag, path in sorted(runs.get(subject, {}).items()):
            d = np.load(path)
            pred, gt = d["pred"], d["gt"]
            for c in keep:
                rows.append({
                    "subject": subject, "run": tag, "chunk": c, "scene": chunks[c][0], "start": chunks[c][1],
                    "mse_to_gt": float(np.mean((pred[c] - gt[c]) ** 2)),
                    "gt_motion_energy": float(np.mean(np.diff(gt[c], axis=0) ** 2)),
                    "pred_motion_energy": float(np.mean(np.diff(pred[c], axis=0) ** 2)),
                })
    with open(os.path.join(a.out, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 4) renderer of every subject and partner used
    if not a.no_renderer:
        for s in sorted(set(subjects) | {PARTNER[s] for s in subjects}):
            src = os.path.join(a.renderer_root, s)
            if not os.path.isdir(src):
                print(f"WARNING: no renderer at {src}")
                continue
            dst = os.path.join(a.out, "ca_body", s)
            if not os.path.isdir(dst):
                shutil.copytree(src, dst)
            print(f"renderer {s}: {src}")

    if not a.no_tar:
        tar = a.out + ".tar.gz"
        with tarfile.open(tar, "w:gz") as t:
            t.add(a.out, arcname=os.path.basename(a.out))
        print(f"\nwritten {tar} ({os.path.getsize(tar) / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
