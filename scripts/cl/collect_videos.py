"""
Collect what the local machine needs to render the qualitative Audio2Photoreal frames and videos (supplementary
Sec. app:qual). Nothing is rendered here. One GPU job, from the repo root in a2p_env:

    python scripts/cl/collect_videos.py

Rows (one participant's test audio, the same noise for every model, audio only as in Tab. a2p):
    Real                          the participant's recorded motion
    Session-0 model               the model right after the participant was learned (PXB184: the released model;
                                  a later participant: fine-tuning after its own session)
    Fine-tuning after Session k   every later session, k = 1..3: the forgetting
    TMR after Session 3           the paper's TMR (runs/tmr)
    TMR-joint after Session 3     runs/tmr_joint, if it exists (--no_joint to leave it out)

Steps:
  1 generate  cl.evaluate --save_samples for the participant on every row's checkpoint, skipped when the samples
              exist (the earlier run left most of them in cl_data/video_samples/); the runs' eval files are not
              touched
  2 choose    --num_chunks test chunks by a fixed rule that does not look at any model output: the largest motion
              energy of the real motion (mean squared frame-to-frame change); --chunks overrides it
  3 pack      cl_data/qual_pack.tar.gz with
                rows.npz     bodies [chunk, row, 600, 104] (raw joint angles, row 0 = Real), the real face codes
                             [chunk, 600, 256] used for every row (the body model makes no faces), labels, chunks
                audio/cNN.wav  the chunk's two-channel 48 kHz audio (participant, partner), 3.7 MB each
                scores.csv   every test chunk: real motion energy, and per row MSE and FD_g to the real motion
                README.txt   rows, checkpoints, chunks and the rule
                ca_body/<participant>/  only the renderer files the local renderer reads (config.yml,
                             body_dec.ckpt, static_assets.pt); --no_renderer if they are already on the local machine
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ORDER = ["PXB184", "RLW104", "TXB805", "GQS883"]
AUDIO_PER_FRAME = 1600
RENDERER_FILES = ("config.yml", "body_dec.ckpt", "static_assets.pt")  # what visualize/render_codes.BodyRenderer reads


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", default="PXB184", choices=ORDER[:3])
    p.add_argument("--num_chunks", type=int, default=3)
    p.add_argument("--chunks", type=int, nargs="*", default=None, help="test chunk indices (overrides the rule)")
    p.add_argument("--no_joint", action="store_true", help="leave out the TMR-joint row")
    p.add_argument("--no_renderer", action="store_true", help="the renderer is already on the local machine")
    p.add_argument("--runs", default="runs")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--s0", default="checkpoints/diffusion/c1_pose/model000340000.pt")
    p.add_argument("--samples_dir", default="cl_data/video_samples")
    p.add_argument("--manifest", default="cl_data/manifest.json")
    p.add_argument("--dataset_root", default="dataset")
    p.add_argument("--renderer_root", default="checkpoints/ca_body/data")
    p.add_argument("--out", default="cl_data/qual_pack")
    return p.parse_args()


def run_ckpt(a, method, k):
    return a.s0 if k == 0 else os.path.join(a.runs, method, f"s{k}_{ORDER[k]}", f"model{a.steps:09d}.pt")


def rows_for(a):
    """(key, label, checkpoint, sample folder) of every generated row."""
    j = ORDER.index(a.subject)
    rows = [("start", "Session-0 model" if j == 0 else f"Fine-tuning after Session {j}",
             run_ckpt(a, "vanilla", j), "s0_s0" if j == 0 else f"vanilla_s{j}")]
    rows += [(f"ft_s{k}", f"Fine-tuning after Session {k}", run_ckpt(a, "vanilla", k), f"vanilla_s{k}")
             for k in range(j + 1, 4)]
    rows.append(("tmr_s3", "TMR after Session 3", run_ckpt(a, "tmr", 3), "tmr_s3"))
    if not a.no_joint:
        if os.path.exists(run_ckpt(a, "tmr_joint", 3)):
            rows.append(("tmr_joint_s3", "TMR-joint after Session 3", run_ckpt(a, "tmr_joint", 3), "tmr_joint_s3"))
        else:
            print(f"no TMR-joint row: {run_ckpt(a, 'tmr_joint', 3)} not found")
    return rows


def sample_path(a, ckpt, tag):
    name = os.path.splitext(os.path.basename(ckpt))[0]
    return os.path.join(a.samples_dir, tag, f"{name}_{a.subject}_test_audio_only_seed0.npz")


def generate(a, rows):
    for key, label, ckpt, tag in rows:
        path = sample_path(a, ckpt, tag)
        if os.path.exists(path):
            print(f"samples {label}: {path} (exists)")
            continue
        if not os.path.exists(ckpt):
            sys.exit(f"missing checkpoint for {label}: {ckpt}")
        print(f"\n== generating {label}: {ckpt}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "cl.evaluate", "--ckpt", ckpt, "--subjects", a.subject, "--modes", "audio_only",
             "--save_samples", "--overwrite", "--manifest", a.manifest, "--out_dir", os.path.dirname(path)],
            check=True,
        )


def main():
    a = parse_args()
    rows = rows_for(a)
    print("rows: Real, " + ", ".join(r[1] for r in rows))
    generate(a, rows)

    import torchaudio

    from cl.evaluate import frechet

    with open(a.manifest) as f:
        manifest = json.load(f)
    window = manifest["window"]
    test = manifest["subjects"][a.subject]["test_chunks"]
    data = {r[0]: np.load(sample_path(a, r[2], r[3])) for r in rows}
    gt = data[rows[0][0]]["gt"]
    for key, d in data.items():
        assert d["pred"].shape == gt.shape and np.allclose(d["gt"], gt, atol=1e-4), f"{key}: different test chunks"

    # choose by a rule that reads only the real motion
    energy = np.mean(np.diff(gt, axis=1) ** 2, axis=(1, 2))
    chosen = a.chunks if a.chunks is not None else np.argsort(-energy)[: a.num_chunks].tolist()
    print(f"chunks {chosen} (rule: {'given' if a.chunks is not None else 'largest real motion energy'})")

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(os.path.join(a.out, "audio"))

    with open(os.path.join(a.out, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk", "scene", "start", "real_motion_energy", "chosen"]
                   + [f"{r[0]}_{m}" for r in rows for m in ("mse", "fd_g")])
        for c in range(len(gt)):
            cells = []
            for r in rows:
                p = data[r[0]]["pred"][c]
                cells += [f"{np.mean((p - gt[c]) ** 2):.5f}", f"{frechet(gt[c], p):.4f}"]
            w.writerow([c, test[c][0], test[c][1], f"{energy[c]:.6f}", int(c in chosen)] + cells)

    bodies = np.stack([np.stack([gt[c]] + [data[r[0]]["pred"][c] for r in rows]) for c in chosen]).astype(np.float32)
    faces = []
    root = os.path.join(a.dataset_root, a.subject)
    for c in chosen:
        scene, start = test[c][0], int(test[c][1])
        faces.append(np.asarray(np.load(os.path.join(root, scene + "_face_expression.npy"), mmap_mode="r")
                                [start : start + window], dtype=np.float32))
        audio, sr = torchaudio.load(os.path.join(root, scene + "_audio.wav"),
                                    frame_offset=start * AUDIO_PER_FRAME, num_frames=window * AUDIO_PER_FRAME)
        torchaudio.save(os.path.join(a.out, "audio", f"c{c:02d}.wav"), audio, sr, encoding="PCM_S", bits_per_sample=16)
    np.savez_compressed(
        os.path.join(a.out, "rows.npz"), bodies=bodies, faces=np.stack(faces), chunks=np.array(chosen),
        labels=np.array(["Real"] + [r[1] for r in rows]), keys=np.array(["real"] + [r[0] for r in rows]),
        subject=a.subject,
    )
    with open(os.path.join(a.out, "README.txt"), "w") as f:
        f.write(f"participant {a.subject}; test audio; audio-only generation, eval seed 0 (the noise of Tab. a2p)\n")
        f.write("rows: Real, " + ", ".join(f"{r[1]} ({r[2]})" for r in rows) + "\n")
        f.write(f"chunks {chosen}: " + ", ".join(f"c{c} = {test[c][0]} from frame {test[c][1]}" for c in chosen) + "\n")
        f.write("chosen by the largest real motion energy (no model output looked at)\n" if a.chunks is None
                else "chunks given on the command line\n")
        f.write("faces: the real face codes in every row (the body model generates only the body)\n")

    if not a.no_renderer:
        src = os.path.join(a.renderer_root, a.subject)
        dst = os.path.join(a.out, "ca_body", a.subject)
        os.makedirs(dst)
        for name in RENDERER_FILES:
            shutil.copy2(os.path.join(src, name), dst)
            print(f"renderer {name}: {os.path.getsize(os.path.join(src, name)) / 2**20:.0f} MB")

    tar = a.out + ".tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(a.out, arcname=os.path.basename(a.out))
    print(f"\nwritten {tar} ({os.path.getsize(tar) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
