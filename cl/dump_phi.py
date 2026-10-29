"""
Dump the audio-encoder trajectories phi_i = cond_projection(wav2vec(a_i)) of one subject's
training windows at a given checkpoint: the input of TMR (and of any feature-based buffer).

    python -m cl.dump_phi --ckpt checkpoints/diffusion/c1_pose/model000340000.pt \
        --subject PXB184 --out cl_data/phi/s0_PXB184.npz

Output .npz: phi [N x T x D] (time average-pooled to --pool_to steps; D = 256),
scenes [N], starts [N], subject, ckpt.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from cl.common import build_model, load_ckpt_args, load_stats, load_window


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint at the end of this subject's session")
    p.add_argument("--subject", required=True)
    p.add_argument("--manifest", default="cl_data/manifest.json")
    p.add_argument("--dataset_root", default="dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--pool_to", type=int, default=100, help="time steps after pooling (0 = keep ~2000 at 100 Hz)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    with open(a.manifest) as f:
        manifest = json.load(f)
    window = manifest["window"]
    windows = manifest["subjects"][a.subject]["train_windows"]
    stats = load_stats(os.path.join(a.dataset_root, a.subject))

    args = load_ckpt_args(a.ckpt)
    model, _ = build_model(args, a.ckpt, split_type="train")
    model.to(a.device).eval()

    phis = []
    with torch.no_grad():
        for i in range(0, len(windows), a.batch):
            audio = []
            for scene, start in windows[i : i + a.batch]:
                _, raw = load_window(a.dataset_root, a.subject, scene, start, window)
                audio.append((raw - stats["audio_mean"]) / stats["audio_std"])
            audio = torch.tensor(np.stack(audio), dtype=torch.float32, device=a.device)
            phi = model.cond_projection(model.encode_audio(audio))  # B x ~2000 x 256
            if a.pool_to:
                phi = F.adaptive_avg_pool1d(phi.transpose(1, 2), a.pool_to).transpose(1, 2)
            phis.append(phi.float().cpu().numpy())
            print(f"\r{min(i + a.batch, len(windows))}/{len(windows)} windows", end="", flush=True)
    phi = np.concatenate(phis)
    print(f"\nphi {phi.shape}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(
        a.out,
        phi=phi,
        scenes=np.array([w[0] for w in windows]),
        starts=np.array([w[1] for w in windows]),
        subject=a.subject,
        ckpt=a.ckpt,
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
