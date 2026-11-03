"""
Build the replay buffer of one finished session, using the model at the end of that session.
Candidates are the subject's 600-frame windows tiling its training scenes (cl/clips.py); the buffer
keeps round(ratio x candidates) of them.

    python -m cl.build_buffer --method tmr --subject PXB184 --session 0 \
        --ckpt checkpoints/diffusion/c1_pose/model000340000.pt --out cl_data/buffers/tmr/s0_PXB184.json

    random     uniform sample (ER, and the buffer of Random*/STAR)
    ocs        clip features: cosine to the minibatch centroid minus redundancy
    gcr        clip features: regularised non-negative OMP towards the session's feature sum
    mir        clips whose loss rises most after one virtual training step on the session
    csrel      largest reducible loss in greedy rounds; --prev_ckpt is the model before the session
               (session 0 has none: a fresh initialisation)
    tmr        whitened temporal modes of the audio-condition trajectory, simplex OMP towards the
               mean atom, rank by BIC with n = N * D (paper version)
    tmr_joint  TMR on the audio-condition trajectory next to the pose target, BIC with the effective
               sample count, pursuit run to the budget

Features for ocs/gcr/tmr/tmr_joint are cached in --feature_dir, one file per checkpoint and subject.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cl import selectors
from cl.clips import replay_budget, session_clips
from cl.common import build_model, load_ckpt_args
from cl.replay import write_buffer
from cl.scoring import ClipBank, clip_features, select_csrel, select_mir
from utils.misc import fixseed
from utils.model_util import create_model_and_diffusion

FEATURE_METHODS = ("ocs", "gcr", "tmr", "tmr_joint")
METHODS = ("random",) + FEATURE_METHODS + ("mir", "csrel")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--ckpt", required=True, help="model at the end of this session")
    p.add_argument("--subject", required=True)
    p.add_argument("--session", type=int, required=True, help="index of this subject's session (seeds the selection)")
    p.add_argument("--out", required=True)
    p.add_argument("--prev_ckpt", default=None, help="CSReL: model before this session (default: fresh init)")
    p.add_argument("--ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--manifest", default="cl_data/manifest.json")
    p.add_argument("--dataset_root", default="dataset")
    p.add_argument("--feature_dir", default="cl_data/features")
    p.add_argument("--device", default="cuda")
    p.add_argument("--load_batch", type=int, default=8, help="clips per batch while encoding audio")
    p.add_argument("--num_workers", type=int, default=None, help="default: min(4, CPUs of this job)")
    # OCS / GCR
    p.add_argument("--ocs_batch", type=int, default=128)
    p.add_argument("--omp_lam", type=float, default=0.5)
    p.add_argument("--omp_tol", type=float, default=1e-4)
    # TMR
    p.add_argument("--tmr_steps", type=int, default=100, help="time steps of the pooled trajectories (5 Hz)")
    p.add_argument("--tmr_stop", default="fw", choices=["fw", "budget"], help="paper version only")
    p.add_argument("--mix_eps", type=float, default=0.02, help="tmr_joint: clips closer than this share a condition")
    # MIR / CSReL
    p.add_argument("--loss_seeds", type=int, default=4, help="fixed noise seeds averaged per clip loss")
    p.add_argument("--score_batch", type=int, default=25)
    p.add_argument("--mir_cand", type=int, default=50)
    p.add_argument("--mir_evals", type=int, default=3)
    p.add_argument("--mir_vbatch", type=int, default=16)
    p.add_argument("--mir_update", default="adam", choices=["adam", "sgd"])
    p.add_argument("--csrel_rounds", type=int, default=5)
    p.add_argument("--csrel_steps", type=int, default=100)
    p.add_argument("--csrel_lr", type=float, default=1e-4)
    p.add_argument("--csrel_batch", type=int, default=None, help="default: the checkpoint's training batch")
    return p.parse_args()


def load(ckpt, device):
    args = load_ckpt_args(ckpt)
    args.timestep_respacing = ""  # per-clip losses use the training chain
    model, diffusion = build_model(args, ckpt, split_type="train")
    model.to(device)
    diffusion.compute_vb = False
    return args, model, diffusion


def feature_path(feature_dir, ckpt, subject, steps):
    slug = os.path.splitext(os.path.normpath(ckpt))[0].replace("\\", "/").strip("./").replace("/", "_")
    return os.path.join(feature_dir, f"{slug}__{subject}_T{steps}.npz")


def features(a, clips, entries, window, workers):
    path = feature_path(a.feature_dir, a.ckpt, a.subject, a.tmr_steps)
    if os.path.exists(path):
        d = np.load(path)
        if str(d["ckpt"]) == a.ckpt and d["starts"].tolist() == [c[1] for c in clips]:
            print(f"features: {path} (cached)")
            return {k: d[k] for k in ("phi", "pose", "pose_stats")}
    print(f"features of {len(clips)} clips at {a.ckpt}")
    _, model, _ = load(a.ckpt, a.device)
    feats = clip_features(model, entries, a.dataset_root, window, a.device, a.tmr_steps, a.load_batch, workers)
    os.makedirs(a.feature_dir, exist_ok=True)
    np.savez(path, **feats, scenes=np.array([c[0] for c in clips]), starts=np.array([c[1] for c in clips]), ckpt=a.ckpt)
    print(f"features: wrote {path} | phi {feats['phi'].shape} pose {feats['pose'].shape}")
    return feats


def previous_state(a, args):
    if a.prev_ckpt:
        print(f"theta_S (before the session): {a.prev_ckpt}")
        return torch.load(a.prev_ckpt, map_location="cpu")
    print("theta_S (before the session): fresh initialisation")
    fixseed(a.seed * 1000 + a.session)
    fresh, _ = create_model_and_diffusion(args, split_type="train")
    return fresh.state_dict()


def main():
    a = parse_args()
    t0 = time.time()
    fixseed(a.seed * 1000 + a.session)
    # exact fp32 so that loss differences between two models are not rounding noise
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    with open(a.manifest) as f:
        manifest = json.load(f)
    window = manifest["window"]
    clips = session_clips(manifest, a.subject)
    n, budget = len(clips), replay_budget(len(clips), a.ratio)
    rng = np.random.default_rng(a.seed * 1000 + a.session)
    entries = [{"subject": a.subject, "scene": s, "start": st, "weight": 1.0} for s, st in clips]
    workers = a.num_workers
    if workers is None:
        workers = min(4, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
    print(f"[{a.method}] {a.subject} (session {a.session}): {n} candidate clips of {window} frames, keeping {budget}")

    info = {}
    if a.method == "random":
        idx, w = selectors.select_random(n, budget, rng)
    elif a.method in FEATURE_METHODS:
        feats = features(a, clips, entries, window, workers)
        if a.method in ("ocs", "gcr"):
            f = selectors.ocs_features(feats["phi"], feats["pose_stats"])
            if a.method == "ocs":
                idx, w = selectors.select_ocs(f, budget, rng, a.ocs_batch)
            else:
                idx, w = selectors.select_gcr(f, budget, rng, a.omp_lam, a.omp_tol)
        elif a.method == "tmr":
            idx, w, info = selectors.select_tmr(feats["phi"], budget, a.tmr_stop)
        else:
            idx, w, info = selectors.select_tmr_joint(feats["phi"], feats["pose"], budget, rng, a.mix_eps)
    else:
        args, model, diffusion = load(a.ckpt, a.device)
        print(f"loading {n} clips with their audio embeddings")
        bank = ClipBank(model, entries, a.dataset_root, window, a.device, a.load_batch, workers)
        seeds = tuple(range(a.loss_seeds))
        if a.method == "mir":
            idx, w, info = select_mir(
                model, diffusion, bank, budget, rng, args.lr, a.mir_update, a.mir_cand, a.mir_evals, a.mir_vbatch,
                seeds, a.score_batch, a.device,
            )
        else:
            idx, w, info = select_csrel(
                model, previous_state(a, args), diffusion, bank, budget, rng, a.csrel_rounds, a.csrel_steps,
                a.csrel_lr, 1.0, a.csrel_batch or args.batch_size, seeds, a.score_batch, a.device,
            )
            info["prev_ckpt"] = a.prev_ckpt

    write_buffer(
        a.out, a.subject, a.method, clips, idx, w,
        session=a.session, ckpt=a.ckpt, candidates=n, budget=budget, seed=a.seed, info=info,
    )
    w = np.asarray(w, dtype=float) * len(w) / np.sum(w)
    scenes = len({clips[i][0] for i in idx})
    print(
        f"wrote {a.out}: {len(idx)} clips from {scenes} scenes, weights {w.min():.3f}-{w.max():.3f}"
        f" | {json.dumps({k: v for k, v in info.items() if not isinstance(v, (list, dict))})}"
        f" | {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
