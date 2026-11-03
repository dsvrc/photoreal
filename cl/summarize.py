"""
Forgetting table of the continual body-model runs, with the paper's metrics (lower is better):

    m[i][t] = metric on subject i after session t   (session 0 = the released PXB184 model)
    F_i     = mean over t > i of (m[i][t] - m[i][i]),   F_avg = mean of F_i over the old subjects
    A_avg   = mean over i of m[i][i]

    python -m cl.summarize --methods vanilla er star ocs gcr mir csrel tmr tmr_joint

Reads the JSON files cl.evaluate writes next to each checkpoint. A method that has not finished
every session is scored over the sessions it has finished ("through" column); --matrix also
prints each method's full m[i][t].
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--s0", default="checkpoints/diffusion/c1_pose/model000340000.pt")
    p.add_argument("--manifest", default="cl_data/manifest.json")
    p.add_argument("--split", default="test")
    p.add_argument("--modes", nargs="+", default=["audio_only", "gt_guide"])
    p.add_argument("--metrics", nargs="+", default=["fd_g", "fd_k"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--matrix", action="store_true")
    return p.parse_args()


def checkpoint(a, order, method, t):
    if t == 0:
        return a.s0
    return os.path.join(a.runs, method, f"s{t}_{order[t]}", f"model{a.steps:09d}.pt")


def read(a, ckpt, subject, mode, metric):
    name = os.path.splitext(os.path.basename(ckpt))[0]
    path = os.path.join(os.path.dirname(ckpt), "eval", f"{name}_{subject}_{a.split}_{mode}_seed{a.seed}.json")
    if not os.path.exists(path):
        return np.nan
    with open(path) as f:
        return json.load(f)[metric]


def forgetting(m):
    """F_avg, A_avg and the last session T such that every m[i][t], i <= t <= T, exists."""
    T = -1
    for t in range(len(m)):
        if np.isnan(m[: t + 1, t]).any():
            break
        T = t
    if T < 0:
        return np.nan, np.nan, T
    A = float(np.mean([m[i, i] for i in range(T + 1)]))
    F = float(np.mean([np.mean(m[i, i + 1 : T + 1] - m[i, i]) for i in range(T)])) if T > 0 else np.nan
    return F, A, T


def main():
    a = parse_args()
    with open(a.manifest) as f:
        order = json.load(f)["order"]
    K = len(order)
    fmt = lambda v: f"{v:8.4f}" if np.isfinite(v) else f"{'-':>8s}"
    for mode in a.modes:
        for metric in a.metrics:
            print(f"\n== {metric} | {mode} | {a.split}, seed {a.seed} | subject columns: after the last finished session")
            print(f"{'method':10s} " + " ".join(f"{s:>8s}" for s in order) + f" {'F_avg':>8s} {'A_avg':>8s}  through")
            matrices = {}
            for method in a.methods:
                m = np.full((K, K), np.nan)
                for t in range(K):
                    ckpt = checkpoint(a, order, method, t)
                    for i in range(t + 1):
                        m[i, t] = read(a, ckpt, order[i], mode, metric)
                F, A, T = forgetting(m)
                last = m[:, T] if T >= 0 else np.full(K, np.nan)
                print(f"{method:10s} " + " ".join(fmt(v) for v in last) + f" {fmt(F)} {fmt(A)}  s{max(T, 0)}")
                matrices[method] = m
            if a.matrix:
                for method, m in matrices.items():
                    print(f"\n{method}: rows = subject, columns = after session 0..{K - 1}")
                    for i, s in enumerate(order):
                        print(f"  {s:8s} " + " ".join(fmt(v) for v in m[i]))


if __name__ == "__main__":
    main()
