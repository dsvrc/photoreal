"""
Select a replay buffer from a phi dump and write it in the format cl.train_session reads.

    python -m cl.select_buffer --phi cl_data/phi/s0_PXB184.npz --method er --frac 0.1 \
        --out cl_data/buffers/er/s0_PXB184.json

Only random replay (er) is built in. To use TMR or another selector, read `phi` [N x T x D],
`scenes` and `starts` from the .npz, choose indices with non-negative weights, and write the
same JSON (weights normalized to mean 1, as in the paper):
    {"subject": ..., "method": ..., "entries": [{"subject", "scene", "start", "weight"}, ...]}
"""

import argparse
import json
import os

import numpy as np


def write_buffer(path, subject, method, scenes, starts, idx, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights * len(weights) / weights.sum()  # mean 1
    entries = [
        {"subject": subject, "scene": str(scenes[i]), "start": int(starts[i]), "weight": float(w)}
        for i, w in zip(idx, weights)
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"subject": subject, "method": method, "entries": entries}, f, indent=1)
    print(f"wrote {path}: {len(entries)} windows of {subject}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phi", required=True)
    p.add_argument("--method", default="er", choices=["er"])
    p.add_argument("--frac", type=float, default=0.10, help="fraction of this subject's windows to keep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    d = np.load(a.phi)
    n = len(d["starts"])
    budget = max(1, int(round(a.frac * n)))
    idx = np.sort(np.random.default_rng(a.seed).choice(n, budget, replace=False))
    write_buffer(a.out, str(d["subject"]), a.method, d["scenes"], d["starts"], idx, np.ones(budget))


if __name__ == "__main__":
    main()
