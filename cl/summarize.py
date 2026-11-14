"""
Results tables of the continual body-model runs (lower is better), laid out like DiscoForcing's
scripts/cl/screen_matrix_table.py: one row group per session t, in it one sub-row per subject seen
up to t, one column per baseline. After S0 is the released PXB184 model every baseline starts from.

    m[i][t] = metric on subject i after session t
    F_avg   = mean over the old subjects i < t of the mean over t' in (i, t] of (m[i][t'] - m[i][i])
    A_avg   = mean over i <= t of m[i][i]

From S1 on, each group ends with its F_avg (forgetting) and A_avg, both through that session.
Bold is the best replay baseline of the row; vanilla (no replay) is the reference and never bold.
"-" = not run yet. One table per mode and metric, printed and written to --out as Markdown.

    python -m cl.summarize --evaluate    # first test what is missing (GPU), then the tables
    python -m cl.summarize               # tables from the existing results only

--evaluate runs cl.evaluate on every finished checkpoint (and the released model) for the subjects
seen up to its session; results that already exist are reused, not recomputed.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

BASELINES = ["vanilla", "er", "star", "ocs", "gcr", "mir", "csrel", "tmr", "tmr_joint"]
NAMES = {"fd_g": "FD_g", "fd_k": "FD_k"}
DECIMALS = {"fd_g": 4, "fd_k": 6}  # as cl.evaluate prints them


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=BASELINES, help="table columns (run folders under --runs)")
    p.add_argument("--runs", default="runs")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--s0", default="checkpoints/diffusion/c1_pose/model000340000.pt")
    p.add_argument("--manifest", default="cl_data/manifest.json")
    p.add_argument("--split", default="test")
    p.add_argument("--modes", nargs="+", default=["audio_only", "gt_guide"])
    p.add_argument("--metrics", nargs="+", default=["fd_g", "fd_k"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evaluate", action="store_true", help="first test the finished checkpoints that miss results")
    p.add_argument("--out", default=None, help="Markdown file of the tables (default: <runs>/table_<split>_seed<seed>.md)")
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


def evaluate_missing(a, order):
    """cl.evaluate on every finished checkpoint after each session, for the subjects seen so far."""
    for t in range(len(order)):
        ckpts = [c for c in dict.fromkeys(checkpoint(a, order, m, t) for m in a.methods) if os.path.exists(c)]
        seen = order[: t + 1]
        if not ckpts:
            print(f"[evaluate] after S{t}: no finished checkpoint yet", flush=True)
            continue
        print(f"[evaluate] after S{t}: {len(ckpts)} checkpoint(s) on {' '.join(seen)}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "cl.evaluate", "--ckpt", *ckpts, "--subjects", *seen, "--modes", *a.modes,
             "--split", a.split, "--seed", str(a.seed), "--manifest", a.manifest],
            check=True,
        )


def through(m, t):
    """F_avg and A_avg through session t; nan unless every m[i][t'], i <= t' <= t, exists."""
    if np.isnan([m[i, s] for s in range(t + 1) for i in range(s + 1)]).any():
        return np.nan, np.nan
    A = float(np.mean([m[i, i] for i in range(t + 1)]))
    F = float(np.mean([np.mean(m[i, i + 1 : t + 1] - m[i, i]) for i in range(t)]))
    return F, A


def table(a, order, mode, metric):
    K, digits = len(order), DECIMALS.get(metric, 4)
    ms = {}
    for method in a.methods:
        m = np.full((K, K), np.nan)
        for t in range(K):
            ckpt = checkpoint(a, order, method, t)
            for i in range(t + 1):
                m[i, t] = read(a, ckpt, order[i], mode, metric)
        ms[method] = m
    rows = []
    for t in range(K):
        sub = [(f"S{i}: {order[i]}", {k: ms[k][i, t] for k in a.methods}) for i in range(t + 1)]
        if t > 0:
            fa = {k: through(ms[k], t) for k in a.methods}
            sub += [("F_avg", {k: v[0] for k, v in fa.items()}), ("A_avg", {k: v[1] for k, v in fa.items()})]
        for n, (label, values) in enumerate(sub):
            ranked = [v for k, v in values.items() if k != "vanilla" and np.isfinite(v)]
            best = min(ranked) if ranked and t > 0 else None  # after S0 every column is the same model
            cells = []
            for k in a.methods:
                v = values[k]
                s = f"{v:.{digits}f}" if np.isfinite(v) else "-"
                cells.append(f"**{s}**" if best is not None and k != "vanilla" and v == best else s)
            rows.append([f"**After S{t}**" if n == 0 else "", label, *cells])
    head = ["Model after", "Subject", *a.methods]
    widths = [max(len(r[c]) for r in [head, *rows]) for c in range(len(head))]
    line = lambda r: "| " + " | ".join(x.ljust(w) if c < 2 else x.rjust(w) for c, (x, w) in enumerate(zip(r, widths))) + " |"
    rule = "|" + "|".join("-" * (w + 2) if c < 2 else "-" * (w + 1) + ":" for c, w in enumerate(widths)) + "|"
    return [line(head), rule, *[line(r) for r in rows]]


def main():
    a = parse_args()
    with open(a.manifest) as f:
        order = json.load(f)["order"]
    if a.evaluate:
        evaluate_missing(a, order)
    out = [
        f"Continual body-model results: {a.split} split, eval seed {a.seed}. Lower is better. Bold: best replay"
        " baseline of the row (vanilla is the reference and never bold). -: not run yet. F_avg (forgetting) and"
        " A_avg: through that session.",
    ]
    for mode in a.modes:
        for metric in a.metrics:
            out += ["", f"## {NAMES.get(metric, metric)}, {mode}", "", *table(a, order, mode, metric)]
    text = "\n".join(out) + "\n"
    print(text)
    path = a.out or os.path.join(a.runs, f"table_{a.split}_seed{a.seed}.md")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"written to {path}")


if __name__ == "__main__":
    main()
