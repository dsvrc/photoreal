"""
Collect, from the server's files, every Audio2Photoreal number the TMR supplementary still leaves red.
Reads only files (no GPU, no model), so it runs on a login node in a2p_env from the repo root:

    python scripts/cl/collect_suppl.py

Writes the report to stdout, cl_data/suppl_report.txt and cl_data/suppl_report.json.

What it fills (sections of sec/X_suppl.tex):
  [1] Tab. a2p_split          train scenes, train hours, candidates N / budget B, test chunks per participant
  [2] Sec. app:a2p, TMR       clips in every TMR buffer, fw_stop, M_hat_k (BIC with n = N*D, and with n = N)
  [3] Sec. app:a2p, Replay    whether any run was resumed after an interruption (tensorboard files, PBS logs)
  [4] FAQ (v), Replay         every method shares the seed, batch sizes and start checkpoint of each session
  [5] Sec. app:a2p, Generator released configuration (learning rate etc.) the sessions inherit
  [6] Tab. a2p                the FD_g in the paper against the eval files (catches a stale table)
  [7] cost                    training wall time per session from the tensorboard files
  [8] all buffers             clips kept by every method against its budget
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ORDER = ["PXB184", "RLW104", "TXB805", "GQS883"]
FPS = 30
# paper column -> run folder (Fine-tuning is "vanilla"; STAR replays the ER-style random buffer)
METHODS = ["vanilla", "tmr", "er", "star", "ocs", "gcr", "mir", "csrel"]
EXTRA_METHODS = ["tmr_joint"]  # reported if present, not in the paper
S0 = "checkpoints/diffusion/c1_pose/model000340000.pt"

# Tab. a2p of the current draft (FD_g, audio only), columns in METHODS order; row (subject, after session)
PAPER = {
    (0, 0): [3.764] * 8,
    (0, 1): [5.099, 4.821, 4.913, 4.883, 5.106, 4.769, 6.011, 4.732],
    (1, 1): [6.412, 6.283, 6.573, 6.340, 6.722, 6.507, 6.499, 6.442],
    (0, 2): [5.394, 4.348, 4.513, 4.427, 4.807, 4.630, 5.890, 4.478],
    (1, 2): [6.619, 5.281, 5.483, 5.641, 5.391, 6.531, 6.415, 5.375],
    (2, 2): [5.122, 4.967, 5.209, 5.137, 5.659, 5.282, 5.302, 5.148],
    (0, 3): [5.647, 3.986, 4.613, 4.406, 5.238, 4.292, 5.589, 4.529],
    (1, 3): [7.294, 4.751, 5.307, 5.044, 5.742, 5.820, 7.486, 5.226],
    (2, 3): [6.098, 5.738, 6.120, 5.884, 6.312, 5.606, 8.244, 5.919],
    (3, 3): [3.313, 2.972, 3.472, 3.196, 3.690, 3.119, 3.461, 3.282],
}
PAPER_SUMMARY = {
    "F_avg": [1.045, 0.042, 0.216, 0.186, 0.261, 0.264, 1.820, 0.149],
    "A_avg": [4.949, 4.741, 5.085, 4.891, 5.357, 4.969, 5.088, 4.957],
    "m_fin": [6.346, 4.825, 5.347, 5.111, 5.764, 5.239, 7.106, 5.225],
}

OUT = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    OUT.append(line)


def head(title):
    say("")
    say("=" * 100)
    say(title)
    say("=" * 100)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="cl_data/manifest.json")
    p.add_argument("--runs", default="runs")
    p.add_argument("--buffers", default="cl_data/buffers")
    p.add_argument("--feature_dir", default="cl_data/features")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--s0", default=S0)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--mode", default="audio_only", help="evaluation mode of Tab. a2p")
    p.add_argument("--logs", nargs="*", default=["*.o[0-9]*", "logs/**/*.o[0-9]*", "logs/**/*.log", "*.log"],
                   help="text logs (PBS -j oe output) scanned for 'resuming from' lines")
    p.add_argument("--out", default="cl_data/suppl_report")
    return p.parse_args()


def run_dir(a, method, k):
    return os.path.join(a.runs, method, f"s{k}_{ORDER[k]}")


def ckpt_after(a, method, k):
    return a.s0 if k == 0 else os.path.join(run_dir(a, method, k), f"model{a.steps:09d}.pt")


def methods_present(a):
    return METHODS + [m for m in EXTRA_METHODS if os.path.isdir(os.path.join(a.runs, m))]


# ------------------------------------------------------------------------------------------------ [1]
def split_table(a, report):
    head("[1] Tab. a2p_split: sessions and splits (from the manifest)")
    man = load_json(a.manifest)
    if man is None:
        say(f"MISSING {a.manifest}: run python scripts/cl/prepare_data.py --data_root dataset --out cl_data")
        return
    from cl.clips import replay_budget, tile_starts

    window = man.get("window", 600)
    rows, latex = {}, []
    say(f"window {window} frames, order {man.get('order')}")
    say(f"{'sess':>4} {'subject':8} {'train':>5} {'val':>4} {'test':>4} {'h train':>8} {'h val':>6} {'h test':>6}"
        f" {'N':>5} {'B':>4} {'test chunks':>11} {'val chunks':>10}  problems")
    for k, s in enumerate(ORDER):
        e = man["subjects"].get(s)
        if e is None:
            say(f"{k:>4} {s:8} NOT IN MANIFEST")
            continue
        sc, fr = e["scenes"], e["frames"]
        hours = {sp: sum(fr[x] for x in sc[sp]) / FPS / 3600 for sp in ("train", "val", "test")}
        n = sum(len(tile_starts(fr[x], window)) for x in sc["train"])
        n_manifest = len(e.get("train_clips", []))
        b = replay_budget(n, 0.10)
        r = {
            "session": k, "train_scenes": len(sc["train"]), "val_scenes": len(sc["val"]), "test_scenes": len(sc["test"]),
            "train_hours": hours["train"], "val_hours": hours["val"], "test_hours": hours["test"],
            "candidates": n, "candidates_in_manifest": n_manifest, "budget": b,
            "test_chunks": len(e["test_chunks"]), "val_chunks": len(e["val_chunks"]),
            "train_frames": sum(fr[x] for x in sc["train"]), "problems": e.get("problems", []),
        }
        rows[s] = r
        say(f"{k:>4} {s:8} {r['train_scenes']:>5} {r['val_scenes']:>4} {r['test_scenes']:>4} {hours['train']:8.2f}"
            f" {hours['val']:6.2f} {hours['test']:6.2f} {n:>5} {b:>4} {r['test_chunks']:>11} {r['val_chunks']:>10}"
            f"  {len(r['problems'])}")
        if n != n_manifest:
            say(f"     NOTE: manifest lists {n_manifest} train clips, cl.clips tiling gives {n}")
        latex.append(f"{k} & {s} & {r['train_scenes']} & {r['val_scenes']} & {r['test_scenes']} & {hours['train']:.2f}"
                     f" & {n} / {b} & {r['test_chunks']}\\\\")
    tot_h = sum(r["train_hours"] for r in rows.values())
    say(f"total train hours {tot_h:.2f}, total candidates {sum(r['candidates'] for r in rows.values())}")
    say("")
    say("LaTeX rows for Tab. a2p_split:")
    for l in latex:
        say("  " + l)
    report["split"] = rows


# ------------------------------------------------------------------------------------------------ [2]
def feature_path(feature_dir, ckpt, subject, steps):  # same as cl.build_buffer.feature_path
    slug = os.path.splitext(os.path.normpath(ckpt))[0].replace("\\", "/").strip("./").replace("/", "_")
    return os.path.join(feature_dir, f"{slug}__{subject}_T{steps}.npz")


def tmr_buffers(a, report):
    head("[2] Sec. app:a2p, TMR: buffer sizes, fw_stop and the modal rank M_hat_k (from the buffer files)")
    rows = []
    for j in range(3):
        s = ORDER[j]
        path = os.path.join(a.buffers, "tmr", f"s{j}_{s}.json")
        buf = load_json(path)
        if buf is None:
            say(f"s{j} {s}: MISSING {path}")
            continue
        info = buf.get("info", {})
        w = np.array([e.get("weight", 1.0) for e in buf["entries"]])
        r = {
            "session": j, "subject": s, "clips": len(buf["entries"]), "budget": buf.get("budget"),
            "candidates": buf.get("candidates"), "scenes": len({e["scene"] for e in buf["entries"]}),
            "fw_stop": info.get("fw_stop"), "M": info.get("M"), "T": info.get("T"), "n_eff": info.get("n_eff"),
            "energy_kept": info.get("energy_kept"), "rel_err": info.get("rel_err"),
            "err_vs_random": info.get("err_vs_random"), "ckpt": buf.get("ckpt"),
            "weight_min": float(w.min()) if len(w) else None, "weight_max": float(w.max()) if len(w) else None,
        }
        # M_hat with n = N*D (used) and n = N, recomputed from the cached trajectories when they exist
        T = r["T"] or 100
        fp = feature_path(a.feature_dir, buf.get("ckpt", ""), s, T)
        if os.path.exists(fp):
            try:
                from cl import selectors

                phi = np.load(fp)["phi"].astype(np.float64)
                N, T_, D = phi.shape
                lam, _ = selectors.eig_desc(selectors.temporal_operator(phi))
                r["M_nND"] = int(selectors.bic_rank(lam, N * D))
                r["M_nN"] = int(selectors.bic_rank(lam, N))
                r["phi_shape"] = [N, T_, D]
                r["top_eig_share"] = (lam[:10] / lam.sum()).round(4).tolist()
            except Exception as ex:  # noqa: BLE001
                r["recompute_error"] = repr(ex)
        else:
            r["features"] = f"not cached: {fp}"
        rows.append(r)
        short = "SHORT OF BUDGET" if r["budget"] and r["clips"] < r["budget"] else "full"
        say(f"s{j} {s}: {r['clips']} / {r['budget']} clips ({short}) from {r['scenes']} scenes of {r['candidates']}"
            f" candidates | fw_stop {r['fw_stop']} | M_hat {r['M']} (T {r['T']}, n_eff {r['n_eff']})"
            f" | energy kept {r['energy_kept']} | rel_err {r['rel_err']} | err/random {r['err_vs_random']}"
            f" | weights {r['weight_min']}-{r['weight_max']}")
        if "M_nND" in r:
            say(f"     recomputed from {fp}: phi {r['phi_shape']}, M_hat(n=N*D) {r['M_nND']}, M_hat(n=N) {r['M_nN']},"
                f" top-10 eigenvalue shares {r['top_eig_share']}")
        elif "features" in r:
            say(f"     {r['features']} (M_hat(n=N) not recomputed)")
    if rows:
        short = [r for r in rows if r["budget"] and r["clips"] < r["budget"]]
        say("")
        say(f"buffers short of the budget: {[(r['subject'], r['clips'], r['budget']) for r in short] or 'none'}")
        say(f"M_hat_k per session: {[(r['subject'], r['M']) for r in rows]}")
    report["tmr_buffers"] = rows


# ------------------------------------------------------------------------------------------------ [3] + [7]
def event_scan(path):
    """Steps and wall times of loss/new in one tensorboard file, plus the file's first/last wall time."""
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    steps, walls, all_walls = [], [], []
    for ev in EventFileLoader(path).Load():
        all_walls.append(ev.wall_time)
        if ev.HasField("summary"):
            for v in ev.summary.value:
                if v.tag == "loss/new":
                    steps.append(int(ev.step))
                    walls.append(ev.wall_time)
    return {"file": os.path.basename(path), "first_step": steps[0] if steps else None,
            "last_step": steps[-1] if steps else None, "logged": len(steps),
            "start": min(all_walls, default=None), "end": max(all_walls, default=None),
            "hours": (max(all_walls) - min(all_walls)) / 3600 if all_walls else 0.0}


def resume_check(a, report):
    head("[3] Sec. app:a2p, Replay: was any run resumed after an interruption? + [7] training wall time")
    say("A resumed run continues from its latest checkpoint with a new batch stream (different batch order).")
    say("Evidence: one tensorboard events file per launch of cl.train_session; a file whose first logged step")
    say("is above 100 started from a checkpoint (resumed). A second file that starts at step 100 again is a")
    say("restart from scratch with the same seed, i.e. the same batch order as an uninterrupted run.")
    try:
        import tensorboard  # noqa: F401
        have_tb = True
    except ImportError:
        have_tb = False
        say("WARNING: tensorboard not importable; only counting files and checkpoints")
    rows = []
    for method in methods_present(a):
        for k in (1, 2, 3):
            d = run_dir(a, method, k)
            if not os.path.isdir(d):
                rows.append({"method": method, "session": k, "status": "missing"})
                say(f"{method:9} s{k} {ORDER[k]}: MISSING {d}")
                continue
            evs = sorted(glob.glob(os.path.join(d, "events.out.tfevents.*")))
            ckpts = sorted(glob.glob(os.path.join(d, "model*.pt")))
            files = []
            if have_tb:
                for p in evs:
                    try:
                        files.append(event_scan(p))
                    except Exception as ex:  # noqa: BLE001
                        files.append({"file": os.path.basename(p), "error": repr(ex)})
            files.sort(key=lambda f: f.get("start") or 0)
            resumed = [f for f in files if f.get("first_step") is not None and f["first_step"] > 100]
            restarts = [f for f in files[1:] if f.get("first_step") == 100]
            empty = [f for f in files if f.get("logged") == 0]
            hours = sum(f.get("hours", 0.0) for f in files)
            useful = None
            if files and files[-1].get("last_step"):
                useful = files[-1]["last_step"]
            finished = os.path.exists(os.path.join(d, f"model{a.steps:09d}.pt"))
            r = {"method": method, "session": k, "launches": len(evs), "resumed_from_steps":
                 [f["first_step"] - 100 for f in resumed], "restarts_from_scratch": len(restarts),
                 "empty_launches": len(empty), "train_hours": hours, "checkpoints": len(ckpts),
                 "finished": finished, "files": files}
            rows.append(r)
            flag = "RESUMED" if resumed else ("restarted from 0" if restarts else "uninterrupted")
            say(f"{method:9} s{k} {ORDER[k]}: {len(evs)} launch(es), {flag}"
                f"{' at steps ' + str(r['resumed_from_steps']) if resumed else ''}"
                f" | {len(ckpts)} ckpts, finished {finished} | wall {hours:.2f} h | last logged step {useful}")
    report["runs"] = rows

    # PBS / text logs: "resuming from <ckpt> at step N"
    say("")
    say("text logs ('resuming from' lines):")
    hits = []
    seen = set()
    for pat in a.logs:
        for p in glob.glob(pat, recursive=True):
            if p in seen or not os.path.isfile(p) or os.path.getsize(p) > 200 * 2**20:
                continue
            seen.add(p)
            try:
                with open(p, errors="replace") as f:
                    for line in f:
                        m = re.search(r"resuming from (\S+) at step (\d+)", line)
                        if m:
                            hits.append({"log": p, "ckpt": m.group(1), "step": int(m.group(2))})
            except OSError:
                pass
    say(f"  scanned {len(seen)} log file(s)")
    for h in hits:
        say(f"  {h['log']}: resumed {h['ckpt']} at step {h['step']}")
    if not hits:
        say("  none found")
    report["log_resumes"] = hits

    say("")
    resumed_runs = [(r["method"], r["session"], r["resumed_from_steps"]) for r in rows if r.get("resumed_from_steps")]
    say(f"SUMMARY resumed runs (tensorboard): {resumed_runs or 'none'}")
    say("SUMMARY training wall time per session (h, all launches):")
    for method in methods_present(a):
        hs = [next((r["train_hours"] for r in rows if r["method"] == method and r["session"] == k
                    and "train_hours" in r), float("nan")) for k in (1, 2, 3)]
        say(f"  {method:9} " + "  ".join(f"s{k} {h:6.2f}" for k, h in zip((1, 2, 3), hs)) + f"  total {np.nansum(hs):6.2f}")


# ------------------------------------------------------------------------------------------------ [4] + [5]
def config_check(a, report):
    head("[4] FAQ (v) + Replay: seed, batches and start checkpoint per session (args.json of every run)")
    keys = ("seed", "replay_batch", "replay_lambda", "passes", "steps")
    rows = []
    for k in (1, 2, 3):
        say(f"-- session {k} ({ORDER[k]})")
        for method in methods_present(a):
            args = load_json(os.path.join(run_dir(a, method, k), "args.json"))
            if args is None:
                say(f"  {method:9} no args.json")
                continue
            cl = args.get("cl", {})
            r = {"method": method, "session": k, "batch_size": args.get("batch_size"), "lr": args.get("lr"),
                 "weight_decay": args.get("weight_decay"), **{x: cl.get(x) for x in keys},
                 "init_ckpt": cl.get("init_ckpt"), "replay": cl.get("replay"), "star": cl.get("star")}
            rows.append(r)
            expect_init = ckpt_after(a, method, k - 1)
            ok_init = os.path.normpath(r["init_ckpt"] or "") == os.path.normpath(expect_init)
            say(f"  {method:9} seed {r['seed']} | batch {r['batch_size']}+{r['replay_batch']} replay"
                f" (lambda {r['replay_lambda']}, passes {r['passes']}) | lr {r['lr']} wd {r['weight_decay']}"
                f" | steps {r['steps']} | init {'OK' if ok_init else 'UNEXPECTED ' + str(r['init_ckpt'])}"
                f" | {len(r['replay'] or [])} buffer(s){' | STAR ' + json.dumps(r['star']) if r['star'] else ''}")
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    passes = sorted({r["passes"] for r in rows if r["passes"] is not None})
    say("")
    say(f"SUMMARY seeds used by all runs: {seeds} | passes: {passes} (1 and 2 give the same loss and gradient)")
    report["run_args"] = rows

    head("[5] Sec. app:a2p, Generator: released configuration inherited by every session")
    rel = load_json(os.path.join(os.path.dirname(a.s0), "args.json"))
    if rel is None:
        say(f"MISSING args.json next to {a.s0}")
    else:
        for x in ("lr", "batch_size", "weight_decay", "max_seq_length", "num_steps", "cond_mask_prob", "latent_dim",
                  "layers", "heads", "lambda_vel", "lambda_rcxyz", "diffusion_steps", "noise_schedule", "data_format",
                  "ff_size", "dropout", "lr_anneal_steps", "keyframe_step"):
            if x in rel:
                say(f"  {x}: {rel[x]}")
        report["released_args"] = rel


# ------------------------------------------------------------------------------------------------ [6]
def results_check(a, report):
    head(f"[6] Tab. a2p: FD_g ({a.mode}, test, eval seed {a.eval_seed}) from the eval files vs the draft")
    ms = {}
    for method in METHODS:
        m = np.full((4, 4), np.nan)
        for t in range(4):
            ck = ckpt_after(a, method, t)
            name = os.path.splitext(os.path.basename(ck))[0]
            for i in range(t + 1):
                p = os.path.join(os.path.dirname(ck), "eval", f"{name}_{ORDER[i]}_test_{a.mode}_seed{a.eval_seed}.json")
                row = load_json(p)
                if row is not None:
                    m[i, t] = row["fd_g"]
        ms[method] = m
    say(f"{'row':14} " + " ".join(f"{x:>8}" for x in METHODS))
    bad = []
    for (i, t), paper in PAPER.items():
        vals = [ms[x][i, t] for x in METHODS]
        cells = []
        for x, v, pv in zip(METHODS, vals, paper):
            if np.isnan(v):
                cells.append(f"{'-':>8}")
                bad.append((ORDER[i], t, x, "missing", pv))
            else:
                mark = "" if abs(round(v, 3) - pv) < 5e-4 else "*"
                if mark:
                    bad.append((ORDER[i], t, x, round(float(v), 4), pv))
                cells.append(f"{v:7.3f}{mark or ' '}")
        say(f"S{t} {ORDER[i]:10} " + " ".join(cells))

    def summary(m):
        F = np.mean([np.mean(m[j, j + 1:] - m[j, j]) for j in range(3)])
        A = np.mean([m[k, k] for k in (1, 2, 3)])
        return {"F_avg": F, "A_avg": A, "m_fin": np.mean(m[:3, 3])}

    for key in ("F_avg", "A_avg", "m_fin"):
        cells = []
        for x, pv in zip(METHODS, PAPER_SUMMARY[key]):
            v = summary(ms[x])[key]
            mark = "" if np.isfinite(v) and abs(round(v, 3) - pv) < 5e-4 else "*"
            if mark:
                bad.append(("summary", key, x, None if not np.isfinite(v) else round(float(v), 4), pv))
            cells.append(f"{v:7.3f}{mark or ' '}" if np.isfinite(v) else f"{'-':>8}")
        say(f"{key:14} " + " ".join(cells))
    say("")
    say("* = differs from the draft at 3 decimals")
    say(f"SUMMARY {len(bad)} cell(s) differ from the draft or are missing" + (":" if bad else ""))
    for b in bad[:30]:
        say(f"  {b}")
    if len(bad) > 30:
        say(f"  ... {len(bad) - 30} more in the JSON report")
    report["fd_g"] = {x: ms[x].tolist() for x in METHODS}
    report["fd_g_mismatches"] = bad


# ------------------------------------------------------------------------------------------------ [8]
def all_buffers(a, report):
    head("[8] Every buffer: clips kept against the budget (GCR fills an early stop with random clips)")
    rows = []
    for method in sorted(os.listdir(a.buffers)) if os.path.isdir(a.buffers) else []:
        for j in range(3):
            buf = load_json(os.path.join(a.buffers, method, f"s{j}_{ORDER[j]}.json"))
            if buf is None:
                continue
            info = {k: v for k, v in buf.get("info", {}).items() if not isinstance(v, (list, dict))}
            r = {"method": method, "session": j, "clips": len(buf["entries"]), "budget": buf.get("budget"),
                 "candidates": buf.get("candidates"), "seed": buf.get("seed"), "ckpt": buf.get("ckpt"), "info": info}
            rows.append(r)
            say(f"{method:9} s{j} {ORDER[j]}: {r['clips']:>4} / {r['budget']} of {r['candidates']} | seed {r['seed']}"
                f" | ckpt {r['ckpt']} | {json.dumps(info)}")
    report["buffers"] = rows


def main():
    a = parse_args()
    report = {}
    say(f"repo {os.getcwd()}")
    for step in (split_table, tmr_buffers, resume_check, config_check, results_check, all_buffers):
        try:
            step(a, report)
        except Exception as ex:  # noqa: BLE001  keep going: one missing piece should not hide the rest
            say(f"ERROR in {step.__name__}: {ex!r}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(report, f, indent=1, default=str)
    with open(a.out + ".txt", "w") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"\nwritten {a.out}.txt and {a.out}.json")


if __name__ == "__main__":
    main()
