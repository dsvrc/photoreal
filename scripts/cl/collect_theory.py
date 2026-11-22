"""
The Audio2Photoreal counterparts of the "Evidence from DiscoForcing" notes in the TMR supplementary, computed
from the cached trajectories (cl_data/features/*.npz) and the saved buffers. CPU only, a few minutes:

    python scripts/cl/collect_theory.py

Writes the report to stdout, cl_data/theory_report.txt and cl_data/theory_report.json. Per TMR buffer
(sessions 0-2, selected right after their session):

  [A] FAQ (vi), "selection adds no noise": re-running cl.selectors.select_tmr on the cached trajectories gives
      exactly the saved buffer (same clips, same weights)?
  [B] Prop. fw: eps_mod of the saved buffer, ||y||, diam_Z, the bound 2 diam_Z / sqrt(B + 2), the eps_mod(b)
      curve over the greedy iterations and its log-log slope, whether the pursuit stopped before the budget
  [C] Lemma modal: in-subspace part ||P Xi||_F against sqrt(lambda_1) eps_mod, out-of-subspace part
      ||(I - P) Xi||_F against tau_k and against sqrt(||w||_inf N D sum_{m > M} lambda_m), and eps_B against the
      whole right-hand side
  [D] Modal rank / Remark rank_whiten: M_hat with n_eff = N*D (used) and n_eff = N, and the overlap of the
      two buffers
  [E] At the shared Session-0 model: eps_mod and eps_B of every method's PXB184 buffer on TMR's objective
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cl import selectors  # noqa: E402

ORDER = ["PXB184", "RLW104", "TXB805", "GQS883"]
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


def feature_path(feature_dir, ckpt, subject, steps):  # same as cl.build_buffer.feature_path
    slug = os.path.splitext(os.path.normpath(ckpt))[0].replace("\\", "/").strip("./").replace("/", "_")
    return os.path.join(feature_dir, f"{slug}__{subject}_T{steps}.npz")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def buffer_indices(buf, scenes, starts):
    """Indices into the feature file of the buffer's clips, and their simplex weights."""
    pos = {(str(s), int(t)): i for i, (s, t) in enumerate(zip(scenes, starts))}
    idx = np.array([pos[(e["scene"], int(e["start"]))] for e in buf["entries"]], dtype=int)
    w = np.array([e.get("weight", 1.0) for e in buf["entries"]], dtype=float)
    return idx, w / w.sum()


def omp_trace(Z, target, budget):
    """cl.selectors.omp_simplex (stop="fw") with eps_mod recorded after every added clip."""
    n = len(Z)
    G, b, yy = Z @ Z.T, Z @ target, float(target @ target)
    first = int(np.argmin(np.diag(G) - 2 * b))
    S, w = [first], np.ones(1)
    used = np.zeros(n, dtype=bool)
    used[first] = True

    def err():
        return float(np.sqrt(max(yy - 2 * b[S] @ w + w @ G[np.ix_(S, S)] @ w, 0.0)))

    curve, stopped = [err()], False
    while len(S) < min(budget, n):
        corr = b - G[:, S] @ w
        gap = corr - w @ corr[S]
        gap[used] = -np.inf
        j = int(np.argmax(gap))
        if not np.isfinite(gap[j]) or gap[j] <= 1e-12 * yy:
            stopped = True
            break
        S.append(j)
        used[j] = True
        w = selectors.simplex_lsq(G[np.ix_(S, S)], b[S], np.append(w, 0.0))
        keep = w > 1e-6 / len(w)
        if not keep.all():
            S = [s for s, k in zip(S, keep) if k]
            w = w[keep] / w[keep].sum()
        curve.append(err())
    return np.array(curve), stopped


def slope(curve):
    b = np.arange(1, len(curve) + 1)
    ok = curve > 0
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(b[ok]), np.log(curve[ok]), 1)[0])


def analyse(j, subject, buf, phi, scenes, starts):
    N, T, D = phi.shape
    budget = int(buf["budget"])
    lam_all, V_all = selectors.eig_desc(selectors.temporal_operator(phi))
    M = selectors.bic_rank(lam_all, N * D)
    V, lam = V_all[:, :M], lam_all[:M]
    Z = selectors.modal_atoms(phi, V, lam)
    y = Z.mean(axis=0)
    idx, w = buffer_indices(buf, scenes, starts)
    r = {"session": j, "subject": subject, "N": N, "T": T, "D": D, "B": budget, "clips": int(len(idx)), "M": int(M),
         "M_saved": buf.get("info", {}).get("M")}

    # [A] determinism
    idx2, w2, info2 = selectors.select_tmr(phi, budget, "fw")
    w2 = np.asarray(w2, dtype=float) / np.sum(w2)
    same_set = set(idx.tolist()) == set(np.asarray(idx2).tolist())
    order = {int(i): k for k, i in enumerate(idx2)}
    wdiff = float(max(abs(w[k] - w2[order[int(i)]]) for k, i in enumerate(idx))) if same_set else float("nan")
    r.update(rebuild_same_clips=same_set, rebuild_max_weight_diff=wdiff, rebuild_fw_stop=info2.get("fw_stop"))

    # [B] Prop. fw
    eps_mod = float(np.linalg.norm(y - w @ Z[idx]))
    sq = (Z**2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    diam = float(np.sqrt(d2.max()))
    curve, stopped = omp_trace(Z, y, budget)
    r.update(eps_mod=eps_mod, y_norm=float(np.linalg.norm(y)), rel_err=eps_mod / float(np.linalg.norm(y)),
             diam_Z=diam, fw_bound=2 * diam / np.sqrt(budget + 2), curve_first=float(curve[0]),
             curve_last=float(curve[-1]), curve_slope=slope(curve), curve_slope_from2=slope(curve[1:]),
             stopped_early=bool(stopped), curve=curve.round(5).tolist())
    # expected eps_mod of a uniform random subset of B clips with uniform weights
    spread2 = float(np.mean(sq - 2 * Z @ y + y @ y))
    r["eps_mod_random_expected"] = float(np.sqrt(spread2 / budget * (N - budget) / max(N - 1, 1)))

    # [C] Lemma modal
    phit = phi - phi.mean(axis=0, keepdims=True)
    Xi = -np.einsum("i,itd->td", w, phit[idx])
    P = V @ V.T
    in_sub = float(np.linalg.norm(P @ Xi))
    out_sub = float(np.linalg.norm(Xi - P @ Xi))
    resid = phit - np.einsum("st,itd->isd", P, phit)
    tau = float(np.sqrt((resid**2).sum(axis=(1, 2))).max())
    alt = float(np.sqrt(w.max() * N * D * lam_all[M:].sum()))
    r.update(eps_B=float(np.linalg.norm(Xi)), in_sub=in_sub, in_sub_bound=float(np.sqrt(lam[0]) * eps_mod),
             out_sub=out_sub, tau_k=tau, out_sub_alt_bound=alt,
             lemma_rhs=float(np.sqrt(lam[0]) * eps_mod + min(tau, alt)), lambda_1=float(lam[0]),
             discarded_energy=float(lam_all[M:].sum() / lam_all.sum()))

    # [D] rank with n_eff = N and the overlap of the two buffers
    M_N = selectors.bic_rank(lam_all, N)
    Z_N = selectors.modal_atoms(phi, V_all[:, :M_N], lam_all[:M_N])
    idx_N, _, info_N = selectors.omp_simplex(Z_N, Z_N.mean(axis=0), budget, "fw")
    r.update(M_N=int(M_N), overlap_N=len(set(idx.tolist()) & set(np.asarray(idx_N).tolist())) / max(len(idx), 1),
             clips_N=int(len(idx_N)), fw_stop_N=info_N.get("fw_stop"))
    return r, (Z, y, phit, P, V, lam)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--buffers", default="cl_data/buffers")
    p.add_argument("--feature_dir", default="cl_data/features")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out", default="cl_data/theory_report")
    a = p.parse_args()
    say(f"repo {os.getcwd()}")
    rows, s0 = [], None

    for j in range(3):
        subject = ORDER[j]
        head(f"TMR buffer of session {j} ({subject})")
        buf = load_json(os.path.join(a.buffers, "tmr", f"s{j}_{subject}.json"))
        if buf is None:
            say("MISSING buffer")
            continue
        fp = feature_path(a.feature_dir, buf["ckpt"], subject, a.steps)
        if not os.path.exists(fp):
            say(f"MISSING features {fp}")
            continue
        d = np.load(fp)
        phi = d["phi"].astype(np.float64)
        try:
            r, ctx = analyse(j, subject, buf, phi, d["scenes"], d["starts"])
        except Exception as ex:  # noqa: BLE001
            say(f"ERROR {ex!r}")
            continue
        rows.append(r)
        if j == 0:
            s0 = (buf["ckpt"], d["scenes"], d["starts"], ctx)
        say(f"N {r['N']}, T {r['T']}, D {r['D']}, B {r['B']}, clips {r['clips']}, M_hat {r['M']} (saved {r['M_saved']})")
        say(f"[A] rebuild gives the same clips: {r['rebuild_same_clips']} | max weight difference"
            f" {r['rebuild_max_weight_diff']:.2e} | rebuild fw_stop {r['rebuild_fw_stop']}")
        say(f"[B] eps_mod {r['eps_mod']:.4f} (||y|| {r['y_norm']:.4f}, relative {r['rel_err']:.4f}; random subset of B"
            f" expected {r['eps_mod_random_expected']:.4f}) | diam_Z {r['diam_Z']:.4f} | bound 2 diam/sqrt(B+2)"
            f" {r['fw_bound']:.4f} | ratio bound/eps {r['fw_bound'] / max(r['eps_mod'], 1e-12):.1f}")
        say(f"    eps_mod over the greedy iterations: {r['curve_first']:.4f} after the first clip -> {r['curve_last']:.4f}"
            f" after iteration {len(r['curve'])} (pruned refits can make this exceed B),"
            f" log-log slope {r['curve_slope']:.3f} (from b=2: {r['curve_slope_from2']:.3f}; bound rate -0.5)"
            f" | stopped before budget {r['stopped_early']}")
        say(f"[C] ||P Xi|| {r['in_sub']:.4f} <= sqrt(lam1) eps_mod {r['in_sub_bound']:.4f}"
            f" (factor {r['in_sub_bound'] / max(r['in_sub'], 1e-12):.2f})"
            f" | ||(I-P) Xi|| {r['out_sub']:.4f} <= tau_k {r['tau_k']:.4f} and <= alt {r['out_sub_alt_bound']:.4f}"
            f" | eps_B {r['eps_B']:.4f} <= rhs {r['lemma_rhs']:.4f} | discarded energy {r['discarded_energy']:.4f}")
        say(f"[D] M_hat n_eff=N*D {r['M']}, n_eff=N {r['M_N']} | buffer with n_eff=N: {r['clips_N']} clips"
            f" (fw_stop {r['fw_stop_N']}), overlap with the saved buffer {100 * r['overlap_N']:.0f}%")

    head("[E] every method's PXB184 buffer at the shared Session-0 model, on TMR's objective")
    e_rows = []
    if s0 is not None:
        ckpt, scenes, starts, (Z, y, phit, P, V, lam) = s0
        for method in sorted(os.listdir(a.buffers)):
            buf = load_json(os.path.join(a.buffers, method, f"s0_{ORDER[0]}.json"))
            if buf is None or buf.get("ckpt") != ckpt:
                continue
            idx, w = buffer_indices(buf, scenes, starts)
            wu = np.full(len(idx), 1.0 / len(idx))
            row = {"method": method, "clips": int(len(idx)),
                   "eps_mod": float(np.linalg.norm(y - w @ Z[idx])),
                   "eps_mod_uniform_w": float(np.linalg.norm(y - wu @ Z[idx])),
                   "eps_B": float(np.linalg.norm(np.einsum("i,itd->td", w, phit[idx])))}
            e_rows.append(row)
            say(f"{method:9} {row['clips']:>3} clips | eps_mod {row['eps_mod']:.4f} (uniform weights"
                f" {row['eps_mod_uniform_w']:.4f}) | eps_B {row['eps_B']:.4f}")
    else:
        say("no Session-0 TMR features")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump({"tmr": rows, "s0_buffers": e_rows}, f, indent=1, default=str)
    with open(a.out + ".txt", "w") as f:
        f.write("\n".join(OUT) + "\n")
    print(f"\nwritten {a.out}.txt and {a.out}.json")


if __name__ == "__main__":
    main()
