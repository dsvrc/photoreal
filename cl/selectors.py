"""
Buffer selectors that only need precomputed clip features: random (ER), OCS, GCR, TMR (paper)
and TMR-joint. Each returns (indices, weights[, info]) with weights of mean 1.

Clip features (cl.scoring.clip_features), for N candidate clips:
    phi         N x T x D    condition-encoder output per time step: cond_projection(wav2vec(audio))
                             of both speakers, average-pooled from ~100 Hz to T steps
    pose        N x T x 104  the diffusion target (normalized body pose), pooled to the same T steps
    pose_stats  N x 208      per-channel mean and std of the normalized pose over the clip
"""

import numpy as np
from scipy.sparse.csgraph import connected_components

EPS = 1e-12


def l2_normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def unit_mean_norm(block):
    return block / max(float(np.linalg.norm(block, axis=1).mean()), EPS)


def mean_one(weights):
    w = np.asarray(weights, dtype=np.float64)
    return w * len(w) / w.sum() if len(w) and w.sum() > 0 else np.ones(len(w))


def fill_budget(idx, weights, n, budget, rng):
    """Weights to mean 1; if the pursuit stopped early, top up with random clips of weight 1."""
    idx, weights = list(idx), list(mean_one(weights))
    if len(idx) < budget:
        rest = np.setdiff1d(np.arange(n), idx)
        extra = rng.choice(rest, size=min(budget - len(idx), len(rest)), replace=False)
        idx += extra.tolist()
        weights += [1.0] * len(extra)
    return np.asarray(idx, dtype=int), np.asarray(weights)


# ----------------------------------------------------------------------------- Random (ER)


def select_random(n, budget, rng):
    idx = np.sort(rng.choice(n, size=min(budget, n), replace=False))
    return idx, np.ones(len(idx))


# ----------------------------------------------------------------------------- OCS and GCR


def ocs_features(phi, pose_stats):
    """Time-averaged condition features next to the target statistics, each block at unit mean norm."""
    return np.concatenate([unit_mean_norm(phi.mean(axis=1)), unit_mean_norm(pose_stats)], axis=1)


def select_ocs(feats, budget, rng, batch=128):
    """Cosine to the minibatch centroid minus mean cosine to the other clips of the minibatch."""
    f = l2_normalize(feats)
    score = np.empty(len(f))
    order = rng.permutation(len(f))
    for s in range(0, len(f), batch):
        b = order[s : s + batch]
        fb = f[b]
        centroid = fb.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), EPS)
        sim = fb @ fb.T
        redundancy = (sim.sum(axis=1) - np.diag(sim)) / max(len(b) - 1, 1)
        score[b] = fb @ centroid - redundancy
    idx = np.argsort(-score, kind="stable")[:budget]
    return idx, np.ones(len(idx))


def omp_select(A, target, budget, lam=0.5, tol=1e-4):
    """Regularised non-negative OMP, as GCR's orthogonalmp_reg_parallel (CRAIG / GradMatch).

    Adds the atom (row of A) most aligned with the residual, refits ridge weights on the support and
    drops atoms whose weight turns negative (they are not picked again). Stops at the budget, when no
    atom is positively aligned, or when the residual falls below tol of the target.
    """
    n = len(A)
    G, b, yy = A @ A.T, A @ target, float(target @ target)
    support, w, banned = [], np.zeros(0), np.zeros(n, dtype=bool)
    for _ in range(4 * budget):
        if len(support) >= budget:
            break
        S = np.asarray(support, dtype=int)
        resid2 = yy - 2 * b[S] @ w + w @ G[np.ix_(S, S)] @ w
        if resid2 <= tol**2 * yy:
            break
        corr = b - G[:, S] @ w
        corr[banned] = -np.inf
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        support.append(j)
        banned[j] = True
        while support:
            S = np.asarray(support, dtype=int)
            w = np.linalg.solve(G[np.ix_(S, S)] + lam * np.eye(len(S)), b[S])
            if w.min() >= 0:
                break
            support.pop(int(np.argmin(w)))
        if not support:
            w = np.zeros(0)
    return np.asarray(support, dtype=int), w


def select_gcr(feats, budget, rng, lam=0.5, tol=1e-4):
    A = l2_normalize(feats)
    idx, w = omp_select(A, A.sum(axis=0), budget, lam, tol)
    return fill_budget(idx, w, len(A), budget, rng)


# ----------------------------------------------------------------------------- TMR


def temporal_operator(traj):
    """C = 1/(N D) sum_i phi~_i phi~_i^T (T x T), with phi~ centred over clips."""
    N, T, D = traj.shape
    flat = (traj - traj.mean(axis=0, keepdims=True)).transpose(1, 0, 2).reshape(T, N * D)
    return flat @ flat.T / (N * D)


def eig_desc(C):
    lam, V = np.linalg.eigh(C)
    return np.clip(lam[::-1], 0.0, None), V[:, ::-1]


def bic_rank(lam, n):
    """Rank M by the BIC of probabilistic PCA (Tipping & Bishop 1999; Minka 2000), n samples of dim T."""
    T = len(lam)
    ev = np.clip(lam, 1e-14, None)
    best_M, best = 1, np.inf
    for M in range(1, T - 1):
        sig2 = ev[M:].mean()
        if sig2 <= 1e-13:
            break
        loglik = -0.5 * n * (np.log(ev[:M]).sum() + (T - M) * np.log(sig2) + T * np.log(2 * np.pi) + T)
        df = T * M - M * (M - 1) / 2 + 1
        bic = -2 * loglik + df * np.log(n)
        if bic < best:
            best_M, best = M, bic
    return best_M


def temporal_modes(traj, n_eff):
    """Top temporal modes of the trajectories and their eigenvalues; the rank comes from the BIC."""
    lam, V = eig_desc(temporal_operator(traj))
    M = bic_rank(lam, n_eff)
    info = {"T": len(lam), "M": M, "n_eff": float(n_eff), "energy_kept": float(lam[:M].sum() / max(lam.sum(), EPS))}
    return V[:, :M], lam[:M], info


def modal_atoms(traj, V, lam):
    """z_i = vec(Lambda_M^{-1/2} V_M^T phi_i) for the uncentred trajectories."""
    beta = np.einsum("tm,ntd->nmd", V, traj)
    scale = 1.0 / np.sqrt(np.maximum(lam, 1e-10 * lam[0]))
    return (beta * scale[None, :, None]).reshape(len(traj), -1)


def project_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1), 0.0)


def simplex_lsq(G, b, w0=None, iters=20000, tol=1e-10):
    """argmin over the simplex of ||y - Z_S^T w||^2, given G = Z_S Z_S^T and b = Z_S y.

    Accelerated projected gradient, stopped when the Frank-Wolfe duality gap on the support is ~0.
    """
    L = max(float(np.linalg.eigvalsh(G)[-1]), EPS)
    w = project_simplex(w0) if w0 is not None else np.full(len(b), 1.0 / len(b))
    z, t = w.copy(), 1.0
    for _ in range(iters):
        w_next = project_simplex(z - (G @ z - b) / L)
        t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = w_next + (t - 1) / t_next * (w_next - w)
        w, t = w_next, t_next
        g = G @ w - b
        if g @ w - g.min() <= tol * L:
            break
    return w


def omp_simplex(Z, target, budget, stop="fw"):
    """Greedy support with simplex-constrained refits towards target (the mean atom).

    Starts from the atom closest to the target, then adds the atom with the largest Frank-Wolfe gap
    <z_i - Z_S w, r> (the one most correlated with the residual r) and refits w on the simplex.
    stop="fw" stops when no atom lowers the error; stop="budget" keeps going to the budget.
    Atoms whose weight falls to zero leave the support and are not picked again.
    """
    n = len(Z)
    G, b, yy = Z @ Z.T, Z @ target, float(target @ target)
    first = int(np.argmin(np.diag(G) - 2 * b))
    S, w = [first], np.ones(1)
    used = np.zeros(n, dtype=bool)
    used[first] = True
    fw_stop = False
    while len(S) < min(budget, n):
        corr = b - G[:, S] @ w
        gap = corr - w @ corr[S]
        gap[used] = -np.inf
        j = int(np.argmax(gap))
        if not np.isfinite(gap[j]):
            break
        if gap[j] <= 1e-12 * yy:
            fw_stop = True
            if stop == "fw":
                break
        S.append(j)
        used[j] = True
        w = simplex_lsq(G[np.ix_(S, S)], b[S], np.append(w, 0.0))
        keep = w > 1e-6 / len(w)
        if not keep.all():
            S = [s for s, k in zip(S, keep) if k]
            w = w[keep] / w[keep].sum()
    resid2 = max(yy - 2 * b[S] @ w + w @ G[np.ix_(S, S)] @ w, 0.0)
    # the error a uniform random subset of the same size makes on average (sampling without replacement)
    spread2 = float(np.mean(np.diag(G) - 2 * b + yy))
    k = len(S)
    random2 = spread2 / k * (n - k) / max(n - 1, 1)
    info = {
        "rel_err": float(np.sqrt(resid2 / max(yy, EPS))),
        "err_vs_random": float(np.sqrt(resid2 / max(random2, EPS))),
        "fw_stop": fw_stop,
    }
    return np.asarray(S, dtype=int), w, info


def select_tmr(phi, budget, stop="fw"):
    """Paper TMR: whitened temporal modes of the condition trajectory, BIC with n = N * D."""
    phi = np.asarray(phi, dtype=np.float64)
    N, T, D = phi.shape
    V, lam, info = temporal_modes(phi, N * D)
    Z = modal_atoms(phi, V, lam)
    idx, w, pinfo = omp_simplex(Z, Z.mean(axis=0), budget, stop)
    return idx, mean_one(w), {**info, **pinfo}


def balanced_blocks(*blocks):
    """Concatenate trajectory blocks along channels, each scaled to the same mean centred energy."""
    scaled = []
    for X in blocks:
        energy = ((X - X.mean(axis=0, keepdims=True)) ** 2).sum() / len(X)
        scaled.append(X / np.sqrt(max(energy, EPS)))
    return np.concatenate(scaled, axis=2)


def condition_groups(cond, eps):
    """Label clips whose condition trajectories coincide (relative distance below eps)."""
    flat = cond.reshape(len(cond), -1).astype(np.float64)
    sq = (flat**2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * flat @ flat.T
    close = d2 <= eps**2 * np.maximum(sq[:, None], sq[None, :])
    return connected_components(close, directed=False)[1]


def effective_samples(traj, groups):
    """Satterthwaite n_eff = (tr K)^2 / ||K||_F^2 for K = clip-group structure (x) channel correlation.

    Equals N * D for independent clips and uncorrelated channels.
    """
    N, T, D = traj.shape
    sizes = np.bincount(groups)
    n_clips = N * N / float((sizes**2).sum())
    X = (traj - traj.mean(axis=0, keepdims=True)).reshape(-1, D)
    var = (X**2).mean(axis=0)
    live = var > EPS * var.max()
    Xs = X[:, live] / np.sqrt(var[live])
    R = Xs.T @ Xs / len(Xs)
    n_channels = float(np.trace(R) ** 2 / (R**2).sum())
    return n_clips * n_channels, {"n_eff_clips": n_clips, "n_eff_channels": n_channels, "groups": int(len(sizes))}


def select_tmr_joint(phi, pose, budget, rng, mix_eps=0.02):
    """TMR on the condition trajectory next to the pose target (equal block weight), the BIC sample
    count from effective_samples, and the pursuit run to the budget. Only if the mean is matched
    exactly before that (atoms of fewer dimensions than the budget) is the rest filled at random."""
    phi, pose = np.asarray(phi, dtype=np.float64), np.asarray(pose, dtype=np.float64)
    traj = balanced_blocks(phi, pose)
    n_eff, ninfo = effective_samples(traj, condition_groups(phi, mix_eps))
    V, lam, info = temporal_modes(traj, n_eff)
    Z = modal_atoms(traj, V, lam)
    idx, w, pinfo = omp_simplex(Z, Z.mean(axis=0), budget, stop="budget")
    pinfo["random_fill"] = int(budget - len(idx))
    idx, w = fill_budget(idx, w, len(Z), budget, rng)
    return idx, w, {**info, **ninfo, **pinfo}
