"""
Model-side pieces of buffer selection for one finished session:

    ClipBank        the subject's candidate clips, normalized as in training, with their frozen audio
                    embeddings (encode_audio = vq-wav2vec of both speakers) kept in memory
    clip_features   per-clip condition trajectories and pose statistics for OCS, GCR and TMR
    clip_losses     per-clip training loss with fixed noise, comparable across models (MIR, CSReL)
    select_mir, select_csrel

Scoring runs with no dropout anywhere (eval mode, and split_type "test" for the pose head, which
otherwise drops out whenever split_type == "train").
"""

import contextlib
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cl.common import fixed_rng, patched
from cl.replay import ReplayWindows, replay_collate


def identity_audio(emb):
    return emb  # y["audio"] already holds encode_audio's output


@contextlib.contextmanager
def module_mode(model, train: bool):
    """Train or eval for the whole model; every submodule's own flag is restored afterwards."""
    flags = [(m, m.training) for m in model.modules()]
    split = model.split_type
    model.train(train)
    model.split_type = "train" if train else "test"
    try:
        yield model
    finally:
        for m, flag in flags:
            m.training = flag
        model.split_type = split


def light_copy(model):
    """Deep copy without the frozen wav2vec; use it with identity_audio and embeddings in y["audio"]."""
    audio = model.audio_model
    model.audio_model = None
    try:
        return copy.deepcopy(model)
    finally:
        model.audio_model = audio


def stream_embeddings(model, entries, dataset_root, window, device, batch=8, workers=2):
    """(motion, y, audio embedding) for consecutive batches of clips, in entry order."""
    loader = DataLoader(
        ReplayWindows(entries, dataset_root, window=window),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=replay_collate,
    )
    done = 0
    for motion, cond, _ in loader:
        with torch.no_grad():
            emb = model.encode_audio(cond["y"]["audio"].to(device))
        done += len(motion)
        print(f"\r  {done}/{len(entries)} clips", end="", flush=True)
        yield motion, cond["y"], emb
    print()


def clip_features(model, entries, dataset_root, window, device, steps=100, batch=8, workers=2):
    """phi [N x steps x D] = cond_projection(encode_audio(audio)) pooled from ~100 Hz to `steps`,
    pose [N x steps x 104] pooled the same way, pose_stats [N x 208] (per-channel mean and std)."""
    phi, pose, stats = [], [], []
    with module_mode(model, train=False):
        for motion, _, emb in stream_embeddings(model, entries, dataset_root, window, device, batch, workers):
            with torch.no_grad():
                traj = model.cond_projection(emb)
            phi.append(F.adaptive_avg_pool1d(traj.transpose(1, 2), steps).transpose(1, 2).float().cpu().numpy())
            m = motion.squeeze(2).double()  # B x 104 x frames, normalized with the subject's stats
            pose.append(F.adaptive_avg_pool1d(m, steps).transpose(1, 2).numpy())
            stats.append(torch.cat([m.mean(-1), m.std(-1)], dim=1).numpy())
    return {"phi": np.concatenate(phi), "pose": np.concatenate(pose), "pose_stats": np.concatenate(stats)}


class ClipBank:
    """A session's candidate clips in memory, laid out like a replay batch, with the frozen audio
    embedding (fp16) in place of the raw audio."""

    def __init__(self, model, entries, dataset_root, window, device, batch=8, workers=2):
        motions, ys, embs = [], [], []
        with module_mode(model, train=False):
            for motion, y, emb in stream_embeddings(model, entries, dataset_root, window, device, batch, workers):
                motions.append(motion)
                ys.append({k: v for k, v in y.items() if k != "audio"})
                embs.append(emb.half().cpu())
        self.motion = torch.cat(motions)
        self.y = {k: torch.cat([y[k] for y in ys]) for k in ys[0]}
        self.emb = torch.cat(embs)
        self.n = len(self.motion)

    def batch(self, idx, device):
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        y = {k: v[idx].to(device) for k, v in self.y.items()}
        y["audio"] = self.emb[idx].to(device, torch.float32)
        return self.motion[idx].to(device), {"y": y}


def clip_losses(model, diffusion, bank, idx, seeds=(0, 1, 2, 3), batch=25, device="cuda"):
    """Per-clip training loss (rot + velocity MSE) averaged over fixed noise seeds.

    Time steps, noise and condition dropout depend only on the seed and on which clips share a batch
    (idx is cut into batches in order), so models scored on the same idx see identical draws.
    """
    idx = np.asarray(idx)
    out = np.zeros(len(idx))
    with module_mode(model, train=False), patched(model, "encode_audio", identity_audio), torch.no_grad():
        for s in range(0, len(idx), batch):
            b = idx[s : s + batch]
            motion, cond = bank.batch(b, device)
            total = torch.zeros(len(b), device=device, dtype=torch.float64)
            for seed in seeds:
                with fixed_rng(seed * 1_000_003 + int(b[0])):
                    t = torch.randint(0, diffusion.num_timesteps, (len(b),), device=device)
                    noise = torch.randn_like(motion)
                    loss = diffusion.training_losses(model, motion, t, model_kwargs={"y": cond["y"]}, noise=noise)
                total += loss["loss"].double()
            out[s : s + len(b)] = (total / len(seeds)).cpu().numpy()
    return out


# ----------------------------------------------------------------------------- MIR


def batch_gradient(model, diffusion, bank, idx, seed, device):
    params = model.parameters_w_grad()
    motion, cond = bank.batch(idx, device)
    with module_mode(model, train=False), patched(model, "encode_audio", identity_audio), fixed_rng(seed):
        t = torch.randint(0, diffusion.num_timesteps, (len(idx),), device=device)
        loss = diffusion.training_losses(model, motion, t, model_kwargs=cond)["loss"].mean()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def select_mir(
    model, diffusion, bank, budget, rng, lr, update="adam", cand=50, evals=3, vbatch=16, seeds=(0, 1, 2, 3),
    batch=25, device="cuda",
):
    """Clips whose loss rises most after one virtual step on a random batch of the same session.

    theta' = theta - lr * sign(g) (the first AdamW step) or theta - lr * g (sgd). Interference =
    loss(theta') - loss(theta), clamped at 0; each clip is scored under `evals` different steps.
    Ties (clips whose loss never rose) are ordered by the unclamped change.
    """
    virtual = light_copy(model)
    params, vparams = model.parameters_w_grad(), virtual.parameters_w_grad()
    total, change, count = np.zeros(bank.n), np.zeros(bank.n), np.zeros(bank.n)
    for e in range(evals):
        order = rng.permutation(bank.n)
        for c in range(0, bank.n, cand):
            chunk = order[c : c + cand]
            vb = rng.choice(bank.n, size=min(vbatch, bank.n), replace=False)
            grads = batch_gradient(model, diffusion, bank, vb, int(rng.integers(2**31)), device)
            with torch.no_grad():
                for pv, p, g in zip(vparams, params, grads):
                    pv.copy_(p - lr * (g.sign() if update == "adam" else g))
            before = clip_losses(model, diffusion, bank, chunk, seeds, batch, device)
            after = clip_losses(virtual, diffusion, bank, chunk, seeds, batch, device)
            total[chunk] += np.maximum(after - before, 0.0)
            change[chunk] += after - before
            count[chunk] += 1
        n_scored = count.sum()
        print(
            f"  MIR pass {e + 1}/{evals}: mean interference {total.sum() / n_scored:.3e},"
            f" mean loss change {change.sum() / n_scored:+.3e}",
            flush=True,
        )
    score, change = total / np.maximum(count, 1), change / np.maximum(count, 1)
    idx = np.lexsort((-change, -score))[:budget]
    info = {
        "mean_interference": float(score.mean()),
        "buffer_interference": float(score[idx].mean()),
        "clips_with_interference": float((score > 0).mean()),
        "mean_loss_change": float(change.mean()),
    }
    return idx, np.ones(len(idx)), info


# ----------------------------------------------------------------------------- CSReL


def finetune(model, diffusion, bank, coreset, steps, lr, clip, batch, rng, device):
    params = model.parameters_w_grad()
    opt = torch.optim.Adam(params, lr=lr)
    with module_mode(model, train=True), patched(model, "encode_audio", identity_audio):
        for _ in range(steps):
            motion, cond = bank.batch(rng.choice(coreset, size=batch, replace=True), device)
            t = torch.randint(0, diffusion.num_timesteps, (len(motion),), device=device)
            loss = diffusion.training_losses(model, motion, t, model_kwargs=cond)["loss"].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
    return float(loss)


def select_csrel(
    model, prev_state, diffusion, bank, budget, rng, rounds=5, steps=100, lr=1e-4, clip=1.0, train_batch=32,
    seeds=(0, 1, 2, 3), batch=25, device="cuda",
):
    """Clips with the largest reducible loss, loss(theta_S) - loss(theta_D), added in greedy rounds.

    theta_D = `model`, trained on the whole session; theta_S = prev_state, the model before the session,
    fine-tuned on the coreset so far (from scratch each round) before scoring.
    """
    every = np.arange(bank.n)
    loss_D = clip_losses(model, diffusion, bank, every, seeds, batch, device)
    work = light_copy(model)
    keys = set(work.state_dict())
    prev_state = {k: v for k, v in prev_state.items() if k in keys}
    per_round = math.ceil(budget / rounds)
    coreset, history = [], []
    for r in range(rounds):
        if len(coreset) >= budget:
            break
        work.load_state_dict(prev_state)
        ft_loss = finetune(work, diffusion, bank, coreset, steps, lr, clip, train_batch, rng, device) if coreset else None
        reducible = clip_losses(work, diffusion, bank, every, seeds, batch, device) - loss_D
        reducible[coreset] = -np.inf
        new = np.argsort(-reducible, kind="stable")[: min(per_round, budget - len(coreset))]
        coreset += new.tolist()
        history.append({"round": r + 1, "added": len(new), "reducible": float(reducible[new].mean()), "finetune_loss": ft_loss})
        print(f"  CSReL round {r + 1}/{rounds}: +{len(new)} clips, mean reducible loss {history[-1]['reducible']:.4f}", flush=True)
    return np.asarray(coreset, dtype=int), np.ones(len(coreset)), {"rounds": history, "loss_after_session": float(loss_D.mean())}
