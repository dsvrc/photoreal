"""
One continual-learning session for the body (pose) diffusion model.

Starts from the previous session's checkpoint, trains on the new subject(s) for exactly
--steps steps with a fresh AdamW, and optionally replays weighted windows from earlier
subjects: every step adds one replay batch (as large as the training batch by default) drawn
from the shuffled union of the buffers, loss = mean(new) + lambda * mean(w * replay).
On GPUs with at least 60 GB both batches go through the model together; on smaller ones
(--passes 2) one after the other, each with its own backward pass, before one optimizer step:
the gradient of the same loss with the memory of one batch.
The session's scenes and the replay buffers stay on the GPU, where every batch is cut
(cl.session_data.SessionCrops, cl.replay.ReplayBatches), so no loader workers move audio.
Every subject keeps its own data_stats.pth normalization, as in the original loader.

Session 1, no replay (vanilla), from the released PXB184 body model:
    python -m cl.train_session --init_ckpt checkpoints/diffusion/c1_pose/model000340000.pt \
        --data_root dataset/RLW104 --save_dir runs/vanilla/s1_RLW104 --steps 60000

Same session with replay of an earlier buffer (cl.build_buffer):
    ... --replay cl_data/buffers/er/s0_PXB184.json
Random* (STAR on the replay batch):
    ... --replay cl_data/buffers/star/s0_PXB184.json --star
Joint upper bound (all subjects at once):
    ... --data_root dataset/PXB184 dataset/RLW104 dataset/TXB805 dataset/GQS883
"""

import argparse
import contextlib
import copy
import glob
import json
import os
import re
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from cl.common import build_model, load_ckpt_args, patched, subject_of
from cl.replay import ReplayBatches, ReplayWindows, read_buffers
from cl.session_data import PreNormalizedSocial, SessionCrops
from cl.star import STAR
from data_loaders.get_data import load_local_data
from diffusion.resample import create_named_schedule_sampler
from utils.misc import fixseed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init_ckpt", required=True, help="checkpoint of the previous session (args.json next to it)")
    p.add_argument("--data_root", required=True, nargs="+", help="subject folder(s) of this session")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--steps", required=True, type=int, help="exact number of optimizer steps")
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=None, help="new-data batch (default: from init args.json)")
    p.add_argument("--lr", type=float, default=None, help="default: from init args.json")
    p.add_argument("--replay", nargs="*", default=[], help="buffer JSON files of earlier sessions")
    p.add_argument("--replay_batch", type=int, default=None, help="default: the training batch size")
    p.add_argument("--replay_lambda", type=float, default=1.0)
    p.add_argument("--star", action="store_true", help="add the STAR loss on every replay batch (Random*)")
    p.add_argument("--star_lambda", type=float, default=0.1)
    p.add_argument("--star_gamma", type=float, default=0.01)
    p.add_argument("--star_eps", type=float, default=0.001)
    p.add_argument("--passes", type=int, choices=[1, 2], default=None,
                   help="model passes per step with replay: 1 = new and replay batch together (about twice the "
                   "memory: 40 GB GPUs run out), 2 = one after the other; default 1 on GPUs with 60 GB or more")
    p.add_argument("--dataset_root", default="dataset", help="where replay subjects live")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in save_dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_tf32", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def latest_checkpoint(save_dir):
    ckpts = sorted(glob.glob(os.path.join(save_dir, "model*.pt")))
    if not ckpts:
        return None, 0
    return ckpts[-1], int(re.search(r"model(\d+)\.pt$", ckpts[-1]).group(1))


def main():
    cl_args = parse_args()
    fixseed(cl_args.seed)
    torch.backends.cudnn.benchmark = True  # fixed-size inputs
    if not cl_args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    start_ckpt, start_step = None, 0
    if cl_args.resume:
        start_ckpt, start_step = latest_checkpoint(cl_args.save_dir)
    elif glob.glob(os.path.join(cl_args.save_dir, "model*.pt")) and not cl_args.overwrite:
        raise FileExistsError(f"{cl_args.save_dir} already has checkpoints; use --resume or --overwrite")
    os.makedirs(cl_args.save_dir, exist_ok=True)

    # model/diffusion args come from the previous session so the architecture always matches
    args = load_ckpt_args(cl_args.init_ckpt)
    assert args.data_format == "pose", "this trainer is for the body (pose) model"
    args.timestep_respacing = ""  # train on the full 1000-step chain
    args.save_dir = cl_args.save_dir
    args.data_root = cl_args.data_root[0]
    args.batch_size = cl_args.batch_size or args.batch_size
    args.lr = cl_args.lr or args.lr
    replay_batch = cl_args.replay_batch or args.batch_size
    assert not cl_args.star or cl_args.replay, "--star acts on the replay batch; pass --replay buffers"
    device = torch.device(cl_args.device)
    passes = cl_args.passes
    if passes is None:
        big = device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory >= 60 * 2**30
        passes = 1 if big else 2

    saved = dict(vars(args))
    saved["cl"] = {
        "init_ckpt": cl_args.init_ckpt,
        "data_roots": cl_args.data_root,
        "steps": cl_args.steps,
        "replay": cl_args.replay,
        "replay_batch": replay_batch,
        "replay_lambda": cl_args.replay_lambda,
        "passes": passes,
        "star": {"lambda": cl_args.star_lambda, "gamma": cl_args.star_gamma, "eps": cl_args.star_eps}
        if cl_args.star
        else None,
        "seed": cl_args.seed,
    }
    with open(os.path.join(cl_args.save_dir, "args.json"), "w") as f:
        json.dump(saved, f, indent=4, sort_keys=True)

    # new-session data: the repo's Social crops (random 400-600 frames, own stats), normalized once and
    # cut on the device; a resumed run continues with a new random stream instead of repeating the first
    datasets = []
    for root in cl_args.data_root:
        sub_args = copy.copy(args)
        sub_args.data_root = root
        datasets.append(
            PreNormalizedSocial(args=sub_args, data_dict=load_local_data(root, audio_per_frame=1600), split="train")
        )
    new_batches = SessionCrops(datasets, args.batch_size, device, seed=(cl_args.seed, start_step))
    del datasets  # the host copies; the scenes now live on the device
    data_bytes = new_batches.nbytes

    replay_batches = None
    if cl_args.replay:
        entries = read_buffers(cl_args.replay)
        replay_batches = ReplayBatches(
            ReplayWindows(entries, cl_args.dataset_root, window=args.max_seq_length),
            replay_batch,
            (cl_args.steps - start_step + 1) * replay_batch,
            seed=cl_args.seed + start_step,
            device=device,
        )
        data_bytes += replay_batches.nbytes
        mean_w = np.mean([e.get("weight", 1.0) for e in entries])
        print(f"replay: {len(entries)} windows from {cl_args.replay} (mean weight {mean_w:.3f})")

    model, diffusion = build_model(args, start_ckpt or cl_args.init_ckpt, split_type="train")
    model.to(device)  # no model.train(): like the original trainer, the frozen wav2vec stays in eval mode
    diffusion.compute_vb = False  # only logged by the original trainer; saves a forward pass
    sampler = create_named_schedule_sampler("uniform", diffusion)
    opt = AdamW(model.parameters_w_grad(), lr=args.lr, weight_decay=args.weight_decay)
    if start_ckpt:
        opt_path = os.path.join(cl_args.save_dir, f"opt{start_step:09d}.pt")
        if os.path.exists(opt_path):
            opt.load_state_dict(torch.load(opt_path, map_location=device))
        print(f"resuming from {start_ckpt} at step {start_step}")
    star = STAR(model, cl_args.star_lambda, cl_args.star_gamma, cl_args.star_eps) if cl_args.star else None
    n_params = sum(p.numel() for p in model.parameters_w_grad()) / 1e6
    replay_note = f"+{replay_batch} replay{' with STAR' if star else ''} in {passes} pass{'es' if passes > 1 else ''}"
    print(
        f"trainable params {n_params:.2f}M | subjects {[subject_of(r) for r in cl_args.data_root]}"
        f" | batch {args.batch_size} ({replay_note if replay_batches else '+0 replay'})"
        f" | data on {device} ({data_bytes / 2**30:.1f} GB) | lr {args.lr} | steps {start_step}->{cl_args.steps}"
    )

    writer = SummaryWriter(cl_args.save_dir)

    def save(step):
        torch.save(model.state_dict(), os.path.join(cl_args.save_dir, f"model{step:09d}.pt"))
        torch.save(opt.state_dict(), os.path.join(cl_args.save_dir, f"opt{step:09d}.pt"))

    def losses(motion, cond):
        t, _ = sampler.sample(motion.shape[0], device)
        return diffusion.training_losses(model, motion, t, model_kwargs=cond)["loss"]

    t0, running = time.time(), {"new": 0.0, "replay": 0.0, "star": 0.0, "n": 0, "data": 0.0}
    for step in range(start_step, cl_args.steps):
        t_data = time.time()
        motion, cond = next(new_batches)
        replay = next(replay_batches) if replay_batches is not None else None
        running["data"] += time.time() - t_data

        # loss = mean(new) + lambda * mean(w * replay) [+ STAR]. With 2 passes each batch has its own
        # backward: the gradients add up to the gradient of the sum, with one batch's activations in memory
        opt.zero_grad(set_to_none=True)
        if replay is None:
            loss_new = losses(motion, cond).mean()
            loss_new.backward()
        else:
            r_motion, r_cond, weights = replay
            encoded = contextlib.nullcontext()
            if star is not None:
                # STAR runs three more forwards on the replay clips: encode the audio (frozen wav2vec) once
                with torch.no_grad():
                    for c in (cond, r_cond):
                        c["y"]["audio"] = model.encode_audio(c["y"]["audio"])
                encoded = patched(model, "encode_audio", lambda emb: emb)
            with encoded:
                if passes == 1:
                    n_new = motion.shape[0]
                    both = {"y": {k: torch.cat([cond["y"][k], r_cond["y"][k]]) for k in cond["y"]}}
                    per_sample = losses(torch.cat([motion, r_motion]), both)
                    loss_new, loss_replay = per_sample[:n_new].mean(), (weights * per_sample[n_new:]).mean()
                    (loss_new + cl_args.replay_lambda * loss_replay).backward()
                else:
                    loss_new = losses(motion, cond).mean()
                    loss_new.backward()
                    loss_replay = (weights * losses(r_motion, r_cond)).mean()
                    (cl_args.replay_lambda * loss_replay).backward()
                if star is not None:
                    loss_star = star.loss(diffusion, r_motion, r_cond, seed=cl_args.seed * 1_000_003 + step)
                    loss_star.backward()
                    running["star"] += loss_star.item()
            running["replay"] += loss_replay.item()
        opt.step()

        running["new"] += loss_new.item()
        running["n"] += 1
        done = step + 1
        if done % cl_args.log_every == 0:
            elapsed = time.time() - t0
            rate = running["n"] / elapsed
            data_pct = 100 * running["data"] / elapsed
            msg = f"step {done}/{cl_args.steps} | loss new {running['new'] / running['n']:.5f}"
            writer.add_scalar("loss/new", running["new"] / running["n"], done)
            if replay_batches is not None:
                msg += f" replay {running['replay'] / running['n']:.5f}"
                writer.add_scalar("loss/replay", running["replay"] / running["n"], done)
            if star is not None:
                msg += f" star {running['star'] / running['n']:.2e}"
                writer.add_scalar("loss/star", running["star"] / running["n"], done)
            eta_h = (cl_args.steps - done) / max(rate, 1e-9) / 3600
            # "waiting for data": the share of time spent getting batches (cutting them, with data on the GPU)
            print(f"{msg} | {rate:.2f} it/s | waiting for data {data_pct:.0f}% | eta {eta_h:.1f} h", flush=True)
            t0, running = time.time(), {"new": 0.0, "replay": 0.0, "star": 0.0, "n": 0, "data": 0.0}
        if done % cl_args.save_every == 0 or done == cl_args.steps:
            save(done)

    writer.close()
    print(f"done: {os.path.join(cl_args.save_dir, f'model{cl_args.steps:09d}.pt')}")


if __name__ == "__main__":
    main()
