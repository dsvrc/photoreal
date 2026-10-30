"""
One continual-learning session for the body (pose) diffusion model.

Starts from the previous session's checkpoint, trains on the new subject(s) for exactly
--steps steps with a fresh AdamW, and optionally replays weighted windows from earlier
subjects (loss = mean(new) + lambda * mean(w * replay), i.e. Eq. 7 with weights of mean 1).
Every subject keeps its own data_stats.pth normalization, as in the original loader.

Session 1, no replay (vanilla), from the released PXB184 body model:
    python -m cl.train_session --init_ckpt checkpoints/diffusion/c1_pose/model000340000.pt \
        --data_root dataset/RLW104 --save_dir runs/vanilla/s1_RLW104 --steps 60000

Same session with replay of an earlier buffer:
    ... --replay cl_data/buffers/er/s0_PXB184.json
Joint upper bound (all subjects at once):
    ... --data_root dataset/PXB184 dataset/RLW104 dataset/TXB805 dataset/GQS883
"""

import argparse
import copy
import glob
import json
import os
import re
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from cl.common import build_model, load_ckpt_args, subject_of
from cl.replay import ReplayWindows, read_buffers, replay_collate
from cl.session_data import PreNormalizedSocial
from data_loaders.get_data import load_local_data
from data_loaders.tensors import social_collate
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
    p.add_argument("--replay_batch", type=int, default=8)
    p.add_argument("--replay_lambda", type=float, default=1.0)
    p.add_argument("--dataset_root", default="dataset", help="where replay subjects live")
    p.add_argument("--num_workers", type=int, default=None, help="default: min(8, CPUs of this job)")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in save_dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_tf32", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def seed_worker(worker_id):
    # the loader crops windows with np.random; give each worker its own stream
    np.random.seed(torch.initial_seed() % 2**32)


def to_device(cond, device):
    cond["y"] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond["y"].items()}
    return cond


def concat_batches(a, b):
    """New-data batch + replay batch; replay tensors take the new batch's dtypes."""
    motion = torch.cat([a[0], b[0].to(a[0].dtype)], dim=0)
    y = {k: torch.cat([a[1]["y"][k], b[1]["y"][k].to(a[1]["y"][k].dtype)], dim=0) for k in a[1]["y"]}
    return motion, {"y": y}


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
    num_workers = cl_args.num_workers
    if num_workers is None:
        num_workers = min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())

    saved = dict(vars(args))
    saved["cl"] = {
        "init_ckpt": cl_args.init_ckpt,
        "data_roots": cl_args.data_root,
        "steps": cl_args.steps,
        "replay": cl_args.replay,
        "replay_batch": cl_args.replay_batch,
        "replay_lambda": cl_args.replay_lambda,
        "seed": cl_args.seed,
    }
    with open(os.path.join(cl_args.save_dir, "args.json"), "w") as f:
        json.dump(saved, f, indent=4, sort_keys=True)

    # new-session data: the repo's Social crops (random 400-600 frames, own stats), normalized once
    datasets = []
    for root in cl_args.data_root:
        sub_args = copy.copy(args)
        sub_args.data_root = root
        datasets.append(
            PreNormalizedSocial(args=sub_args, data_dict=load_local_data(root, audio_per_frame=1600), split="train")
        )
    new_data = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    new_iter = cycle(
        DataLoader(
            new_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=social_collate,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=num_workers > 0,
        )
    )

    replay_iter = None
    if cl_args.replay:
        entries = read_buffers(cl_args.replay)
        replay_data = ReplayWindows(entries, cl_args.dataset_root, window=args.max_seq_length)
        mean_w = np.mean([e.get("weight", 1.0) for e in entries])
        print(f"replay: {len(entries)} windows from {cl_args.replay} (mean weight {mean_w:.3f})")
        replay_iter = iter(
            DataLoader(
                replay_data,
                batch_size=cl_args.replay_batch,
                sampler=RandomSampler(
                    replay_data, replacement=True, num_samples=(cl_args.steps - start_step + 1) * cl_args.replay_batch
                ),
                num_workers=max(1, num_workers // 4) if num_workers > 0 else 0,
                drop_last=True,
                collate_fn=replay_collate,
                pin_memory=True,
            )
        )

    device = torch.device(cl_args.device)
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
    n_params = sum(p.numel() for p in model.parameters_w_grad()) / 1e6
    print(
        f"trainable params {n_params:.2f}M | subjects {[subject_of(r) for r in cl_args.data_root]}"
        f" | batch {args.batch_size} (+{cl_args.replay_batch if replay_iter else 0} replay)"
        f" | lr {args.lr} | workers {num_workers} | steps {start_step}->{cl_args.steps}"
    )

    writer = SummaryWriter(cl_args.save_dir)

    def save(step):
        torch.save(model.state_dict(), os.path.join(cl_args.save_dir, f"model{step:09d}.pt"))
        torch.save(opt.state_dict(), os.path.join(cl_args.save_dir, f"opt{step:09d}.pt"))

    t0, running = time.time(), {"new": 0.0, "replay": 0.0, "n": 0, "data": 0.0}
    for step in range(start_step, cl_args.steps):
        t_data = time.time()
        motion, cond = next(new_iter)
        n_new = motion.shape[0]
        weights = None
        if replay_iter is not None:
            r_motion, r_cond, weights = next(replay_iter)
            motion, cond = concat_batches((motion, cond), (r_motion, r_cond))
        running["data"] += time.time() - t_data
        motion = motion.to(device)
        cond = to_device(cond, device)

        t, _ = sampler.sample(motion.shape[0], device)
        per_sample = diffusion.training_losses(model, motion, t, model_kwargs=cond)["loss"]
        loss_new = per_sample[:n_new].mean()
        loss = loss_new
        if weights is not None:
            loss_replay = (weights.to(device) * per_sample[n_new:]).mean()
            loss = loss + cl_args.replay_lambda * loss_replay
            running["replay"] += loss_replay.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
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
            if replay_iter is not None:
                msg += f" replay {running['replay'] / running['n']:.5f}"
                writer.add_scalar("loss/replay", running["replay"] / running["n"], done)
            eta_h = (cl_args.steps - done) / max(rate, 1e-9) / 3600
            # "waiting for data" near 0% means the GPU is the limit; high means the loader is
            print(f"{msg} | {rate:.2f} it/s | waiting for data {data_pct:.0f}% | eta {eta_h:.1f} h", flush=True)
            t0, running = time.time(), {"new": 0.0, "replay": 0.0, "n": 0, "data": 0.0}
        if done % cl_args.save_every == 0 or done == cl_args.steps:
            save(done)

    writer.close()
    print(f"done: {os.path.join(cl_args.save_dir, f'model{cl_args.steps:09d}.pt')}")


if __name__ == "__main__":
    main()
