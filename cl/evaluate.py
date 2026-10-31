"""
Evaluate body-model checkpoints on subjects' val/test chunks (all 600-frame chunks from the
manifest, i.e. the repo's chunked test set) with FD_g (static pose) and FD_k (velocity).

    # session 0 (released PXB184 model) on PXB184
    python -m cl.evaluate --ckpt checkpoints/diffusion/c1_pose/model000340000.pt --subjects PXB184
    # session 1 checkpoint on both subjects seen so far
    python -m cl.evaluate --ckpt runs/vanilla/s1_RLW104/model000060000.pt --subjects PXB184 RLW104
    # pick the step budget on val over all saved checkpoints
    python -m cl.evaluate --ckpt runs/vanilla/s1_RLW104/model*.pt --subjects PXB184 RLW104 --split val

Guide-pose modes (the body model is conditioned on audio + guide poses every 30 frames):
    audio_only  guide poses replaced by the model's learned null embedding: audio-driven motion
    gt_guide    ground-truth guide poses (oracle keyframes, what the model sees in training)
    guide       guide poses from a released guide transformer: --guide_ckpt SUBJECT=path
Same noise for every checkpoint (fixed --seed), DDIM with --respacing, CFG scale --guidance.
Each result is written to <ckpt dir>/eval/*.json and skipped when it already exists.
"""

import argparse
import contextlib
import glob
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cl.common import build_model, load_ckpt_args, load_stats
from cl.replay import ReplayWindows, replay_collate
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.eval import calculate_activation_statistics, calculate_frechet_distance

MODES = ("audio_only", "gt_guide", "guide")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--modes", nargs="+", default=["audio_only", "gt_guide"], choices=MODES)
    p.add_argument("--guide_ckpt", nargs="*", default=[], help="SUBJECT=checkpoints/guide/cN_pose/checkpoints/iter-XXXXXXX.pt")
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--respacing", default="ddim100")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--manifest", default="cl_data/manifest.json")
    p.add_argument("--dataset_root", default="dataset")
    p.add_argument("--out_dir", default=None, help="default: <ckpt dir>/eval")
    p.add_argument("--save_samples", action="store_true", help="also store generated poses (.npz)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@contextlib.contextmanager
def patched(obj, name, value):
    """Temporarily shadow a method on one module instance."""
    setattr(obj, name, value)
    try:
        yield
    finally:
        delattr(obj, name)


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    mu_a, cov_a = calculate_activation_statistics(a)
    mu_b, cov_b = calculate_activation_statistics(b)
    return float(calculate_frechet_distance(mu_a, cov_a, mu_b, cov_b))


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred/gt: [chunks x frames x 104] joint angles."""
    vel_p, vel_g = np.diff(pred, axis=1), np.diff(gt, axis=1)
    return {
        "fd_g": frechet(gt.reshape(-1, gt.shape[-1]), pred.reshape(-1, pred.shape[-1])),
        "fd_k": frechet(vel_g.reshape(-1, gt.shape[-1]), vel_p.reshape(-1, pred.shape[-1])),
        "var_k_pred": float(np.var(pred, axis=1).mean()),  # temporal variance; collapses toward 0 if motion freezes
        "var_k_gt": float(np.var(gt, axis=1).mean()),
        "mse_to_gt": float(np.mean((pred - gt) ** 2)),
    }


def guide_keyframes(base, y):
    """Keyframes from the loaded guide transformer, as sample/generate.py does."""
    B, T = y["keyframes"].shape[0], y["keyframes"].shape[1]
    with torch.no_grad():
        tokens = base.transformer.generate(y["audio"], T, layers=base.tokenizer.residual_depth, n_sequences=B)
    tokens = tokens.reshape((B, -1, base.tokenizer.residual_depth))
    return base.tokenizer.decode(tokens).detach().to(y["keyframes"].device, y["keyframes"].dtype)


def evaluate_one(base, cfg, diffusion, subject, chunks, mode, a, stats):
    entries = [{"subject": subject, "scene": s, "start": st, "weight": 1.0} for s, st in chunks]
    loader = DataLoader(
        ReplayWindows(entries, a.dataset_root, window=base.seq_len),
        batch_size=a.batch,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        collate_fn=replay_collate,
    )
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    preds, gts = [], []

    def null_keyframes(y, cond_drop_prob, batch_size):
        return base.null_pose_embed[:, : y["keyframes"].shape[1]].expand(batch_size, -1, -1)

    for motion, cond, _ in loader:
        y = {k: v.to(a.device) if torch.is_tensor(v) else v for k, v in cond["y"].items()}
        B = motion.shape[0]
        y["scale"] = torch.full((B,), a.guidance, device=a.device)
        if mode == "guide":
            y["keyframes"] = guide_keyframes(base, y)
        noise = torch.randn((B, base.nfeats, 1, motion.shape[-1]), generator=gen, device=a.device)
        with torch.no_grad():
            audio_emb = base.encode_audio(y["audio"])  # frozen wav2vec: depends on the audio only
            with contextlib.ExitStack() as stack:
                stack.enter_context(patched(base, "encode_audio", lambda raw_audio: audio_emb))
                if mode == "audio_only":
                    stack.enter_context(patched(base, "encode_keyframes", null_keyframes))
                sample = diffusion.ddim_sample_loop(
                    cfg, noise.shape, noise=noise, clip_denoised=False, model_kwargs={"y": y}, progress=False
                )
        # [B x 104 x 1 x T] (normalized) -> [B x T x 104] joint angles with the subject's own stats
        to_raw = lambda x: x.squeeze(2).permute(0, 2, 1).cpu().numpy() * stats["pose_std"] + stats["pose_mean"]
        preds.append(to_raw(sample))
        gts.append(to_raw(motion))
    return np.concatenate(preds), np.concatenate(gts)


def main():
    a = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with open(a.manifest) as f:
        manifest = json.load(f)
    guides = dict(g.split("=", 1) for g in a.guide_ckpt)
    if "guide" in a.modes:
        missing = [s for s in a.subjects if s not in guides]
        assert not missing, f"--modes guide needs --guide_ckpt for {missing}"
    ckpts = sorted({c for pattern in a.ckpt for c in (glob.glob(pattern) or [pattern])})

    rows = []
    for ckpt in ckpts:
        out_dir = a.out_dir or os.path.join(os.path.dirname(ckpt), "eval")
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(ckpt))[0]
        todo = []
        for subject in a.subjects:
            for mode in a.modes:
                path = os.path.join(out_dir, f"{name}_{subject}_{a.split}_{mode}_seed{a.seed}.json")
                if os.path.exists(path) and not a.overwrite:
                    with open(path) as f:
                        rows.append(json.load(f))
                else:
                    todo.append((subject, mode, path))
        if not todo:
            continue

        args = load_ckpt_args(ckpt)
        args.timestep_respacing = a.respacing
        base, diffusion = build_model(args, ckpt, split_type="test")
        base.to(a.device).eval()
        cfg = ClassifierFreeSampleModel(base).to(a.device).eval()
        for subject, mode, path in todo:
            if mode == "guide":
                base.setup_guide_predictor(guides[subject])
                base.to(a.device).eval()
            chunks = manifest["subjects"][subject][f"{a.split}_chunks"]
            t0 = time.time()
            pred, gt = evaluate_one(
                base, cfg, diffusion, subject, chunks, mode, a, load_stats(os.path.join(a.dataset_root, subject))
            )
            row = {
                "ckpt": ckpt,
                "subject": subject,
                "split": a.split,
                "mode": mode,
                "seed": a.seed,
                "guidance": a.guidance,
                "respacing": a.respacing,
                "chunks": len(chunks),
                "guide_ckpt": guides.get(subject) if mode == "guide" else None,
                **metrics(pred, gt),
                "seconds": round(time.time() - t0, 1),
            }
            with open(path, "w") as f:
                json.dump(row, f, indent=1)
            if a.save_samples:
                np.savez_compressed(path.replace(".json", ".npz"), pred=pred.astype(np.float32), gt=gt.astype(np.float32))
            rows.append(row)
            print(
                f"{ckpt} | {subject} {a.split} {mode}: FD_g {row['fd_g']:.4f}  FD_k {row['fd_k']:.6f}"
                f"  var_k {row['var_k_pred']:.4f} (gt {row['var_k_gt']:.4f})  [{row['seconds']} s]",
                flush=True,
            )

    print(f"\n{'checkpoint':58s} {'subject':8s} {'mode':10s} {'FD_g':>9s} {'FD_k':>11s} {'var_k':>8s}")
    for r in sorted(rows, key=lambda r: (r["mode"], r["subject"], r["ckpt"])):
        print(f"{r['ckpt'][-58:]:58s} {r['subject']:8s} {r['mode']:10s} {r['fd_g']:9.4f} {r['fd_k']:11.6f} {r['var_k_pred']:8.4f}")


if __name__ == "__main__":
    main()
