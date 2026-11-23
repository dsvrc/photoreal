"""
Render the qualitative Audio2Photoreal frames from cl_data/qual_pack (scripts/cl/collect_videos.py) on a machine
without pytorch3d: the photoreal body decoder of visualize/ca_body runs as is, and the stand-ins in
visualize/local_render/shim replace the pytorch3d rasterizer (numba, CPU).

    # 1) candidates: score every frame from the poses, render the best-scoring frames of every row
    python visualize/local_render/render_qual.py candidates --pack <qual_pack dir> --out <dir>
    # 2) figure: chosen chunk and frames, rows as in the paper, matplotlib with the paper's fonts
    python visualize/local_render/render_qual.py figure --pack <qual_pack dir> --out <dir> \
        --chunk 12 --frames 40 115 190 ... --png fig/qual_a2p.png
    # 3) video of one chunk (rows side by side, conversation audio; needs imageio-ffmpeg)
    python visualize/local_render/render_qual.py video --pack <qual_pack dir> --out <dir> --chunk 12

Frame score (step 1): the Session-0 model and the later models share the audio and the generation noise, so a model
that kept the participant stays close to what the Session-0 model generated. A frame scores high when fine-tuning
after the last session is far from the Session-0 model's pose (and from the real pose) and TMR is close to it.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(HERE, "shim"), HERE, REPO]

import torch as th  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["candidates", "partner", "figure", "video"])
    p.add_argument("--pack", required=True)
    p.add_argument("--partner_pack", default=None, help="cl_data/partner_pack (scripts/cl/collect_partner.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--view", type=int, default=0, help="camera index of the render defaults")
    p.add_argument("--per_chunk", type=int, default=10, help="candidates: frames rendered per chunk")
    p.add_argument("--min_gap", type=int, default=30, help="candidates: frames apart")
    p.add_argument("--chunk", type=int, default=None)
    p.add_argument("--frames", type=int, nargs="*", default=None)
    p.add_argument("--rows", nargs="*", default=None, help="figure/video: row keys (default: every row but tmr_joint)")
    p.add_argument("--png", default=None, help="figure: output image")
    p.add_argument("--title", default=None)
    p.add_argument("--crop_frac", type=float, default=1.0, help="figure: keep this top fraction of the body (0.6: head to thighs)")
    p.add_argument("--max_frames", type=int, default=600, help="video: first N frames")
    return p.parse_args()


class Renderer:
    """visualize/render_codes.BodyRenderer for one camera, without torchaudio/mediapy, with the coverage mask."""

    def __init__(self, base, subject, view, device):
        from attrdict import AttrDict
        from omegaconf import OmegaConf

        from visualize.ca_body.utils.train import load_checkpoint, load_from_config

        config = OmegaConf.load(os.path.join(base, "config.yml"))
        assets = AttrDict(th.load(os.path.join(base, "static_assets.pt"), map_location="cpu", weights_only=False))
        self.device = device
        self.model = load_from_config(config.model, assets=assets).to(device)
        self.model.cal_enabled = False
        self.model.pixel_cal_enabled = False
        self.model.learn_blur_enabled = False
        load_checkpoint(os.path.join(base, "body_dec.ckpt"), modules={"model": self.model},
                        ignore_names={"model": ["lbs_fn.*"]})
        self.model.eval().to(device)
        defaults = th.load(os.path.join(REPO, "assets", f"render_defaults_{subject}.pth"), map_location=device,
                           weights_only=False)
        B = defaults["K"].shape[0]
        self.inputs = {k: (v[view : view + 1] if th.is_tensor(v) and v.ndim > 0 and v.shape[0] == B else v)
                       for k, v in defaults.items()}
        # keep the rasterizer's coverage, to put the avatar on a white background
        rast = self.model.renderer.rasterizer
        self.last = {}

        def wrapped(meshes, cameras):
            frag = rast(meshes, cameras)
            self.last["mask"] = frag.pix_to_face[0, ..., 0].cpu().numpy() >= 0
            return frag

        self.model.renderer.rasterizer = wrapped

    @th.no_grad()
    def __call__(self, body, face):
        """body [104], face [256] -> RGB uint8 [H x W x 3] (white background), mask [H x W]."""
        from visualize.ca_body.utils.image import linear2displayBatch

        m = self.model
        inp = copy.copy(self.inputs)
        motion = th.tensor(body[None], device=self.device, dtype=th.float)
        inp["lbs_motion"] = motion
        inp["geom"] = m.lbs_fn.lbs_fn(motion, m.lbs_fn.lbs_scale.unsqueeze(0), m.lbs_fn.lbs_template_verts.unsqueeze(0)) \
            * m.lbs_fn.global_scaling
        inp["face_embs"] = th.tensor(face[None], device=self.device, dtype=th.float)
        preds = m(**inp)
        rgb = linear2displayBatch(preds["rgb"])[0].permute(1, 2, 0).clip(0, 255).byte().cpu().numpy()
        mask = self.last["mask"]
        out = np.full_like(rgb, 255)
        out[mask] = rgb[mask]
        return out, mask


def load_pack(pack):
    d = np.load(os.path.join(pack, "rows.npz"))
    keys = [str(k) for k in d["keys"]]
    return d["bodies"], d["faces"], [int(c) for c in d["chunks"]], keys, [str(x) for x in d["labels"]], str(d["subject"])


def frame_scores(bodies_c, keys):
    """Per frame: distance of fine-tuning after the last session and of TMR to the Session-0 model and to Real."""
    rows = {k: bodies_c[i] for i, k in enumerate(keys)}
    last_ft = max((k for k in keys if k.startswith("ft_s")), key=lambda k: int(k[4:]))
    ref = rows["start"]
    d = lambda a, b: np.sqrt(((a[:, 6:] - b[:, 6:]) ** 2).sum(1))  # skip the 6 root dims
    s = (d(rows[last_ft], ref) - d(rows["tmr_s3"], ref)) + 0.5 * (d(rows[last_ft], rows["real"]) - d(rows["tmr_s3"], rows["real"]))
    return s


def pick(score, k, gap):
    order, chosen = np.argsort(-score), []
    for f in order:
        if all(abs(int(f) - c) >= gap for c in chosen):
            chosen.append(int(f))
        if len(chosen) == k:
            break
    return sorted(chosen)


def crop_box(masks, pad=12):
    m = np.any(np.stack(masks), axis=0)
    ys, xs = np.where(m)
    return max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad


def main():
    a = parse_args()
    from PIL import Image

    bodies, faces, chunks, keys, labels, subject = load_pack(a.pack)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if th.cuda.is_available() else "cpu"
    cache = {}

    def R(body, face):  # the participant's renderer, loaded on first use
        if "r" not in cache:
            cache["r"] = Renderer(os.path.join(a.pack, "ca_body", subject), subject, a.view, device)
        return cache["r"](body, face)

    if a.stage == "partner":
        # the conversation partner's recorded motion in the same frames, with the partner's own renderer
        pd = np.load(os.path.join(a.partner_pack, "partner.npz"))
        partner = str(pd["partner"])
        pci = [int(c) for c in pd["chunks"]].index(a.chunk)
        print(f"partner {partner}, chunk {a.chunk}: audio correlation {float(pd['audio_corr'][pci]):.3f}")
        PR = Renderer(os.path.join(a.partner_pack, "ca_body", partner), partner, a.view, device)
        for f in a.frames:
            path = os.path.join(a.out, f"c{a.chunk:02d}", f"f{f:03d}_partner.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img, mask = PR(pd["bodies"][pci, f], pd["faces"][pci, f])
            Image.fromarray(img).save(path)
            np.save(path.replace(".png", "_mask.npy"), mask)
        with open(os.path.join(a.out, "partner.json"), "w") as fh:
            json.dump({"partner": partner}, fh)
        print(f"partner frames in {a.out}/c{a.chunk:02d}/f???_partner.png")

    elif a.stage == "candidates":
        index = {}
        for ci, c in enumerate(chunks):
            s = frame_scores(bodies[ci], keys)
            frames = pick(s, a.per_chunk, a.min_gap)
            index[c] = {"frames": frames, "scores": [float(s[f]) for f in frames]}
            print(f"chunk {c}: frames {frames}", flush=True)
            for f in frames:
                for ri, k in enumerate(keys):
                    path = os.path.join(a.out, f"c{c:02d}", f"f{f:03d}_{k}.png")
                    if os.path.exists(path):
                        continue
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    img, mask = R(bodies[ci, ri, f], faces[ci, f])
                    Image.fromarray(img).save(path)
                    np.save(path.replace(".png", "_mask.npy"), mask)
                # contact sheet of this frame: rows side by side
                imgs = [np.asarray(Image.open(os.path.join(a.out, f"c{c:02d}", f"f{f:03d}_{k}.png"))) for k in keys]
                masks = [np.load(os.path.join(a.out, f"c{c:02d}", f"f{f:03d}_{k}_mask.npy")) for k in keys]
                y0, y1, x0, x1 = crop_box(masks)
                sheet = np.concatenate([im[y0:y1, x0:x1] for im in imgs], axis=1)
                Image.fromarray(sheet).save(os.path.join(a.out, f"c{c:02d}", f"sheet_f{f:03d}.jpg"), quality=85)
        with open(os.path.join(a.out, "candidates.json"), "w") as fh:
            json.dump({"keys": keys, "labels": labels, "chunks": index}, fh, indent=1)
        print(f"sheets (columns: {', '.join(labels)}) in {a.out}/c??/sheet_f???.jpg")

    elif a.stage == "figure":
        ci = chunks.index(a.chunk)
        rows = a.rows or [k for k in keys if k != "tmr_joint_s3"]
        own = [k for k in rows if k != "partner"]  # the participant's rows share one camera and one crop
        load = lambda k, f: os.path.join(a.out, f"c{a.chunk:02d}", f"f{f:03d}_{k}.png")
        bbox = {}  # (row, frame) -> bounding box of the avatar; full frames are not kept (RAM)
        for k in rows:
            for f in a.frames:
                path = load(k, f)
                if not os.path.exists(path):
                    assert k != "partner", f"render the partner first: {path}"
                    img, mask = R(bodies[ci, keys.index(k), f], faces[ci, f])
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    Image.fromarray(img).save(path)
                    np.save(path.replace(".png", "_mask.npy"), mask)
                    del img
                bbox[(k, f)] = crop_box([np.load(path.replace(".png", "_mask.npy"))])

        def union(bs):
            return min(b[0] for b in bs), max(b[1] for b in bs), min(b[2] for b in bs), max(b[3] for b in bs)

        def crop(k, f, y0, y1, x0, x1):
            with Image.open(load(k, f)) as im:
                return np.asarray(im.crop((x0, y0, x1, y1)))

        # the participant's rows: one crop per column (frame) across rows, the same height for all columns
        boxes = {f: union([bbox[(k, f)] for k in own]) for f in a.frames}
        y0 = min(b[0] for b in boxes.values())
        y1 = max(b[1] for b in boxes.values())
        y1 = y0 + int(round((y1 - y0) * a.crop_frac))
        H = y1 - y0
        crops = {(k, f): crop(k, f, y0, y1, boxes[f][2], boxes[f][3]) for k in own for f in a.frames}
        # the partner (another person, another camera): its own crop with the same body fraction, scaled to H
        if "partner" in rows:
            pboxes = {f: bbox[("partner", f)] for f in a.frames}
            py0 = min(b[0] for b in pboxes.values())
            py1 = max(b[1] for b in pboxes.values())
            py1 = py0 + int(round((py1 - py0) * a.crop_frac))
            for f in a.frames:
                c = crop("partner", f, py0, py1, pboxes[f][2], pboxes[f][3])
                w = int(round(c.shape[1] * H / c.shape[0]))
                crops[("partner", f)] = np.asarray(Image.fromarray(c).resize((w, H), Image.LANCZOS))
        gap = 24
        widths = {f: max(crops[(k, f)].shape[1] for k in rows) + gap for f in a.frames}
        partner = json.load(open(os.path.join(a.out, "partner.json")))["partner"] if "partner" in rows else None
        strips = []
        for k in rows:
            cells = []
            for f in a.frames:
                c = crops[(k, f)]
                cell = np.full((H, widths[f], 3), 255, dtype=np.uint8)
                l = (widths[f] - c.shape[1]) // 2
                cell[:, l : l + c.shape[1]] = c
                cells.append(cell)
            if k == "partner":
                name = f"Partner ({partner}), real"
            else:
                name = labels[keys.index(k)]
                name = {"Real": f"{subject}, real"}.get(name, name)
                name = name.replace("Fine-tuning after", "Fine-tuning,").replace("TMR after", "TMR (ours),")
            strips.append((name, np.concatenate(cells, axis=1)))
        compose(strips, a.png or os.path.join(a.out, "qual_a2p.png"), a.title,
                separator_after=1 if rows[0] == "partner" else None)

    else:  # video
        import imageio.v2 as imageio

        ci = chunks.index(a.chunk)
        rows = a.rows or ["start", "ft_s3", "tmr_s3", "real"]
        path = os.path.join(a.out, f"c{a.chunk:02d}_compare_silent.mp4")
        writer = imageio.get_writer(path, fps=30, codec="libx264", quality=7, macro_block_size=1)
        box = None
        for f in range(min(a.max_frames, bodies.shape[2])):
            imgs = [R(bodies[ci, keys.index(k), f], faces[ci, f]) for k in rows]
            if box is None:
                box = crop_box([m for _, m in imgs], pad=60)
            y0, y1, x0, x1 = box
            frame = np.concatenate([im[y0:y1, x0:x1] for im, _ in imgs], axis=1)
            writer.append_data(frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2])
            if f % 50 == 0:
                print(f"frame {f}", flush=True)
        writer.close()
        print(f"{path} (columns: {', '.join(labels[keys.index(k)] for k in rows)}); add audio/c{a.chunk:02d}.wav with ffmpeg")


def compose(strips, out, title, separator_after=None):
    """Rows of frames with row labels, as scripts/cl/qual_frames.py of DiscoForcing (the paper's fonts).
    separator_after: a thin rule and a small gap after that many rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix"})
    from PIL import Image

    # downscale the strips to twice the printed resolution first (matplotlib works in float64 RGBA)
    fig_w, lab_w = 7.0, 1.25
    target = int((fig_w - lab_w) * 300 * 2)
    strips = [(n, np.asarray(Image.fromarray(im).resize((target, int(round(im.shape[0] * target / im.shape[1]))),
                                                         Image.LANCZOS)) if im.shape[1] > target else im)
              for n, im in strips]
    H, W = strips[0][1].shape[:2]
    cell_h = (fig_w - lab_w) * H / W
    title_h = 0.24 if title else 0.0
    sep_h = 0.1 if separator_after else 0.0
    fig_h = len(strips) * cell_h + title_h + sep_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    y = fig_h
    if title:
        y -= title_h
        fig.text(lab_w / fig_w, (y + 0.05) / fig_h, title, fontsize=8.5, ha="left", va="bottom", style="italic")
    for i, (name, img) in enumerate(strips):
        if separator_after and i == separator_after:
            y -= sep_h
            fig.add_artist(plt.Line2D([0.02, 0.98], [(y + sep_h / 2) / fig_h] * 2, color="#bbbbbb", linewidth=0.6))
        y -= cell_h
        ax = fig.add_axes([lab_w / fig_w, y / fig_h, (fig_w - lab_w) / fig_w, cell_h / fig_h])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        bold = name.startswith("TMR")
        fig.text((lab_w - 0.06) / fig_w, (y + cell_h / 2) / fig_h, name, fontsize=8.5, ha="right", va="center",
                 weight="bold" if bold else "normal", color="#1a5fb4" if bold else "black")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({fig_w:.1f} x {fig_h:.2f} in)")


if __name__ == "__main__":
    main()
