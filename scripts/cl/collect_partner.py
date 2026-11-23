"""
Collect the conversation partner of the qualitative chunks (cl_data/qual_pack, scripts/cl/collect_videos.py): the
partner's recorded body and face in the same scene and frames, and the partner's photoreal renderer files. CPU only,
from the repo root in a2p_env:

    python scripts/cl/collect_partner.py

The released loader pairs PXB184 with RLW104 and TXB805 with GQS883 through the same scene file names
(data_loaders/get_data.py, flip_person), and channel 1 of a participant's audio is the partner's microphone. The
script checks both: the partner's scene has the same number of frames, and the participant's channel 1 correlates with
the partner's channel 0 over the chunk.

Writes cl_data/partner_pack.tar.gz:
    partner.npz             bodies [chunks x 600 x 104] (raw joint angles, the loader's pose fix applied),
                            faces [chunks x 600 x 256], chunks, scenes, starts, audio correlation, partner, subject
    ca_body/<partner>/      config.yml, body_dec.ckpt, static_assets.pt (about 1.2 GB); --no_renderer to skip
"""

import argparse
import json
import os
import shutil
import sys
import tarfile

import numpy as np

PARTNER = {"PXB184": "RLW104", "RLW104": "PXB184", "TXB805": "GQS883", "GQS883": "TXB805"}
WRAP_SUBJECTS = ("PXB184", "RLW104")  # same pose fix as data_loaders/get_data.py
AUDIO_PER_FRAME = 1600
RENDERER_FILES = ("config.yml", "body_dec.ckpt", "static_assets.pt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", default="cl_data/qual_pack/rows.npz", help="the qualitative pack: subject and chunks")
    p.add_argument("--subject", default=None, help="default: from --rows")
    p.add_argument("--chunks", type=int, nargs="*", default=None, help="default: from --rows")
    p.add_argument("--manifest", default="cl_data/manifest.json")
    p.add_argument("--dataset_root", default="dataset")
    p.add_argument("--renderer_root", default="checkpoints/ca_body/data")
    p.add_argument("--no_renderer", action="store_true")
    p.add_argument("--out", default="cl_data/partner_pack")
    return p.parse_args()


def main():
    a = parse_args()
    import torchaudio

    subject, chunks = a.subject, a.chunks
    if subject is None or chunks is None:
        d = np.load(a.rows)
        subject = subject or str(d["subject"])
        chunks = chunks if chunks is not None else [int(c) for c in d["chunks"]]
    partner = PARTNER[subject]
    with open(a.manifest) as f:
        manifest = json.load(f)
    window = manifest["window"]
    test = manifest["subjects"][subject]["test_chunks"]
    sroot, proot = os.path.join(a.dataset_root, subject), os.path.join(a.dataset_root, partner)

    bodies, faces, scenes, starts, corrs = [], [], [], [], []
    for c in chunks:
        scene, start = test[c][0], int(test[c][1])
        pb = os.path.join(proot, scene + "_body_pose.npy")
        if not os.path.exists(pb):
            sys.exit(f"{partner} has no {scene}: {pb}")
        n_s = len(np.load(os.path.join(sroot, scene + "_body_pose.npy"), mmap_mode="r"))
        n_p = len(np.load(pb, mmap_mode="r"))
        body = np.array(np.load(pb, mmap_mode="r")[start : start + window], dtype=np.float32)
        if partner in WRAP_SUBJECTS:
            body[:, 3] = (body[:, 3] + np.pi) % (2 * np.pi)
            body[:, 3] = (body[:, 3] + np.pi) % (2 * np.pi)
        face = np.asarray(np.load(os.path.join(proot, scene + "_face_expression.npy"), mmap_mode="r")
                          [start : start + window], dtype=np.float32)
        sa, _ = torchaudio.load(os.path.join(sroot, scene + "_audio.wav"), frame_offset=start * AUDIO_PER_FRAME,
                                num_frames=window * AUDIO_PER_FRAME)
        pa, _ = torchaudio.load(os.path.join(proot, scene + "_audio.wav"), frame_offset=start * AUDIO_PER_FRAME,
                                num_frames=window * AUDIO_PER_FRAME)
        r = float(np.corrcoef(sa[1].numpy(), pa[0].numpy())[0, 1])
        print(f"chunk {c}: {scene} frames {start}-{start + window} | {subject} {n_s} frames, {partner} {n_p} frames"
              f" | corr({subject} ch1, {partner} ch0) = {r:.3f}")
        if n_s != n_p:
            print(f"  WARNING: the two scenes have different lengths")
        bodies.append(body)
        faces.append(face)
        scenes.append(scene)
        starts.append(start)
        corrs.append(r)

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    np.savez_compressed(os.path.join(a.out, "partner.npz"), bodies=np.stack(bodies), faces=np.stack(faces),
                        chunks=np.array(chunks), scenes=np.array(scenes), starts=np.array(starts),
                        audio_corr=np.array(corrs), partner=partner, subject=subject)
    if not a.no_renderer:
        dst = os.path.join(a.out, "ca_body", partner)
        os.makedirs(dst)
        for name in RENDERER_FILES:
            shutil.copy2(os.path.join(a.renderer_root, partner, name), dst)
            print(f"renderer {name}: {os.path.getsize(os.path.join(dst, name)) / 2**20:.0f} MB")
    tar = a.out + ".tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(a.out, arcname=os.path.basename(a.out))
    print(f"\nwritten {tar} ({os.path.getsize(tar) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
