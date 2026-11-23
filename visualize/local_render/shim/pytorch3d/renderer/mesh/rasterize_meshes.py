import numpy as np
import torch

from raster import rasterize


def rasterize_meshes(meshes, image_size, faces_per_pixel=1, z_clip_value=None, bin_size=None, **kwargs):
    """pytorch3d.renderer.mesh.rasterize_meshes for verts already in NDC (x, y) with depth z: returns
    pix_to_face [B x H x W x 1], zbuf, bary [B x H x W x 1 x 3], dists (None)."""
    H, W = (image_size, image_size) if isinstance(image_size, int) else image_size
    S = min(H, W)
    p2fs, zs, bs = [], [], []
    for n in range(meshes.verts.shape[0]):
        V = meshes.verts[n].detach().double().cpu().numpy()
        u = (W - V[:, 0] * S) / 2.0
        v = (H - V[:, 1] * S) / 2.0
        p2f, bary, zbuf = rasterize(np.stack([u, v, V[:, 2]], 1), meshes.faces[n].detach().cpu().numpy(), H, W)
        p2fs.append(p2f)
        zs.append(np.where(np.isinf(zbuf), -1.0, zbuf))
        bs.append(bary)
    dev = meshes.verts.device
    return (
        torch.from_numpy(np.stack(p2fs))[..., None].to(dev),
        torch.from_numpy(np.stack(zs))[..., None].float().to(dev),
        torch.from_numpy(np.stack(bs))[..., None, :].float().to(dev),
        None,
    )
