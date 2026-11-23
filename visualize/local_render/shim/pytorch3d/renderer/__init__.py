from dataclasses import dataclass

import numpy as np
import torch

from raster import rasterize


@dataclass
class RasterizationSettings:
    image_size: tuple = (256, 256)
    blur_radius: float = 0.0
    faces_per_pixel: int = 1


@dataclass
class Fragments:
    pix_to_face: torch.Tensor  # B x H x W x 1
    zbuf: torch.Tensor  # B x H x W x 1
    bary_coords: torch.Tensor  # B x H x W x 1 x 3
    dists: torch.Tensor = None


class MeshRasterizer:
    def __init__(self, raster_settings=None, cameras=None):
        self.raster_settings = raster_settings or RasterizationSettings()

    def __call__(self, meshes, cameras):
        H, W = self.raster_settings.image_size
        B = meshes.verts.shape[0]
        p2fs, bs, zs = [], [], []
        for n in range(B):
            X = meshes.verts[n].detach().double().cpu().numpy()
            R, t, K = (x[n].detach().double().cpu().numpy() for x in (cameras.R, cameras.T, cameras.K))
            Xc = X @ R.T + t
            z = Xc[:, 2]
            zs_safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
            u = K[0, 0] * Xc[:, 0] / zs_safe + K[0, 1] * Xc[:, 1] / zs_safe + K[0, 2]
            v = K[1, 1] * Xc[:, 1] / zs_safe + K[1, 2]
            faces = meshes.faces[n].detach().cpu().numpy()
            p2f, bary, zbuf = rasterize(np.stack([u, v, z], 1), faces, int(H), int(W))
            p2fs.append(p2f)
            bs.append(bary)
            zs.append(zbuf)
        dev = meshes.verts.device
        return Fragments(
            pix_to_face=torch.from_numpy(np.stack(p2fs))[..., None].to(dev),
            zbuf=torch.from_numpy(np.where(np.isinf(np.stack(zs)), -1.0, np.stack(zs)))[..., None].float().to(dev),
            bary_coords=torch.from_numpy(np.stack(bs))[..., None, :].float().to(dev),
        )
