import torch
import torch.nn.functional as F


class Meshes:
    def __init__(self, verts, faces, textures=None):
        self.verts = verts  # [B x V x 3]
        self.faces = faces  # [B x F x 3]
        self.textures = textures

    def sample_textures(self, fragments):
        """[B x H x W x 1 x C] texels at the rasterized pixels, as pytorch3d's TexturesUV does."""
        t = self.textures
        B, H, W, _ = fragments.pix_to_face.shape
        out = []
        for n in range(B):
            p2f = fragments.pix_to_face[n, ..., 0]
            bary = fragments.bary_coords[n, ..., 0, :]  # H x W x 3
            f = p2f.clamp(min=0)
            uv_idx = t.faces_uvs[n][f]  # H x W x 3
            uvs = t.verts_uvs[n][uv_idx]  # H x W x 3 x 2
            puv = (bary[..., None] * uvs).sum(dim=2)  # H x W x 2
            grid = (puv * 2.0 - 1.0)[None]
            tex = t.maps[n].permute(2, 0, 1)[None].flip(2)  # 1 x C x Ht x Wt, y flipped as in pytorch3d
            texel = F.grid_sample(tex, grid.to(tex.dtype), mode="bilinear", align_corners=True,
                                  padding_mode="border")[0].permute(1, 2, 0)
            out.append(texel)
        return torch.stack(out)[:, :, :, None, :]
