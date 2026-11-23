"""Z-buffer triangle rasterizer (numba, CPU) with perspective-correct barycentrics.

Pixel (row i, column j) has its center at continuous coordinates (u, v) = (j + 0.5, i + 0.5).
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _raster(uvz, faces, H, W, zbuf, p2f, bary):
    for f in range(faces.shape[0]):
        a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
        x0, y0, z0 = uvz[a, 0], uvz[a, 1], uvz[a, 2]
        x1, y1, z1 = uvz[b, 0], uvz[b, 1], uvz[b, 2]
        x2, y2, z2 = uvz[c, 0], uvz[c, 1], uvz[c, 2]
        if z0 <= 1e-6 or z1 <= 1e-6 or z2 <= 1e-6:
            continue
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        minx = max(int(np.floor(min(x0, min(x1, x2)) - 0.5)), 0)
        maxx = min(int(np.ceil(max(x0, max(x1, x2)) - 0.5)), W - 1)
        miny = max(int(np.floor(min(y0, min(y1, y2)) - 0.5)), 0)
        maxy = min(int(np.ceil(max(y0, max(y1, y2)) - 0.5)), H - 1)
        for py in range(miny, maxy + 1):
            cy = py + 0.5
            for px in range(minx, maxx + 1):
                cx = px + 0.5
                w0 = ((x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)) / area
                w1 = ((x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                q0, q1, q2 = w0 / z0, w1 / z1, w2 / z2
                s = q0 + q1 + q2
                z = 1.0 / s
                if z < zbuf[py, px]:
                    zbuf[py, px] = z
                    p2f[py, px] = f
                    bary[py, px, 0] = q0 / s
                    bary[py, px, 1] = q1 / s
                    bary[py, px, 2] = q2 / s


def rasterize(uvz: np.ndarray, faces: np.ndarray, H: int, W: int):
    """uvz [V x 3] (pixel u, pixel v, depth > 0), faces [F x 3] -> pix_to_face [H x W] (-1 empty),
    bary [H x W x 3], zbuf [H x W] (inf empty)."""
    uvz = np.ascontiguousarray(uvz, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    zbuf = np.full((H, W), np.inf)
    p2f = np.full((H, W), -1, dtype=np.int64)
    bary = np.zeros((H, W, 3))
    _raster(uvz, faces, H, W, zbuf, p2f, bary)
    return p2f, bary, zbuf
