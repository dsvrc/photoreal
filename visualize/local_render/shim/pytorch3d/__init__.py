"""Minimal stand-in for the parts of pytorch3d that visualize/ca_body uses (forward rendering only), for machines
where pytorch3d cannot be installed. Only put on sys.path by visualize/local_render/render_qual.py.

Conventions follow pytorch3d: NDC +x left, +y up, pixel (i, j) center at
x_ndc = (1 - (2j + 1) / W) * W / S, y_ndc = (1 - (2i + 1) / H) * H / S with S = min(H, W); OpenCV cameras
project with u = fx X / Z + cx, v = fy Y / Z + cy; TexturesUV samples bilinearly with align_corners=True and
border padding, the texture's first row at v = 1 (pytorch3d flips the map's y axis before sampling).
"""
