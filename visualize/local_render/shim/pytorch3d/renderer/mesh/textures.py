class TexturesUV:
    def __init__(self, maps, faces_uvs, verts_uvs, align_corners=True, padding_mode="border", sampling_mode="bilinear"):
        self.maps = maps  # B x Ht x Wt x C
        self.faces_uvs = faces_uvs  # B x F x 3
        self.verts_uvs = verts_uvs  # B x Vt x 2
