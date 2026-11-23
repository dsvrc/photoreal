import torch


def _angle_from_tan(axis, other_axis, data, horizontal, tait_bryan):
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix, convention):
    i0, i2 = "XYZ".index(convention[0]), "XYZ".index(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central = torch.asin(matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        central = torch.acos(matrix[..., i0, i0])
    return torch.stack(
        (
            _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
            central,
            _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
        ),
        -1,
    )
