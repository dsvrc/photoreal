from dataclasses import dataclass

import torch


@dataclass
class OpenCVCameras:
    R: torch.Tensor  # B x 3 x 3
    T: torch.Tensor  # B x 3
    K: torch.Tensor  # B x 3 x 3
    image_size: torch.Tensor  # B x 2 (H, W)


def cameras_from_opencv_projection(R, tvec, camera_matrix, image_size):
    return OpenCVCameras(R=R, T=tvec, K=camera_matrix, image_size=image_size)
