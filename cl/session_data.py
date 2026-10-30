"""New-session training data: the repo's Social dataset, normalized once instead of per sample.

Social.__getitem__ z-normalizes a subject's whole scene (minutes of 48 kHz stereo audio) for
every sample and then keeps a 20 s crop, which makes data loading the bottleneck. Here each
scene is normalized once at start-up; samples are then only cropped and zero-padded, exactly
as before (normalization is element-wise and happens before padding in both cases).
"""

import numpy as np
import torch

from data_loaders.data import Social


class PreNormalizedSocial(Social):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.data_format == "pose" and self.split == "train" and not self.chunk
        for i in range(len(self.data)):
            self.data[i] = (np.asarray(self.data[i]) - self.mean) / self.std
            audio = self.audio[i].numpy() if torch.is_tensor(self.audio[i]) else np.asarray(self.audio[i])
            self.audio[i] = ((audio - self.audio_mean) / self.audio_std).astype(np.float32)

    def __getitem__(self, item):
        item = self._get_idx(item)
        motion = self.data[item]
        audio = self.audio[item]
        keyframes = motion[:: self.step]
        data_dict = {
            "motion": motion,
            "m_length": self.lengths[item],
            "audio": audio,
            "a_length": len(audio),
            "keyframes": keyframes,
            "k_length": len(keyframes),
            "missing": self.missing[item],
        }
        return self._get_random_subsection(data_dict)
