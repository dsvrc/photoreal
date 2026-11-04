"""New-session training data: the repo's Social dataset, normalized once instead of per sample.

Social.__getitem__ z-normalizes a subject's whole scene (minutes of 48 kHz stereo audio) for
every sample and then keeps a 20 s crop, which makes data loading the bottleneck. Here each
scene is normalized once at start-up; samples are then only cropped and zero-padded, exactly
as before (normalization is element-wise and happens before padding in both cases).

SessionCrops goes further: it keeps the normalized scenes on the GPU and cuts every batch
there, so the ~250 MB of audio in a batch never goes through loader workers and host memory.
"""

import bisect

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


def scene_of(item: int, cum_frames: np.ndarray) -> int:
    """Social._get_idx: the first scene whose cumulative length reaches item (item 0 gives the last scene)."""
    return int(np.searchsorted(cum_frames, item, side="left")) if item > 0 else len(cum_frames) - 1


class SessionCrops:
    """The batches of DataLoader(PreNormalizedSocial datasets, shuffle=True, drop_last=True,
    collate_fn=social_collate), cut on `device` from the scenes kept there.

    Same sampling as that loader: all frame indices of the (concatenated) datasets in a new random
    order every pass, each mapped to its scene as Social._get_idx does (scenes come up in proportion
    to their length), then Social._get_random_subsection's random start and length (400-599 frames)
    and zero padding to max_seq_length. Only the random stream differs: one generator here instead
    of one per loader worker.
    """

    def __init__(self, datasets, batch_size: int, device, seed):
        ds = datasets[0]
        self.batch_size, self.device = batch_size, torch.device(device)
        self.window, self.min_length, self.step = ds.max_seq_length, ds.min_seq_length, ds.step
        self.add_padding, self.audio_per_frame = ds.add_padding, ds.audio_per_frame
        self.rng = np.random.default_rng(seed)
        self.dataset_ends = np.cumsum([len(d) for d in datasets])  # ConcatDataset.cumulative_sizes
        self.scene_ends, self.first_scene, self.scenes = [], [], []
        for d in datasets:
            assert d.data_format == "pose" and all(np.all(m == 1) for m in d.missing), "pose data has no missing frames"
            self.scene_ends.append(np.cumsum([len(m) for m in d.data]))  # what Social._get_idx walks
            self.first_scene.append(len(self.scenes))
            for motion, audio, length in zip(d.data, d.audio, d.lengths):
                motion = torch.as_tensor(np.asarray(motion), dtype=torch.float32).to(self.device)
                self.scenes.append((motion, torch.as_tensor(audio).to(self.device), int(length)))
        self.order, self.pos = np.zeros(0, dtype=np.int64), 0
        self.nbytes = sum(t.numel() * t.element_size() for m, a, _ in self.scenes for t in (m, a))

    def scene(self, index: int) -> int:
        """Scene (over all datasets) of a ConcatDataset index."""
        d = bisect.bisect_right(self.dataset_ends, index)
        local = index - (self.dataset_ends[d - 1] if d > 0 else 0)
        return self.first_scene[d] + scene_of(local, self.scene_ends[d])

    def draw(self):
        """(scene, start, length) of the next batch."""
        if self.pos + self.batch_size > len(self.order):  # a new pass; drop_last drops the short batch
            self.order, self.pos = self.rng.permutation(int(self.dataset_ends[-1])), 0
        picks = []
        for index in self.order[self.pos : self.pos + self.batch_size]:
            s = self.scene(int(index))
            start = int(self.rng.integers(0, self.scenes[s][2] - self.window))
            length = int(self.rng.integers(self.min_length, self.window)) if self.add_padding else self.window
            picks.append((s, start, length))
        self.pos += self.batch_size
        return picks

    def cut(self, picks):
        """social_collate's (motion, cond) for these crops, on the device."""
        n, w, a = len(picks), self.window, self.audio_per_frame
        dims, channels = self.scenes[0][0].shape[1], self.scenes[0][1].shape[1]
        motion = torch.zeros(n, w, dims, device=self.device)
        keyframes = torch.zeros(n, len(range(0, w, self.step)), dims, device=self.device)
        audio = torch.zeros(n, w * a, channels, device=self.device)
        for i, (s, start, length) in enumerate(picks):
            crop = self.scenes[s][0][start : start + length]
            motion[i, :length] = crop
            keyframes[i, : len(range(0, length, self.step))] = crop[:: self.step]
            audio[i, : length * a] = self.scenes[s][1][start * a : (start + length) * a]
        lengths = torch.tensor([length for _, _, length in picks], device=self.device)
        valid = torch.arange(w, device=self.device) < lengths[:, None]
        y = {
            "missing": valid[:, :, None].expand(n, w, dims).float(),
            "mask": valid[:, None, None, :],
            "lengths": lengths,
            "audio": audio,
            "alengths": lengths * a,
            "keyframes": keyframes,
            "klengths": (lengths + self.step - 1) // self.step,
        }
        return motion.permute(0, 2, 1).unsqueeze(2).contiguous(), {"y": y}

    def __iter__(self):
        return self

    def __next__(self):
        return self.cut(self.draw())
