"""Replay candidates: fixed training-length windows that tile every training scene of a subject.

Each scene is cut into back-to-back windows of manifest["window"] frames (600 = 20 s, the
training length), plus one end-aligned window when at least half a window is left over.
A session's buffer keeps round(ratio x candidates) of them.
"""

from typing import Dict, List, Tuple


def tile_starts(length: int, window: int) -> List[int]:
    starts = list(range(0, length - window + 1, window))
    if starts and length - (starts[-1] + window) >= window // 2:
        starts.append(length - window)
    return starts


def session_clips(manifest: Dict, subject: str) -> List[Tuple[str, int]]:
    """(scene, start) of every candidate window of the subject's training scenes."""
    entry = manifest["subjects"][subject]
    window = manifest["window"]
    return [(scene, start) for scene in entry["scenes"]["train"] for start in tile_starts(entry["frames"][scene], window)]


def replay_budget(n_candidates: int, ratio: float) -> int:
    return max(1, int(round(ratio * n_candidates)))
