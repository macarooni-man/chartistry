from __future__ import annotations
import os, glob, torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset

def _find_precomp(dirpath: str) -> Optional[str]:
    cand = os.path.join(dirpath, "chart_feats.pt")
    return cand if os.path.exists(cand) else None

def discover_song_dirs(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    dirs = [os.path.join(root, n) for n in os.listdir(root)
            if os.path.isdir(os.path.join(root, n))]
    out = []
    for d in dirs:
        if _find_precomp(d):
            out.append(d)
    return sorted(out)

class DrumGridDataset(Dataset):
    """
    Trains from precomputed mel windows and labels.
    Each item:
      X: [1, n_mels, W]  (W = window_frames)
      y: [8]
    """
    def __init__(self, song_dirs: list[str]):
        self.song_dirs = song_dirs
        self.data: Dict[str, dict] = {}
        self.index: List[Tuple[str, int]] = []  # (song_dir, local_tick_idx)

        total = 0
        for d in song_dirs:
            feat_path = _find_precomp(d)
            if not feat_path:
                continue
            pack = torch.load(feat_path, map_location="cpu")
            self.data[d] = pack

            n = int(pack["centers_frames"].numel())
            self.index.extend((d, i) for i in range(n))
            total += n

        # cache per song for speed
        self._cache_X: Dict[str, torch.Tensor] = {}
        print(f"[precomp dataset] songs={len(self.data)} ticks={total:,}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        song_dir, i = self.index[idx]
        pack = self.data[song_dir]

        S = pack["mel"]                     # [n_mels, F]
        centers = pack["centers_frames"]    # [N]
        W = int(pack["window_frames"])
        half = W // 2

        c = int(centers[i])
        s = max(0, c - half)
        e = min(S.size(-1), c + half)

        x = S[:, s:e]                       # [n_mels, <=W]
        need = W - x.size(-1)
        if need > 0:
            x = F.pad(x, (0, need))         # right pad to W

        x = x.unsqueeze(0).contiguous()     # [1, n_mels, W]
        y = pack["labels"][i]               # [8]
        return x, y
