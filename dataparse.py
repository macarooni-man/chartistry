from __future__ import annotations

from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import torchaudio.functional as AF
import torch.nn.functional as F
import os, glob
import numpy as np
import torch
import torchaudio
import librosa  # still available if you keep logmel() for utilities

from chartlib import Chart, normalize_pro_drums_tick

# ───── Audio/feature params (simple, sensible defaults) ─────
sample_rate = 16000
n_mels = 64
window_sec = 0.64
# sample_rate = 22050        # sample rate for features
# n_mels      = 96           # mel bands
# window_sec  = 1.0          # audio window centered on each grid tick (seconds)
win_len_ms  = 23            # spectrogram frame size (ms)
hop_ms      = 10            # spectrogram hop (ms)

_AUDIO_EXTS = (".wav", ".ogg", ".mp3", ".flac", ".m4a")

def _nearest_pow2(n: int) -> int:
    return 1 << (int(n - 1).bit_length())

def mel_config():
    """
    Single source of truth for mel parameters.
    Trainer will import this so we never diverge.
    """
    n_fft_est = int(round(sample_rate * win_len_ms / 1000))
    n_fft = _nearest_pow2(n_fft_est)                   # e.g., 507 -> 512
    return {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": int(sample_rate * hop_ms / 1000),
        "n_mels": n_mels,
        "f_min": 20.0,                                 # avoids zeroed low bands
        "f_max": None,                                 # None => sr/2
        "power": 2.0,
        "norm": "slaney",
        "mel_scale": "htk",
    }

def _ms_to_samples(ms: float) -> int:
    return int(round(ms * sample_rate / 1000.0))

def logmel(y: np.ndarray) -> np.ndarray:
    """
    Log-mel spectrogram (a "picture of sound": time × frequency with log energy).
    (Utility; not used in the fast training path.)
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sample_rate,
        n_fft=int(sample_rate * win_len_ms / 1000),
        hop_length=int(sample_rate * hop_ms / 1000),
        n_mels=n_mels, power=2.0,
    )
    S = np.log1p(S).astype(np.float32)  # log(1 + power) for stability
    return S

def label_vector_from_notes_at_tick(notes_at_tick) -> np.ndarray:
    """
    (Legacy utility.) Convert raw notes at a tick into an 8-dim multi-label vector:
    [K, R, Y, B, G, Yc, Bc, Gc]
    """
    d = normalize_pro_drums_tick(notes_at_tick)
    return np.array([
        1.0 if d["kick"]     else 0.0,
        1.0 if d["red"]      else 0.0,
        1.0 if d["yellow"]   else 0.0,
        1.0 if d["blue"]     else 0.0,
        1.0 if d["green"]    else 0.0,
        1.0 if d["cymbal_y"] else 0.0,
        1.0 if d["cymbal_b"] else 0.0,
        1.0 if d["cymbal_g"] else 0.0,
    ], dtype=np.float32)

# Fast label builder (used in precompute)
def _label_from_lanes(lanes: set[int]) -> torch.Tensor:
    # [K, R, Y, B, G, Yc, Bc, Gc]
    return torch.tensor([
        1.0 if 0 in lanes  else 0.0,
        1.0 if 1 in lanes  else 0.0,
        1.0 if 2 in lanes  else 0.0,
        1.0 if 3 in lanes  else 0.0,
        1.0 if 4 in lanes  else 0.0,
        1.0 if 66 in lanes else 0.0,
        1.0 if 67 in lanes else 0.0,
        1.0 if 68 in lanes else 0.0,
    ], dtype=torch.float32)

# ───────────────────────── robust discovery ─────────────────────────

def _find_chart(dirpath: str) -> Optional[str]:
    hits = glob.glob(os.path.join(dirpath, "*.chart"))
    return hits[0] if hits else None

def _find_audio(dirpath: str) -> Optional[str]:
    # prefer drums.*, then mix.*, then audio.*, else any audio file in folder
    for base in ("drums", "mix", "audio"):
        for ext in _AUDIO_EXTS:
            for variant in (base, base.upper(), base.capitalize()):
                cand = os.path.join(dirpath, variant + ext)
                if os.path.exists(cand):
                    return cand
    for ext in _AUDIO_EXTS:
        hits = glob.glob(os.path.join(dirpath, "*" + ext))
        if hits:
            return hits[0]
    return None

def discover_song_dirs(root: str, recursive: bool = False) -> list[str]:
    """
    Find subfolders that contain both a .chart (any name) and an audio file.
    Names can have spaces; we don't care about folder naming.
    """
    songs: list[str] = []
    if not os.path.isdir(root):
        return songs

    dirs: list[str]
    if recursive:
        dirs = []
        for dp, subdirs, _ in os.walk(root):
            for d in subdirs:
                dirs.append(os.path.join(dp, d))
    else:
        dirs = [os.path.join(root, name) for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))]

    for d in dirs:
        if _find_chart(d) and _find_audio(d):
            songs.append(d)

    return sorted(songs)

# ───────────────────────── fast dataset (flat index + LRU audio + precomputed labels) ─────────────────────────

class _LRUAudioCache(OrderedDict):
    def __init__(self, max_items: int = 2):
        super().__init__(); self.max_items = max_items

    def get_audio(self, path: str) -> torch.Tensor:
        """
        Return a 1D torch.float32 tensor at target sample_rate.
        Cached as a torch tensor to avoid numpy round-trips.
        """
        if path in self:
            y = super().pop(path)
            super().__setitem__(path, y)
            return y

        wav, sr = torchaudio.load(path)          # [C, N], torch.float32
        y = wav.mean(dim=0)                      # [N] mono

        if sr != sample_rate:
            # fast, vectorized resample in torch
            y = AF.resample(y, orig_freq=sr, new_freq=sample_rate)

        y = y.contiguous()                       # slicing friendly
        super().__setitem__(path, y)
        if len(self) > self.max_items:
            self.popitem(last=False)
        return y

class DrumGridDataset(torch.utils.data.Dataset):
    """
    One item corresponds to one grid tick (default 48th notes):
      X: raw mono waveform window centered on the grid tick → torch.FloatTensor [1, T]
      y: 8-dim multi-label vector                           → torch.FloatTensor [8]

    Expected folder layout per song:
      <song_dir>/notes.chart (or any *.chart)
      <song_dir>/drums.wav (or mix/audio with common extensions)
    """
    def __init__(self,
                 song_dirs: list[str] | None = None, *,
                 root: str | None = None,
                 subdivision: int = 48,
                 track_hint: str = "ExpertDrums",
                 limit_songs: Optional[int] = None,
                 limit_ticks: Optional[int] = None,
                 debug: bool = False):
        self.subdivision = subdivision
        self.track_hint  = track_hint
        self.debug       = debug

        # select songs
        if song_dirs is None:
            if root is None:
                raise ValueError("Provide either song_dirs or root.")
            song_dirs = discover_song_dirs(root)
        if limit_songs is not None:
            song_dirs = song_dirs[:limit_songs]

        # per-song state
        self._chart: Dict[str, Chart]      = {}
        self._audio_path: Dict[str, str]   = {}
        self._grid_ticks: Dict[str, List[int]] = {}
        self._labels: Dict[str, List[torch.Tensor]] = {}
        self._tick_pos: Dict[str, Dict[int, int]] = {}       # tick -> local index

        # global flat index: list of (song_dir, tick)
        self.index: List[Tuple[str, int]] = []

        # small audio cache (avoid re-reading/resampling every item)
        self._cache = _LRUAudioCache(max_items=2)

        total_ticks = 0
        for d in song_dirs:
            # chart path: pick first *.chart
            chart_path = _find_chart(d)
            if chart_path is None:
                if self.debug: print(f"[warn] no chart in {d}")
                continue
            ch = Chart(chart_path)
            self._chart[d] = ch

            # audio path
            ap = _find_audio(d)
            if ap is None:
                if self.debug: print(f"[warn] no audio in {d}")
                continue
            self._audio_path[d] = ap

            # grid ticks
            ticks = ch.build_grid_ticks(subdivision=self.subdivision, track_hint=self.track_hint)
            if limit_ticks is not None:
                ticks = ticks[:limit_ticks]
            self._grid_ticks[d] = ticks
            self._tick_pos[d]   = {t: i for i, t in enumerate(ticks)}

            # ----- PRECOMPUTE LABELS FOR THESE TICKS (O(n) once) -----
            step = ch.resolution // round(self.subdivision / 4)
            tol_ticks = max(1, step // 2)

            note_list = []
            if self.track_hint in ch.tracks:
                note_list = sorted(ch.tracks[self.track_hint].notes, key=lambda n: n.tick)
            note_ticks = [n.tick for n in note_list]

            import bisect as _bisect
            labels: List[torch.Tensor] = []
            for t in ticks:
                left  = _bisect.bisect_left(note_ticks, t - tol_ticks)
                right = _bisect.bisect_right(note_ticks, t + tol_ticks)
                lanes: set[int] = set()
                for i in range(left, right):
                    lanes.add(note_list[i].lane)
                labels.append(_label_from_lanes(lanes))
            self._labels[d] = labels
            # ---------------------------------------------------------

            # extend flat index
            self.index.extend((d, t) for t in ticks)
            total_ticks += len(ticks)

        if self.debug:
            print(f"[dataset] songs={len(self._chart)} ticks={total_ticks:,} (subdiv={self.subdivision})")

    def __len__(self) -> int:
        return len(self.index)

    # internal helpers
    def _audio_for(self, song_dir: str) -> torch.Tensor:
        return self._cache.get_audio(self._audio_path[song_dir])

    def __getitem__(self, index: int):
        song_dir, tick = self.index[index]
        ch = self._chart[song_dir]
        y  = self._audio_for(song_dir)  # torch.Tensor [N]

        # Center time in ms
        center_ms = ch.tick_to_ms(tick)
        half_ms = (window_sec * 1000.0) / 2.0
        start_ms = max(0.0, center_ms - half_ms)
        end_ms   = min(y.numel() * 1000.0 / sample_rate, center_ms + half_ms)

        s = int(round(start_ms * sample_rate / 1000.0))
        e = int(round(end_ms   * sample_rate / 1000.0))
        y_win = y[s:e]  # torch slice

        # Pad to fixed length with zeros (right pad)
        need = int(round(2 * half_ms * sample_rate / 1000.0)) - y_win.numel()
        if need > 0:
            y_win = F.pad(y_win, (0, need))

        x = y_win.unsqueeze(0)  # [1, T] torch.float32

        # O(1) label fetch aligned with tick list
        li = self._tick_pos[song_dir][tick]
        y_vec = self._labels[song_dir][li]       # torch.FloatTensor [8]

        return x, y_vec
