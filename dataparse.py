from __future__ import annotations

from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
import os, glob
import numpy as np
import torch
import torchaudio
import librosa

from chartlib import Chart, normalize_pro_drums_tick

# ───── Audio/feature params (simple, sensible defaults) ─────
sample_rate = 16000
n_mels = 64
window_sec = 0.64
# sample_rate = 22050        # sample rate for features
# n_mels      = 96           # mel bands
# window_sec  = 1.0          # audio window centered on each grid tick (seconds)
win_len_ms  = 23           # spectrogram frame size (ms)
hop_ms      = 10           # spectrogram hop (ms)

_AUDIO_EXTS = (".wav", ".ogg", ".mp3", ".flac", ".m4a")

def _ms_to_samples(ms: float) -> int:
    return int(round(ms * sample_rate / 1000.0))

def logmel(y: np.ndarray) -> np.ndarray:
    """
    Log-mel spectrogram (a "picture of sound": time × frequency with log energy).
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
    Convert raw notes at a tick into an 8-dim multi-label vector:
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

# ───────────────────────── fast dataset (flat index + LRU audio) ─────────────────────────

class _LRUAudioCache(OrderedDict):
    def __init__(self, max_items: int = 2):
        super().__init__(); self.max_items = max_items
    def get_audio(self, path: str) -> np.ndarray:
        if path in self:
            y = super().pop(path)        # move to end (most-recent)
            super().__setitem__(path, y)
            return y
        wav, sr = torchaudio.load(path)  # [C, N]
        y = wav.mean(dim=0).numpy()
        if sr != sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        super().__setitem__(path, y)
        if len(self) > self.max_items:
            self.popitem(last=False)     # evict least-recent
        return y

class DrumGridDataset(torch.utils.data.Dataset):
    """
    One item corresponds to one grid tick (default 48th notes):
      X: log-mel spectrogram for a 1.0s window around the tick  → torch.FloatTensor [1, N_MELS, T]
      y: 8-dim multi-label vector                               → torch.FloatTensor [8]

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

        # per-song data
        self._chart: Dict[str, Chart] = {}
        self._audio_path: Dict[str, str] = {}
        self._grid_ticks: Dict[str, List[int]] = {}

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

            # extend flat index
            self.index.extend((d, t) for t in ticks)
            total_ticks += len(ticks)

        if self.debug:
            print(f"[dataset] songs={len(self._chart)} ticks={total_ticks:,} (subdiv={self.subdivision})")

    def __len__(self) -> int:
        return len(self.index)

    # internal helpers
    def _audio_for(self, song_dir: str) -> np.ndarray:
        return self._cache.get_audio(self._audio_path[song_dir])

    def __getitem__(self, index: int):
        song_dir, tick = self.index[index]
        ch = self._chart[song_dir]
        y  = self._audio_for(song_dir)

        # Center time (ms) at this grid tick using the chart's internal TempoMap
        center_ms = ch.tick_to_ms(tick)

        # Extract a centered window
        half_ms = (window_sec * 1000.0) / 2.0
        start_ms = max(0.0, center_ms - half_ms)
        end_ms   = min(len(y) * 1000.0 / sample_rate, center_ms + half_ms)

        s = _ms_to_samples(start_ms)
        e = _ms_to_samples(end_ms)
        y_win = y[s:e]

        # Pad to fixed length (begin/end of file)
        need = _ms_to_samples(2 * half_ms) - len(y_win)
        if need > 0:
            y_win = np.pad(y_win, (0, need))

        # Features
        S = logmel(y_win)        # [N_MELS, T]
        S = S[None, ...]         # [1, N_MELS, T] — single "channel"

        # Labels: any note within ± half a grid step counts for this grid cell
        step = ch.resolution // round(self.subdivision / 4)  # your build_grid_ticks convention
        tol_ticks = max(1, step // 2)

        # Collect nearby notes from the hinted track (if present); else empty
        notes = []
        if self.track_hint in ch.tracks:
            # (iterate local list is fine — per-song, not per-item scanning anymore)
            for n in ch.tracks[self.track_hint].notes:
                if abs(n.tick - tick) <= tol_ticks:
                    notes.append(n)

        y_vec = label_vector_from_notes_at_tick(notes)  # [8]
        return torch.from_numpy(S), torch.from_numpy(y_vec)
