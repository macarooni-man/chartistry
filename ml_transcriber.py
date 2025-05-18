import torch
import librosa
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict
from dataclasses import dataclass
from chart_parser import Chart

@dataclass
class Note:
    time: float
    tick: int
    category: str

@dataclass
class Transcription:
    type: str
    notes: List[List[Note]]

def _build_timing_map(chart: Chart):
    # Returns list of (tick_i, time_i, bpm_i)
    bpms = sorted(
        [(e.tick, e.values[0] / 1000.0) for e in chart.sync if e.type == 'B'],
        key=lambda x: x[0]
    )
    if not bpms:
        raise ValueError("No BPM events in chart")
    timing = []
    time_acc = 0.0
    prev_tick, prev_bpm = bpms[0]
    timing.append((prev_tick, time_acc, prev_bpm))
    for tick_i, bpm_i in bpms[1:]:
        dt = (tick_i - prev_tick) / chart.resolution * 60.0 / prev_bpm
        time_acc += dt
        timing.append((tick_i, time_acc, bpm_i))
        prev_tick, prev_bpm = tick_i, bpm_i
    return timing

def _time_to_tick(timing_map, resolution: int, t: float) -> int:
    for (ti, time_i, bpm_i), (tn, time_n, _) in zip(timing_map, timing_map[1:]):
        if time_i <= t < time_n:
            return int(round(ti + (t - time_i) * (bpm_i / 60.0) * resolution))
    # beyond last
    ti, time_i, bpm_i = timing_map[-1]
    return int(round(ti + (t - time_i) * (bpm_i / 60.0) * resolution))

def tick_to_time(tick: int, timing_map, resolution: int) -> float:
    for (ti, time_i, bpm_i), (tn, time_n, _) in zip(timing_map, timing_map[1:]):
        if ti <= tick < tn:
            return time_i + (tick - ti) / resolution * 60.0 / bpm_i
    ti, time_i, bpm_i = timing_map[-1]
    return time_i + (tick - ti) / resolution * 60.0 / bpm_i

def load_model(model_path: str):
    """Load your trained PyTorch drum-detection model."""
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

def transcribe_drums(
    chart: Chart,
    audio_path: str,
    model,
    class_thresholds: Dict[str, float],
    hop_length: int = 512,
    n_mels: int = 64
) -> Transcription:
    """
    1. Featurize drum stem → log-Mel spectrogram
    2. Framewise inference via `model` → (T, K) probabilities
    3. Peak‐pick per class with class_thresholds
    4. Convert frame indices → times → ticks
    5. Group hits per tick → Transcription
    """

    # 1) load audio & build mel
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(
        y, sr=sr,
        n_fft=2048,
        hop_length=hop_length,
        n_mels=n_mels
    )
    X = np.log1p(S).astype(np.float32)              # (n_mels, T)

    # 2) inference
    inp = torch.from_numpy(X[None, None, ...])      # (1,1,n_mels,T)
    with torch.no_grad():
        P = model(inp).squeeze(0).cpu().numpy()      # (T, K)

    # 3) pick peaks per class
    frame_times = librosa.frames_to_time(
        np.arange(P.shape[0]), sr=sr, hop_length=hop_length
    )
    timing_map = _build_timing_map(chart)
    hits: List[Note] = []

    # assume model.classes = ["kick","snare","tom1",...]
    for k, cat in enumerate(model.classes):
        thresh = class_thresholds.get(cat, 0.5)
        frames, _ = find_peaks(
            P[:, k],
            height=thresh,
            distance=int(0.05 * sr / hop_length)
        )
        for f in frames:
            t = float(frame_times[f])
            tick = _time_to_tick(timing_map, chart.resolution, t)
            hits.append(Note(time=t, tick=tick, category=cat))

    # 4) group simultaneous hits
    groups: Dict[int, List[Note]] = {}
    for h in hits:
        groups.setdefault(h.tick, []).append(h)

    return Transcription(type="ExpertDrums", notes=[groups[t] for t in sorted(groups)])