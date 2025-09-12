from __future__ import annotations
import os, bisect, torch, torchaudio
import torchaudio.functional as AF
import torch.nn.functional as F
from typing import List, Dict, Tuple
from tqdm import tqdm

from chartlib import Chart
from dataparse import discover_song_dirs, mel_config, sample_rate, window_sec

def _label_from_lanes(lanes: set[int]) -> torch.Tensor:
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

def _find_chart(dirpath: str):
    for name in os.listdir(dirpath):
        if name.lower().endswith(".chart"):
            return os.path.join(dirpath, name)
    return None

def _find_audio(dirpath: str):
    exts = (".wav",".ogg",".mp3",".flac",".m4a")
    for base in ("drums","mix","audio"):
        for ext in exts:
            cand = os.path.join(dirpath, base+ext)
            if os.path.exists(cand):
                return cand
            cand2 = os.path.join(dirpath, base.capitalize()+ext)
            if os.path.exists(cand2):
                return cand2
            cand3 = os.path.join(dirpath, base.upper()+ext)
            if os.path.exists(cand3):
                return cand3
    for fn in os.listdir(dirpath):
        if fn.lower().endswith(exts):
            return os.path.join(dirpath, fn)
    return None

def precompute_song(song_dir: str, track_hint="ExpertDrums", subdivision=48) -> bool:
    chart_path = _find_chart(song_dir)
    audio_path = _find_audio(song_dir)
    if not chart_path or not audio_path:
        return False

    ch = Chart(chart_path)

    # grid ticks
    ticks = ch.build_grid_ticks(subdivision=subdivision, track_hint=track_hint)

    # labels (once per tick)
    step = ch.resolution // round(subdivision / 4)
    tol_ticks = max(1, step // 2)

    note_list = []
    if track_hint in ch.tracks:
        note_list = sorted(ch.tracks[track_hint].notes, key=lambda n: n.tick)
    note_ticks = [n.tick for n in note_list]

    labels: List[torch.Tensor] = []
    for t in ticks:
        L = bisect.bisect_left(note_ticks,  t - tol_ticks)
        R = bisect.bisect_right(note_ticks, t + tol_ticks)
        lanes = {note_list[i].lane for i in range(L, R)}
        labels.append(_label_from_lanes(lanes))
    labels = torch.stack(labels, dim=0)  # [N_ticks, 8]

    # center frame index per tick (mel-time)
    centers_ms = torch.tensor([ch.tick_to_ms(t) for t in ticks], dtype=torch.float32)  # [N_ticks]

    # load & resample audio ONCE
    wav, sr = torchaudio.load(audio_path)           # [C,N]
    y = wav.mean(dim=0)                             # mono [N]
    if sr != sample_rate:
        y = AF.resample(y, sr, sample_rate)

    # mel on CPU (one-time offline)
    cfg = mel_config()
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
        f_min=cfg["f_min"],
        f_max=cfg["f_max"],
        power=cfg["power"],
        norm=cfg["norm"],
        mel_scale=cfg["mel_scale"],
    )
    S = mel_tf(y.unsqueeze(0))     # [1, n_mels, F]
    S = torch.log1p(S)             # [1, n_mels, F]
    S = S.squeeze(0).contiguous()  # [n_mels, F]

    hop = cfg["hop_length"]
    centers_frames = torch.round(centers_ms/1000.0 * sample_rate / hop).to(torch.long)  # [N_ticks]
    win_frames = int(round(window_sec * sample_rate / hop))  # total frames per window
    half = win_frames // 2

    # save once per song
    torch.save({
        "mel": S,                               # [n_mels, F]
        "hop": hop,
        "sample_rate": sample_rate,
        "window_frames": win_frames,
        "centers_frames": centers_frames,       # [N_ticks]
        "labels": labels,                       # [N_ticks, 8]
        "ticks": torch.tensor(ticks, dtype=torch.long),
        "track_hint": track_hint,
        "subdivision": subdivision,
    }, os.path.join(song_dir, "chart_feats.pt"))
    return True

def main(data_root: str, track_hint="ExpertDrums", subdivision=48):
    dirs = discover_song_dirs(data_root)
    print(f"[precompute] {len(dirs)} song(s)")
    ok = 0
    for d in tqdm(dirs):
        out = os.path.join(d, "chart_feats.pt")
        if os.path.exists(out):
            continue
        try:
            if precompute_song(d, track_hint=track_hint, subdivision=subdivision):
                ok += 1
        except Exception as e:
            print(f"[warn] {d}: {e}")
    print(f"[done] wrote {ok} new files")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./dataset")
    p.add_argument("--track_hint", default="ExpertDrums")
    p.add_argument("--subdivision", type=int, default=48)
    a = p.parse_args()
    main(a.data_root, a.track_hint, a.subdivision)
