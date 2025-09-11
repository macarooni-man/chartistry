from __future__ import annotations

import os
import io
import tempfile
from typing import List, Optional
import numpy as np
import torch
import torchaudio
import librosa

from chartlib import Chart, add_pro_drums_note
from dataparse import sample_rate, win_len_ms, hop_ms, n_mels, window_sec
from model import TinyDrumCNN

def _ms_to_samples(ms: float) -> int:
    return int(round(ms * sample_rate / 1000.0))

def _logmel(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sample_rate,
        n_fft=int(sample_rate * win_len_ms / 1000),
        hop_length=int(sample_rate * hop_ms / 1000),
        n_mels=n_mels, power=2.0,
    )
    S = np.log1p(S).astype(np.float32)
    return S

# ─────────────────────────────────────────────────────────────
# Build a temporary .chart *from audio only* by beat tracking.
# We:
#   1) Estimate beat times (seconds) with librosa.
#   2) Assume "beat == quarter note".
#   3) Build [SyncTrack] with B events at ticks i*Resolution where
#      us_per_qn = (interval_to_next_beat_sec * 1e6).
# This yields a piecewise tempo map that follows the audio.
# ─────────────────────────────────────────────────────────────
def _make_temp_chart_from_audio(audio_path: str,
                                resolution: int = 960,
                                min_bpm: float = 60.0,
                                max_bpm: float = 220.0) -> str:
    # Load audio mono→sample_rate
    wav, sr = torchaudio.load(audio_path)
    y = wav.mean(dim=0).numpy()
    if sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)

    # Beat tracking → beat_times (seconds)
    # backtrack=True places beats slightly earlier to align with onsets; helps drums.
    # Tighten tempo range to reduce half/double tempo errors.
    tempo, beats = librosa.beat.beat_track(
        y=y, sr=sample_rate, units="time", trim=True,
        bpm=120.0, tightness=100, start_bpm=None
    )
    beat_times: np.ndarray = beats  # already in seconds

    # Fallback if tracker fails: assume flat 120 BPM, beats every 0.5s
    if beat_times.size < 2:
        dur_s = len(y) / sample_rate
        beat_times = np.arange(0.0, dur_s, 0.5, dtype=float)

    # Clamp suspicious intervals (guard against spurious huge/small gaps)
    intervals = np.diff(beat_times)
    if intervals.size == 0:
        intervals = np.array([0.5], dtype=float)
    # Convert BPM per-interval and clamp
    bpms = 60.0 / np.maximum(intervals, 1e-3)
    bpms = np.clip(bpms, min_bpm, max_bpm)
    intervals = 60.0 / bpms  # back to seconds after clamping

    # Build .chart text in memory
    # We place a B event for each beat i at tick = i*Resolution with
    # us_per_qn = (interval_to_next_beat_sec * 1e6). For the last beat,
    # reuse the previous interval.
    out = io.StringIO()
    print("[Song]\n{", file=out)
    print(f"  Resolution = {resolution}", file=out)
    print(f'  Name = "temp_from_audio"', file=out)
    print("}", file=out)

    print("[SyncTrack]\n{", file=out)
    # First beat at tick 0
    print(f"  0 = TS 4", file=out)  # optional; 4 = numerator only in many charts
    # Initial tempo from first interval (or 120 BPM default if missing)
    if intervals.size > 0:
        us0 = int(round(intervals[0] * 1_000_000))
    else:
        us0 = int(round((60.0 / 120.0) * 1_000_000))
    print(f"  0 = B {us0}", file=out)

    # For each subsequent beat, set a new B using the interval to the next beat
    for i in range(1, len(beat_times)):
        tick_i = i * resolution  # assume beat==quarter note
        us = int(round((intervals[i-1] if i-1 < len(intervals) else intervals[-1]) * 1_000_000))
        print(f"  {tick_i} = B {us}", file=out)
    print("}", file=out)

    # Empty drums track to write predictions into
    print("[ExpertDrums]\n{", file=out)
    print("}", file=out)

    chart_text = out.getvalue()

    # Write to a temp file path and return it
    tmpdir = tempfile.mkdtemp(prefix="auto_tempo_")
    tmp_chart_path = os.path.join(tmpdir, "auto.chart")
    with open(tmp_chart_path, "w", encoding="utf-8") as f:
        f.write(chart_text)
    return tmp_chart_path

# ─────────────────────────────────────────────────────────────

def predict_chart(
    input_dir: Optional[str] = None,
    out_chart_path: Optional[str] = None,
    ckpt_path: str = "drum48_best.pt",
    subdivision: int = 48,
    track_name: str = "ExpertDrums",
    thresholds: Optional[List[float]] = None,
    # audio-only path: if provided, we ignore input_dir and build a temp chart
    audio_path: Optional[str] = None,
):
    """
    Generate a predicted .chart at a fixed musical grid (default 48ths).

    Modes:
      - Folder mode: set input_dir containing notes.chart + drums.wav (or mix.wav)
      - Audio-only mode: set audio_path to a WAV/OGG/MP3; we beat-track and build a temp chart.

    thresholds: optional per-class list of 8 values for [K,R,Y,B,G,Yc,Bc,Gc]
    """
    if thresholds is None:
        thresholds = [0.5] * 8

    if audio_path:
        # AUDIO-ONLY MODE
        tmp_chart_path = _make_temp_chart_from_audio(audio_path, resolution=960)
        chart = Chart(tmp_chart_path)
        audio_file = audio_path
        if out_chart_path is None:
            out_chart_path = os.path.splitext(audio_path)[0] + ".pred.chart"
    else:
        # FOLDER MODE
        if not input_dir:
            raise ValueError("Either audio_path or input_dir must be provided.")
        chart_path = os.path.join(input_dir, "notes.chart")
        if not os.path.exists(chart_path):
            # Fall back to audio-only if there is an obvious audio file present
            for cand in ("drums.wav", "mix.wav", "audio.wav"):
                ap = os.path.join(input_dir, cand)
                if os.path.exists(ap):
                    return predict_chart(
                        audio_path=ap,
                        out_chart_path=out_chart_path or os.path.join(input_dir, "predicted.chart"),
                        ckpt_path=ckpt_path,
                        subdivision=subdivision,
                        track_name=track_name,
                        thresholds=thresholds,
                    )
            raise FileNotFoundError("notes.chart not found and no audio file detected.")
        chart = Chart(chart_path)
        # choose audio
        for cand in ("drums.wav", "mix.wav", "audio.wav"):
            ap = os.path.join(input_dir, cand)
            if os.path.exists(ap):
                audio_file = ap
                break
        else:
            raise FileNotFoundError("No audio file found in the input_dir.")
        if out_chart_path is None:
            out_chart_path = os.path.join(input_dir, "predicted.chart")

    # Load audio (mono, sample_rate)
    wav, sr = torchaudio.load(audio_file)
    y = wav.mean(dim=0).numpy()
    if sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)

    # Load model
    model = TinyDrumCNN()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # Prepare output track
    tr = chart.get_track(track_name)
    tr.notes.clear()
    tr.specials.clear()

    # Build grid & decode per tick
    ticks = chart.build_grid_ticks(subdivision=subdivision, track_hint=track_name)

    half_ms = (window_sec * 1000.0) / 2.0
    for tick in ticks:
        center_ms = chart.tick_to_ms(tick)
        start_ms = max(0.0, center_ms - half_ms)
        end_ms   = min(len(y) * 1000.0 / sample_rate, center_ms + half_ms)

        s = _ms_to_samples(start_ms)
        e = _ms_to_samples(end_ms)

        expected_left  = _ms_to_samples(center_ms - half_ms)
        expected_right = _ms_to_samples(center_ms + half_ms)
        pad_left  = max(0, expected_left - s)
        pad_right = max(0, expected_right - e)

        y_win = y[s:e]
        if pad_left > 0:  y_win = np.pad(y_win, (pad_left, 0))
        if pad_right > 0: y_win = np.pad(y_win, (0, pad_right))

        S = _logmel(y_win)[None, None, ...]  # [1,1,N_MELS,T]
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(S))).cpu().numpy()[0]  # [8]

        K,R,Y,B,G,Yc,Bc,Gc = (probs >= np.array(thresholds)).astype(int)

        if K: add_pro_drums_note(tr, tick, 0, is_cymbal=False)
        if R: add_pro_drums_note(tr, tick, 1, is_cymbal=False)
        if Y: add_pro_drums_note(tr, tick, 2, is_cymbal=bool(Yc))
        if B: add_pro_drums_note(tr, tick, 3, is_cymbal=bool(Bc))
        if G: add_pro_drums_note(tr, tick, 4, is_cymbal=bool(Gc))

    chart.write(out_chart_path)
    print("wrote", out_chart_path)

if __name__ == "__main__":
    # Examples:
    # 1) Folder with notes.chart + drums.wav:
    # predict_chart(input_dir="data/songs/0001", out_chart_path="data/songs/0001/predicted.chart")
    #
    # 2) Audio-only (MP3/WAV):
    # predict_chart(audio_path="some_song.mp3", out_chart_path="some_song.pred.chart")
    pass
