from scipy.signal import butter, filtfilt, find_peaks
from chart_parser import Chart, SyncEvent
from typing import List, Dict, Optional
from sklearn.cluster import KMeans
from dataclasses import dataclass
import numpy as np
import librosa


@dataclass
class Transcription:
    type: str          # chart type
    notes: list        # list of notes

@dataclass
class Note:
    time: float        # seconds
    tick: int          # chart tick
    category: str      # "kick", "snare", etc.

def _build_timing_map(chart: Chart):
    """
    From chart.sync (SyncEvent with type 'B' for BPM), build a list of
    (tick_i, time_i, bpm_i) sorted by tick_i, where time_i is the
    cumulative time in seconds at tick_i.
    """
    # extract BPM events
    bpms = sorted(
        [(e.tick, e.values[0] / 1000.0) for e in chart.sync if e.type == 'B'],
        key=lambda x: x[0]
    )
    if not bpms:
        raise ValueError("No BPM events in chart.sync")

    timing = []
    time_acc = 0.0
    prev_tick, prev_bpm = bpms[0]
    # assume chart offset=0 so time at first BPM tick is 0
    timing.append((prev_tick, time_acc, prev_bpm))

    for (tick_i, bpm_i) in bpms[1:]:
        delta_ticks = tick_i - prev_tick
        # duration = (ticks / resolution) * 60 / bpm
        duration = (delta_ticks / chart.resolution) * 60.0 / prev_bpm
        time_acc += duration
        timing.append((tick_i, time_acc, bpm_i))
        prev_tick, prev_bpm = tick_i, bpm_i

    return timing  # last BPM holds beyond end

def _time_to_tick(timing_map, resolution: int, t: float) -> int:
    """
    Given timing_map = [(tick_i, time_i, bpm_i), ...], find segment
    where time_i <= t < time_{i+1}, then:
      tick = tick_i + (t - time_i) * bpm_i/60 * resolution
    """
    for (ti, time_i, bpm_i), (tn, time_n, _) in zip(timing_map, timing_map[1:]):
        if time_i <= t < time_n:
            return int(round(ti + (t - time_i) * (bpm_i / 60.0) * resolution))
    # if beyond last segment:
    tick_i, time_i, bpm_i = timing_map[-1]
    return int(round(tick_i + (t - time_i) * (bpm_i / 60.0) * resolution))

def _bandpass(y, sr, low, high):
    # 4th-order butterworth bandpass
    nyq = sr/2
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, y)

# def transcribe_drums(
#     chart: Chart,
#     audio_path: str,
#     hop_length: int = 512,
#     bands: Optional[Dict[str, tuple]] = None
# ) -> Transcription:
#     # 1) load stereo
#     y, sr = librosa.load(audio_path, sr=None, mono=False)
#     if y.ndim == 1:
#         mid, side = y, np.zeros_like(y)
#     else:
#         L, R = y[0], y[1]
#         mid  = (L + R) / 2.0
#         side = (L - R) / 2.0
#
#     nyq = sr / 2.0
#     timing_map = _build_timing_map(chart)
#     hits: List[Note] = []
#
#     # 2) your bands + threshold + mid/side flag
#     if bands is None:
#         bands = {
#             "kick":    (20,   100,    0.005, False),  # RMS+peak
#             "snare":   (200,  500,    0.003, False),  # Onset strength
#             "tom1":    (200,  600,    0.002, True),
#             "tom2":    (600, 1200,    0.0015, True),
#             "tom3":    (1200,2000,    0.001, True),
#             "cymbal":  (5000, nyq*0.99, 0.0005, True)
#         }
#
#     # 3) process each band
#     for cat, (low_hz, high_hz, thresh, use_side) in bands.items():
#         low  = max(0.0, low_hz)
#         high = min(high_hz, nyq * 0.999)
#         if low >= high:
#             continue
#
#         sig = side if use_side else mid
#         yb = _bandpass(sig, sr, low, high)
#         yb = np.nan_to_num(yb)
#
#         if cat == "kick":
#             # ——— RMS + peak-pick for kicks ———
#             rms_env = librosa.feature.rms(
#                 y=yb, frame_length=2048, hop_length=hop_length
#             )[0]
#             # find peaks in the RMS that exceed thresh
#             peaks, _ = find_peaks(rms_env, height=thresh,
#                                   distance=int(0.2 * sr / hop_length))
#             frames = peaks
#         else:
#             # ——— onset strength for everything else ———
#             oenv = librosa.onset.onset_strength(
#                 y=yb, sr=sr, hop_length=hop_length
#             )
#             oenv = np.where(oenv > thresh, oenv, 0.0)
#             frames = librosa.onset.onset_detect(
#                 onset_envelope=oenv,
#                 sr=sr,
#                 hop_length=hop_length,
#                 backtrack=False
#             )
#
#         times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
#
#         for t in times:
#             tick = _time_to_tick(timing_map, chart.resolution, t)
#             hits.append(Note(time=t, tick=tick, category=cat))
#
#     # 4) group simultaneous hits
#     groups: Dict[int, List[Note]] = {}
#     for h in hits:
#         groups.setdefault(h.tick, []).append(h)
#
#     return Transcription(
#         type="ExpertDrums",
#         notes=[groups[t] for t in sorted(groups)]
#     )

# def transcribe_drums(chart, audio_path: str) -> Transcription:
#     # 0) your initial guesses
#     starting_freqs = {
#         "kick":    60,
#         "snare":   200,
#         "tom1":    400,
#         "tom2":    800,
#         "tom3":    1200,
#         "cymbal1": 6000,
#         "cymbal2": 9000,
#         "cymbal3": 12000,
#     }
#     sensitivity = {
#         "kick":    0.0,
#         "snare":   0.0,
#         "tom1":    0.0,
#         "tom2":    0.0,
#         "tom3":    0.0,
#         "cymbal1": 0.0,
#         "cymbal2": 0.0,
#         "cymbal3": 0.0,
#     }
#     categories = list(starting_freqs.keys())
#
#     print("Starting freqs:", starting_freqs)
#
#     # 1) load & find onsets
#     y, sr = librosa.load(audio_path, sr=None, mono=True)
#     oenv   = librosa.onset.onset_strength(y=y, sr=sr)
#     frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr)
#     times  = librosa.frames_to_time(frames, sr=sr)
#
#     # 2) measure each onset’s dominant freq
#     n_fft = 1024
#     win   = int(0.05 * sr)
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     doms  = []
#     for t in times:
#         c   = int(t * sr)
#         seg = y[max(0, c-win//2):c+win//2]
#         if len(seg) < n_fft:
#             seg = np.pad(seg, (0, n_fft-len(seg)))
#         S   = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=win))
#         mag = S.mean(axis=1)
#         idx = mag.argmax()
#         dom = float(freqs[idx])
#         doms.append(dom)
#         print(f"Transient @{t:.3f}s -> dominant {dom:.1f}Hz")
#     doms = np.array(doms)
#
#     # 3) assign each dom -> nearest starting center
#     start_centers = np.array([starting_freqs[c] for c in categories])
#     cat_doms = {c: [] for c in categories}
#     for f in doms:
#         i = int(np.argmin(np.abs(start_centers - f)))
#         cat = categories[i]
#         cat_doms[cat].append(f)
#
#     # 4) recompute each center as mean of assigned
#     centers = {}
#     for c in categories:
#         if cat_doms[c]:
#             centers[c] = float(np.mean(cat_doms[c]))
#         else:
#             centers[c] = float(starting_freqs[c])
#     print("Learned centers:", centers)
#
#     # 5) build sorted, non-overlapping bands
#     sorted_cats    = sorted(categories, key=lambda c: centers[c])
#     sorted_centers = [centers[c] for c in sorted_cats]
#     bounds = [0.0] + [
#         (sorted_centers[i] + sorted_centers[i+1]) / 2.0
#         for i in range(len(sorted_centers)-1)
#     ] + [sr/2.0]
#     bands = {
#         cat: (bounds[i], bounds[i+1])
#         for i, cat in enumerate(sorted_cats)
#     }
#     print("Frequency bands:")
#     for cat,(lo,hi) in bands.items():
#         print(f"  {cat}: {lo:.1f}–{hi:.1f} Hz (sens={sensitivity[cat]})")
#
#     # helper: bandpass
#     def _bandpass(sig, low, high):
#         nyq = sr/2.0
#         b, a = butter(4, [low/nyq, high/nyq], btype='band')
#         return filtfilt(b, a, sig)
#
#     # 6) per-band detection
#     timing_map = _build_timing_map(chart)
#     raw_hits = []
#
#     for cat, (low, high) in bands.items():
#         low  = max(1e-3, low)
#         high = min(high, sr/2.0 - 1e-3)
#         if low >= high:
#             continue
#
#         yb = _bandpass(y, low, high)
#         yb = np.nan_to_num(yb)
#         env = librosa.onset.onset_strength(y=yb, sr=sr)
#
#         floor = np.percentile(env, 10)
#         peak  = env.max()
#         sens  = np.clip(sensitivity[cat], 0.0, 1.0)
#         thresh = floor + sens * (peak - floor)
#         print(f"\n{cat} envelope stats: floor={floor:.3f}, peak={peak:.3f}, thresh={thresh:.3f}")
#
#         # show a few env values around each transient
#         for i, t in zip(frames, times):
#             print(f"  raw env at {t:.3f}s (frame {i}) = {env[i]:.3f}")
#
#         frames_p, props = find_peaks(env, height=thresh)
#         times_p = librosa.frames_to_time(frames_p, sr=sr)
#         print(f"  {cat} detected {len(frames_p)} peaks at times:", np.round(times_p,3))
#
#         for t in times_p:
#             tick = _time_to_tick(timing_map, chart.resolution, t)
#             raw_hits.append(Note(time=t, tick=tick, category=cat))
#
#     # 7) dedupe per tick+category
#     tick_map: Dict[int, Dict[str, Note]] = {}
#     for h in raw_hits:
#         cm = tick_map.setdefault(h.tick, {})
#         if h.category not in cm:
#             cm[h.category] = h
#
#     # 8) tom>cymbal priority
#     for cm in tick_map.values():
#         for i in (1,2,3):
#             tom, cym = f"tom{i}", f"cymbal{i}"
#             if tom in cm and cym in cm:
#                 del cm[cym]
#
#     # 9) build final groups
#     groups = [list(cm.values()) for _, cm in sorted(tick_map.items())]
#     return Transcription(type="ExpertDrums", notes=groups)


def transcribe_drums(
    chart: Chart,
    audio_path: str,
    centroid_thresholds: Optional[Dict[str, float]] = None
) -> Transcription:
    """
    1) Load drum-only audio
    2) Detect onsets
    3) Compute spectral centroid per onset
    4) If no static thresholds provided, cluster those centroids into
       four groups (kick/snare/tom/cymbal) and set thresholds at the
       midpoints between cluster centers.
    5) Classify each onset’s centroid against those thresholds
    6) Map onset times → ticks and group simultaneous hits
    """

    import numpy as np
    import librosa
    from sklearn.cluster import KMeans
    from typing import Dict, List

    # 1) load & 2) detect onsets
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr,
                                        hop_length=hop, backtrack=False)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)

    # 3) compute spectral centroid for each onset window
    win = int(0.05 * sr)
    centroids: List[float] = []
    for t in times:
        center = int(t * sr)
        seg = y[max(0, center - win//2): center + win//2]
        if len(seg) < 2:
            centroids.append(0.0)
            continue
        S, _ = librosa.magphase(librosa.stft(seg, n_fft=1024, hop_length=512))
        cent = librosa.feature.spectral_centroid(S=S)[0].mean()
        centroids.append(float(cent))

    cent_arr = np.array(centroids).reshape(-1, 1)

    # 4) dynamic threshold derivation via clustering
    if centroid_thresholds is None:
        # cluster into 4 groups
        km = KMeans(n_clusters=4, random_state=0).fit(cent_arr)
        centers = np.sort(km.cluster_centers_.flatten())
        # thresholds are midpoints between adjacent cluster centers
        t0 = (centers[0] + centers[1]) / 2.0
        t1 = (centers[1] + centers[2]) / 2.0
        t2 = (centers[2] + centers[3]) / 2.0
        centroid_thresholds = {
            "kick":   t0,
            "snare":  t1,
            "tom":    t2
        }

    # build time→tick map
    timing_map = _build_timing_map(chart)

    # 5) classify and map to ticks
    hits: List[Note] = []
    for t, cent in zip(times, centroids):
        if cent < centroid_thresholds["kick"]:
            cat = "kick"
        elif cent < centroid_thresholds["snare"]:
            cat = "snare"
        elif cent < centroid_thresholds["tom"]:
            cat = "tom1"
        else:
            cat = "cymbal1"

        tick = _time_to_tick(timing_map, chart.resolution, t)
        hits.append(Note(time=t, tick=tick, category=cat))

    # 6) group simultaneous hits
    groups: Dict[int, List[Note]] = {}
    for h in hits:
        groups.setdefault(h.tick, []).append(h)

    return Transcription("ExpertDrums", [groups[t] for t in sorted(groups)])
