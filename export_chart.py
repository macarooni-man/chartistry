# export_chart.py
from __future__ import annotations
import os, glob, torch
import torch.nn.functional as F
from typing import List, Tuple
from tqdm import tqdm

from model import TinyDrumCNN
from chartlib import Chart, Track, add_pro_drums_note, Note

# --------- helpers ---------

def _find_precomp(song_dir: str) -> str | None:
    p = os.path.join(song_dir, "chart_feats.pt")
    return p if os.path.exists(p) else None

def _find_chart(song_dir: str) -> str | None:
    hits = [os.path.join(song_dir, n) for n in os.listdir(song_dir) if n.lower().endswith(".chart")]
    return hits[0] if hits else None

def _median_filter_bool(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """x: [N] bool; odd k; returns bool median-smoothed."""
    if k <= 1: return x
    pad = (k - 1) // 2
    x_f = x.float().view(1, 1, -1)
    w = torch.ones(1, 1, k, device=x.device)
    s = F.conv1d(F.pad(x_f, (pad, pad), mode="replicate"), w)
    return (s >= ((k + 1) // 2)).view(-1).bool()

# --------- core: predict windows -> boolean events ---------

@torch.no_grad()
def predict_song(song_dir: str,
                 ckpt: str = "drum48_best.pt",
                 batch: int = 256,
                 thresholds: List[float] | None = None,
                 smooth_k: int = 3) -> tuple[torch.Tensor, dict]:
    """
    Returns:
      preds_bool: [N_ticks, 8] bool tensor
      pack: dict from chart_feats.pt (contains ticks, window_frames, centers_frames, mel, etc.)
    """
    pack_path = _find_precomp(song_dir)
    if not pack_path:
        raise FileNotFoundError(f"No chart_feats.pt in {song_dir}. Run precompute.py first.")

    pack = torch.load(pack_path, map_location="cpu")
    S = pack["mel"]                     # [n_mels, F]
    centers = pack["centers_frames"]    # [N]
    W = int(pack["window_frames"])
    half = W // 2
    n_mels, Ftot = S.shape
    N = centers.numel()

    if thresholds is None:
        # reasonable defaults: cymbal flags usually need to be higher
        thresholds = [0.45, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6]
    thr = torch.tensor(thresholds, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDrumCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    outs = []
    for i0 in tqdm(range(0, N, batch), desc=os.path.basename(song_dir)):
        i1 = min(N, i0 + batch)
        xs = []
        for i in range(i0, i1):
            c = int(centers[i])
            s = max(0, c - half)
            e = min(Ftot, c + half)
            x = S[:, s:e]  # expect [n_mels, <= W]

            # If a stray channel dim is present (e.g., [1, n_mels, T]), drop it.
            if x.ndim == 3 and x.size(0) == 1:
                x = x.squeeze(0)

            # Right-pad to fixed window width
            if x.size(1) < W:
                x = F.pad(x, (0, W - x.size(1)))  # [n_mels, W]

            assert x.ndim == 2 and x.size(0) == n_mels and x.size(1) == W, \
                f"window shape wrong: {tuple(x.shape)}"
            xs.append(x)

        X = torch.stack(xs, dim=0)  # [B, n_mels, W]
        assert X.ndim == 3, f"stack produced {tuple(X.shape)}"
        X = X.unsqueeze(1).contiguous().to(device)  # [B, 1, n_mels, W]
        # print(f"batch X shape: {tuple(X.shape)}")  # uncomment to verify

        logits = model(X)  # [B, 8]
        outs.append(torch.sigmoid(logits).cpu())
    probs = torch.cat(outs, dim=0)  # [N, 8]

    # --- DIAGNOSTICS: see ranges and counts ---
    names = ["K", "R", "Y", "B", "G", "Yc", "Bc", "Gc"]
    mx = probs.max(dim=0).values
    mn = probs.min(dim=0).values
    mu = probs.mean(dim=0)
    print("[stats] max:", {n: f"{mx[i]:.3f}" for i, n in enumerate(names)})
    print("[stats] mean:", {n: f"{mu[i]:.3f}" for i, n in enumerate(names)})
    print("[stats] min:", {n: f"{mn[i]:.3f}" for i, n in enumerate(names)})

    if thresholds is None:
        thresholds = [0.45, 0.50, 0.50, 0.50, 0.50, 0.60, 0.60, 0.60]
    thr = torch.tensor(thresholds, dtype=torch.float32)
    preds = probs >= thr
    counts = preds.sum(dim=0)
    print("[stats] thr:", {n: f"{thr[i]:.2f}" for i, n in enumerate(names)})
    print("[stats] positives:", {n: int(counts[i]) for i, n in enumerate(names)})


    # Threshold + (optional) tiny temporal smoothing per class
    preds = probs >= thr
    if smooth_k and smooth_k > 1:
        for j in range(preds.size(1)):
            preds[:, j] = _median_filter_bool(preds[:, j], k=smooth_k)

    return preds, pack

# --------- build a new [ExpertDrums] from predictions ---------

def build_track_from_preds(preds: torch.Tensor, pack: dict) -> Track:
    """
    preds: [N_ticks, 8] bool  (K,R,Y,B,G,Yc,Bc,Gc)
    """
    dr = Track()
    ticks = pack["ticks"].tolist()
    for i, tick in enumerate(ticks):
        row = preds[i]
        # pads
        if row[0]:  # kick
            dr.notes.append(Note(tick=tick, lane=0, length=0))
        if row[1]:  # red (snare)
            dr.notes.append(Note(tick=tick, lane=1, length=0))
        if row[2]:  # yellow
            dr.notes.append(Note(tick=tick, lane=2, length=0))
        if row[3]:  # blue
            dr.notes.append(Note(tick=tick, lane=3, length=0))
        if row[4]:  # green
            dr.notes.append(Note(tick=tick, lane=4, length=0))

        # cymbal flags (only meaningful on Y/B/G)
        if row[2] and row[5]:  # yellow cymbal
            dr.notes.append(Note(tick=tick, lane=66, length=0))
        if row[3] and row[6]:  # blue cymbal
            dr.notes.append(Note(tick=tick, lane=67, length=0))
        if row[4] and row[7]:  # green cymbal
            dr.notes.append(Note(tick=tick, lane=68, length=0))
    return dr

# --------- top-level: export one song ---------

def export_song(song_dir: str,
                out_path: str | None = None,
                ckpt: str = "drum48_best.pt",
                batch: int = 256,
                thresholds: List[float] | None = None,
                smooth_k: int = 3) -> str:
    preds, pack = predict_song(song_dir, ckpt=ckpt, batch=batch,
                               thresholds=thresholds, smooth_k=smooth_k)

    # load original chart to copy metadata + sync, then replace [ExpertDrums]
    chart_path = _find_chart(song_dir)
    if not chart_path:
        raise FileNotFoundError(f"No .chart found in {song_dir}")
    ch = Chart(chart_path)

    ch.tracks["ExpertDrums"] = build_track_from_preds(preds, pack)

    if out_path is None:
        base = os.path.basename(song_dir).replace(" ", "_")
        out_path = os.path.join(song_dir, f"{base}_pred.chart")

    ch.write(out_path)
    return out_path

# --------- CLI ---------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("song_dir", help="Path to the song folder")
    p.add_argument("--ckpt", default="drum48_best.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--smooth", type=int, default=3, help="median filter width (odd). 0/1=off")
    p.add_argument("--thr", type=float, nargs=8, default=None,
                   help="8 thresholds: K R Y B G Yc Bc Gc (defaults: 0.45 0.5 0.5 0.5 0.5 0.6 0.6 0.6)")
    args = p.parse_args()

    out = export_song(
        args.song_dir,
        out_path=args.out,
        ckpt=args.ckpt,
        batch=args.batch,
        thresholds=args.thr,
        smooth_k=args.smooth,
    )
    print(f"[wrote] {out}")
