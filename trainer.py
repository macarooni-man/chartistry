from __future__ import annotations
import warnings
warnings.filterwarnings(
    "once",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec"
)

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio

from dataparse_precomputed import DrumGridDataset, discover_song_dirs
from dataparse import mel_config
from model import TinyDrumCNN


# ------------------------- helpers -------------------------

def _split_dirs(root: str, train=0.8, val=0.1):
    dirs = discover_song_dirs(root)
    random.shuffle(dirs)
    n = len(dirs)
    n_tr = int(n * train)
    n_val = int(n * val)
    n_te = n - n_tr - n_val
    return dirs[:n_tr], dirs[n_tr:n_tr+n_val], dirs[-n_te:]


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_mel_on(device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    cfg = mel_config()
    mel = torchaudio.transforms.MelSpectrogram(
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
    return mel.to(device)



# ------------------------- training -------------------------

def main(
    data_root: str = "data/songs",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    subdivision: int = 48,          # 48th notes by default
    track_hint: str = "ExpertDrums",
    ckpt_path: str = "drum48_best.pt",
    num_workers: int = 4,
):
    # tr_dirs, va_dirs, _ = _split_dirs(data_root)
    tr_dirs = discover_song_dirs(data_root)
    va_dirs = tr_dirs[:max(1, len(tr_dirs) // 10)]
    tr_dirs = tr_dirs[len(va_dirs):]

    tr_set = DrumGridDataset(tr_dirs)
    va_set = DrumGridDataset(va_dirs)

    # Datasets (set debug=True for a one-time summary)
    # tr_set = DrumGridDataset(tr_dirs, subdivision=subdivision, track_hint=track_hint, debug=True)
    # va_set = DrumGridDataset(va_dirs, subdivision=subdivision, track_hint=track_hint, debug=True)

    device = _pick_device()
    print(f"[i] Device: {device}")

    model = TinyDrumCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    # AMP (new API; avoids deprecation warning)
    scaler = torch.amp.GradScaler(device.type if device.type in ("cuda", "xpu") else "cpu")
    if device.type in ("cuda", "xpu"):
        torch.set_float32_matmul_precision("high")

    # GPU mel front-end (used only if dataset returns raw waveform)
    mel = _build_mel_on(device)

    # DataLoaders
    pin = (device.type in ("cuda", "xpu"))
    tr_loader = DataLoader(
        tr_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
        pin_memory=pin, prefetch_factor=4 if num_workers > 0 else None,
    )
    va_loader = DataLoader(
        va_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
        pin_memory=pin, prefetch_factor=4 if num_workers > 0 else None,
    )

    print(f"[i] Train examples: {len(tr_set):,} | Val examples: {len(va_set):,}")

    best_val = float("inf")

    for ep in range(1, epochs + 1):
        # ---------------- train ----------------
        model.train()
        train_loss = 0.0

        for X, y in tqdm(tr_loader, desc=f"epoch {ep} [train]", dynamic_ncols=True):
            # X shape can be either:
            #   - raw waveform: [B, 1, T]
            #   - mel already:  [B, 1, n_mels, F]
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # If input is raw waveform (3D), compute mel on device
            if X.ndim == 3:
                with torch.amp.autocast(device_type=device.type, enabled=(device.type in ("cuda","xpu"))):
                    S = mel(X)          # [B, 1, n_mels, F]
                    S = torch.log1p(S)  # match previous np.log1p
            elif X.ndim == 4:
                S = X                  # already a mel image
            else:
                raise RuntimeError(f"Unexpected input shape {tuple(X.shape)}")

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type in ("cuda","xpu"))):
                logits = model(S)
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * X.size(0)

        train_loss /= max(1, len(tr_loader.dataset))

        # ---------------- val ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in tqdm(va_loader, desc=f"epoch {ep} [val]  ", dynamic_ncols=True):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if X.ndim == 3:
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type in ("cuda","xpu"))):
                        S = mel(X)
                        S = torch.log1p(S)
                elif X.ndim == 4:
                    S = X
                else:
                    raise RuntimeError(f"Unexpected input shape {tuple(X.shape)}")

                with torch.amp.autocast(device_type=device.type, enabled=(device.type in ("cuda","xpu"))):
                    logits = model(S)
                    loss = crit(logits, y)

                val_loss += loss.item() * X.size(0)

        val_loss /= max(1, len(va_loader.dataset))

        print(f"epoch {ep:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"[i] saved checkpoint → {ckpt_path}")


# ------------------------- CLI -------------------------

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subdivision", type=int, default=48)   # 48ths
    parser.add_argument("--track_hint", default="ExpertDrums")
    parser.add_argument("--ckpt_path", default="drum48_best.pt")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # quick sanity
    found = discover_song_dirs(args.data_root)
    if not found:
        print(f"[!] No songs found under {args.data_root}.")
        print("    Expected each folder to contain a .chart and an audio file.")
        sys.exit(1)
    print(f"[i] Found {len(found)} song folder(s). Example: {found[0]}")

    main(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subdivision=args.subdivision,
        track_hint=args.track_hint,
        ckpt_path=args.ckpt_path,
        num_workers=args.num_workers,
    )
