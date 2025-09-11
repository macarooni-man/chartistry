from __future__ import annotations
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataparse import DrumGridDataset, discover_song_dirs
from model import TinyDrumCNN


def _split_dirs(root: str, train=0.8, val=0.1):
    dirs = discover_song_dirs(root)
    random.shuffle(dirs)
    n = len(dirs)
    n_tr = int(n*train)
    n_val = int(n*val)
    n_te = n - n_tr - n_val
    return dirs[:n_tr], dirs[n_tr:n_tr+n_val], dirs[-n_te:]

def main(
    data_root: str = "data/songs",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    subdivision: int = 48,
    track_hint: str = "ExpertDrums",
    ckpt_path: str = "drum48_best.pt",
):
    tr_dirs, va_dirs, _ = _split_dirs(data_root)

    tr_set = DrumGridDataset(tr_dirs, subdivision=subdivision, track_hint=track_hint, debug=True)
    va_set = DrumGridDataset(va_dirs, subdivision=subdivision, track_hint=track_hint, debug=True)

    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available():       device = torch.device("cuda")
    else:                                 device = torch.device("cpu")
    model = TinyDrumCNN().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    # build loaders (tweak workers to taste)
    tr_loader = DataLoader(
        tr_set, batch_size=batch_size, shuffle=True,
        num_workers=4, persistent_workers=True, pin_memory=True
    )
    va_loader = DataLoader(
        va_set, batch_size=batch_size, shuffle=False,
        num_workers=4, persistent_workers=True, pin_memory=True
    )

    print(f"[i] Train examples: {len(tr_set):,} | Val examples: {len(va_set):,}")
    print(f"[i] Device: {device}")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(tr_loader, desc=f"epoch {ep} [train]", dynamic_ncols=True):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(1, len(tr_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            for x, y in tqdm(va_loader, desc=f"epoch {ep} [val]  ", dynamic_ncols=True):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = crit(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= max(1, len(va_loader.dataset))

        print(f"epoch {ep:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)


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
    args = parser.parse_args()

    # quick sanity
    from dataparse import discover_song_dirs
    found = discover_song_dirs(args.data_root)
    if not found:
        print(f"[!] No songs found under {args.data_root}.")
        print("    Expected each folder to contain both 'notes.chart' and 'drums.wav' (or 'mix.wav').")
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
    )

