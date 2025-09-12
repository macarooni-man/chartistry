import torch
from dataparse import DrumGridDataset
from model import TinyDrumCNN

def main(song_dir, ckpt="drum48_best.pt"):
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDrumCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # make dataset for just this song
    ds = DrumGridDataset([song_dir], subdivision=48, track_hint="ExpertDrums")

    # run through every tick
    preds = []
    with torch.no_grad():
        for x, _ in ds:
            x = x.unsqueeze(0).to(device)     # add batch dim
            out = torch.sigmoid(model(x))     # values 0..1
            preds.append(out.cpu())

    # save raw predictions
    preds = torch.cat(preds, dim=0)
    torch.save(preds, "preds.pt")
    print(f"[i] wrote preds.pt with shape {preds.shape}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_infer.py <song_folder>")
    else:
        main(sys.argv[1])
