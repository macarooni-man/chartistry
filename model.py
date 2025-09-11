from __future__ import annotations
import torch.nn as nn

class TinyDrumCNN(nn.Module):
    """
    Input:  [B, 1, N_MELS, T]  (log-mel "image")
    Output: [B, 8]              (multi-label logits for [K,R,Y,B,G,Yc,Bc,Gc])
    """
    def __init__(self, n_mels: int = 96, n_out: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool2d((n_mels // 8, 16)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((n_mels // 8) * 16 * 64, 128), nn.ReLU(),
            nn.Linear(128, n_out),  # logits; use BCEWithLogitsLoss
        )

    def forward(self, x):
        z = self.features(x)
        return self.classifier(z)
