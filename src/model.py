import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Two conv layers + BN + ReLU + MaxPool + Dropout.
    Double conv per block (VGG-style) gives more depth without
    parameter explosion — important on a 253-sample training set.
    """
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class BrugadaCNN(nn.Module):
    """
    1D CNN for 12-lead ECG binary classification.

    Architecture reasoning:
    - Kernel sizes decrease deeper (7→5→5→3): large kernels early capture
      long-range patterns (full QRS complex ~10 samples at 100Hz), smaller
      kernels later refine local morphology features
    - Global Average Pooling instead of Flatten: reduces parameters from
      ~2M to ~200K — critical for avoiding overfit on 253 training samples
    - No sigmoid in forward(): BCEWithLogitsLoss handles it numerically
      stable — always do this in PyTorch

    Input:  (batch, 12, 1200)
    Output: (batch, 1) — raw logit
    """

    def __init__(self, dropout=0.3):   # ← back to 0.3
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(12,  32,  kernel_size=7, dropout=dropout),  # → (32,  600)
            ConvBlock(32,  64,  kernel_size=5, dropout=dropout),  # → (64,  300)
            ConvBlock(64,  128, kernel_size=5, dropout=dropout),  # → (128, 150)
            ConvBlock(128, 256, kernel_size=3, dropout=dropout),  # → (256,  75)
        )

        # Collapses (256, 75) → (256, 1) regardless of input length
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x)
        return self.classifier(x)


def count_parameters(model):
    """Print total and trainable parameter count."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    return trainable