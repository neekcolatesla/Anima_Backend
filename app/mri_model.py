"""
Anima - MRI image classification model (shared architecture).

Defines the shallow 2D CNN that classifies a patient's anatomical MRI slices as
ADHD vs control. This module is the SHARED contract, exactly mirroring the role
dsm5_model.py plays for the text model:

    * train_mri.py       (built later) constructs + trains MRICNN, saves weights.
    * MRI_Analysis.py    rebuilds MRICNN and loads those weights to score.

Keeping the class in one place means train and serve use the byte-for-byte same
architecture (no skew). The artifact is a {config, state_dict} dict of tensors +
primitives, so torch.load stays safe under the weights_only default of torch >= 2.6.

Deliberately SHALLOW (why so small?)
------------------------------------
Medical imaging cohorts are tiny - here ~130 labelled patients. A deep CNN
(ResNet-scale, millions of parameters) would simply memorise those volumes and
generalise poorly: classic small-data overfitting. So the network is kept
intentionally light:

    * only THREE convolutional blocks (16 -> 32 -> 64 channels);
    * a Global Average Pool instead of large flattened fully-connected layers,
      which collapses each feature map to one number and removes the parameter
      explosion a flatten+dense head would add;
    * a single linear classifier (64 -> 1) producing one ADHD-risk logit;
    * BatchNorm for stable training on small batches and Dropout for regularisation.

The whole model is on the order of ~10^4 parameters - enough to learn gross
grey-matter morphometric differences, small enough to resist memorising the set.

Input contract
--------------
A batch of 2-channel slices, shape (N, in_channels, H, W). The two channels are
the paired anatomical scans - channel 0 = structural T1 (`anat`), channel 1 =
grey-matter segmentation (`anat_gm`) - so each spatial location carries both the
whole-brain anatomy and the grey-matter signal ADHD differences concentrate in.
forward() returns one logit per slice, shape (N,); sigmoid(logit) = ADHD risk.
Per-patient aggregation (mean over the slice stack) is the caller's job, NOT the
model's - keeping the CNN a pure per-image classifier.
"""

import torch
import torch.nn as nn

# Two input channels: anat (structural T1) + anat_gm (grey-matter segmentation).
IN_CHANNELS = 2
# Slices are resized to this square edge before entering the network.
INPUT_SIZE = 128
# Convolutional block widths (kept small on purpose - see module docstring).
CHANNELS = (16, 32, 64)


class _ConvBlock(nn.Module):
    """Conv(3x3, pad 1) -> BatchNorm -> ReLU -> MaxPool(2). Halves H,W each block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MRICNN(nn.Module):
    """Shallow 2D CNN: 2-channel MRI slice -> single ADHD-risk logit."""

    def __init__(self, in_channels: int = IN_CHANNELS, input_size: int = INPUT_SIZE,
                 dropout: float = 0.4, channels=CHANNELS):
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_size = int(input_size)
        self.dropout = float(dropout)
        self.channels = tuple(int(c) for c in channels)

        # Stack the (few) convolutional blocks: in_channels -> 16 -> 32 -> 64.
        blocks = []
        prev = self.in_channels
        for width in self.channels:
            blocks.append(_ConvBlock(prev, width))
            prev = width
        self.features = nn.Sequential(*blocks)

        # Global Average Pool -> one value per channel, independent of input size.
        # This is what keeps the parameter count tiny (no giant flattened dense layer).
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(prev, 1)   # 64 -> 1 ADHD-risk logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, in_channels, H, W) -> (N,) ADHD-risk logits (pre-sigmoid)."""
        x = self.features(x)
        x = self.pool(x)                 # (N, C, 1, 1)
        x = torch.flatten(x, 1)          # (N, C)
        x = self.drop(x)
        return self.classifier(x).squeeze(-1)   # (N,)

    def config(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "input_size": self.input_size,
            "dropout": self.dropout,
            "channels": list(self.channels),
        }


def save_cnn(model: MRICNN, path: str) -> None:
    """Persist as {config, state_dict} - all tensors/primitives (weights_only-safe)."""
    torch.save({"config": model.config(), "state_dict": model.state_dict()}, path)


def load_cnn(path: str, map_location: str = "cpu") -> MRICNN:
    """Rebuild MRICNN from a saved artifact; returns an eval-mode model.

    Accepts the {config, state_dict} format and (for backward compatibility) a
    bare state_dict, which is assumed to be a default-geometry model.
    """
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj:
        cfg = obj.get("config", {})
        model = MRICNN(
            in_channels=cfg.get("in_channels", IN_CHANNELS),
            input_size=cfg.get("input_size", INPUT_SIZE),
            dropout=cfg.get("dropout", 0.4),
            channels=cfg.get("channels", CHANNELS),
        )
        model.load_state_dict(obj["state_dict"])
    else:
        model = MRICNN()
        model.load_state_dict(obj)
    model.eval()
    return model
