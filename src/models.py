"""
Model definitions and a small registry so train / evaluate can pick a model by name.

    v2 - rPPGCNN_v2: 3D-CNN, spatial global pooling + 1D temporal head (~507K params)
"""

import os
from typing import Dict, Optional, Type

import torch
import torch.nn as nn

from . import config


class rPPGCNN_v2(nn.Module):
    """
    Spatio-temporal 3D-CNN for BVP estimation.

    Input  (B, 3, T, 64, 64)
    Output (B, T)
    """

    def __init__(
        self,
        input_channels: int = config.INPUT_CHANNELS,
        output_dim: int = config.WINDOW_FRAMES,
        dropout_rate: float = 0.2,
        negative_slope: float = 0.1,
    ):
        super().__init__()

        # (B,3,T,64,64) -> (B,32,T,32,32)
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.act1 = nn.LeakyReLU(negative_slope, inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # -> (B,64,T,16,16)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.act2 = nn.LeakyReLU(negative_slope, inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # -> (B,128,T,8,8)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(128)
        self.act3 = nn.LeakyReLU(negative_slope, inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # -> (B,64,T,8,8)
        self.conv4 = nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(64)
        self.act4 = nn.LeakyReLU(negative_slope, inplace=True)

        # Spatial global average pooling, time preserved -> (B,64,T,1,1)
        self.spatial_pool = nn.AdaptiveAvgPool3d((output_dim, 1, 1))

        # 1D temporal refinement -> (B,32,T)
        self.temporal_conv = nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False)
        self.temporal_bn = nn.BatchNorm1d(32)
        self.temporal_act = nn.LeakyReLU(negative_slope, inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        # Pointwise projection -> (B,1,T)
        self.output_conv = nn.Conv1d(32, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = self.act4(self.bn4(self.conv4(x)))

        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)  # (B, 64, T)

        x = self.temporal_act(self.temporal_bn(self.temporal_conv(x)))
        x = self.dropout(x)
        return self.output_conv(x).squeeze(1)  # (B, T)


# ==============================================================================
# REGISTRY
# ==============================================================================
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "v2": rPPGCNN_v2,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_weights(
    model: nn.Module, path: str, device: Optional[torch.device] = None
) -> nn.Module:
    """
    Loads a checkpoint into `model`. Accepts either a bare state_dict (legacy files) or a
    training checkpoint dict with a "model" key.
    """
    if device is None:
        device = config.get_device()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model
