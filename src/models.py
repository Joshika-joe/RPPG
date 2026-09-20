"""
Model definitions and a small registry so train / evaluate can pick a model by name.

    v2 - rPPGCNN_v2: 3D-CNN, spatial global pooling + 1D temporal head (~507K params)
    v3 - rPPGCNN_v3: 2D+1D. Shared 2D spatial encoder per frame, appearance-driven spatial
         attention pooling, dilated 1D temporal residual blocks, conv1d head (~165K params)

All models take forward(x, appearance=None) with x (B, 3, T, H, W). `appearance` is the
window's mean frame (B, 3, H, W) - v2 ignores it, v3 uses it for the attention mask.
Use run_model() to call a model on a DataLoader batch.
"""

import os
from typing import Dict, Optional, Type

import torch
import torch.nn as nn

from . import config


# ==============================================================================
# v2: 3D CNN
# ==============================================================================
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

    def forward(self, x: torch.Tensor, appearance: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = self.act4(self.bn4(self.conv4(x)))

        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)  # (B, 64, T)

        x = self.temporal_act(self.temporal_bn(self.temporal_conv(x)))
        x = self.dropout(x)
        return self.output_conv(x).squeeze(1)  # (B, T)


# ==============================================================================
# v3: 2D + 1D
# ==============================================================================
def _conv_bn_act(cin: int, cout: int, k: int = 3, stride: int = 1, slope: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(slope, inplace=True),
    )


class TemporalBlock(nn.Module):
    """Residual dilated 1D conv: (B, C, T) -> (B, C, T); receptive field grows by (k-1)*dilation."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
        slope: float = 0.1,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(slope, inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.bn(self.conv(x))))


class rPPGCNN_v3(nn.Module):
    """
    2D+1D network for BVP estimation.

      x (B,3,T,64,64)   -> shared 2D encoder on every frame        -> (B*T, C, 8, 8)
      mean frame (B,3,64,64) -> appearance branch -> attention mask -> (B, 1, 8, 8)
      attention-weighted spatial pooling                            -> (B, C, T)
      dilated residual 1D temporal blocks (RF = 1 + 4*sum(dil))     -> (B, C, T)
      1x1 conv head                                                 -> (B, T)

    Temporal normalization removes appearance from x, so "where is the skin" is decided
    from the window's mean frame instead - computed once per window, not per frame.
    The last mask is kept in `self.last_attention` for visualization.
    """

    def __init__(
        self,
        input_channels: int = config.INPUT_CHANNELS,
        output_dim: int = config.WINDOW_FRAMES,  # unused: output length follows T
        channels: int = 64,
        dilations=(1, 2, 4, 8, 16),
        dropout: float = 0.1,
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self.channels = channels

        # Spatial encoder (per frame): 64 -> 32 -> 16 -> 8
        self.spatial = nn.Sequential(
            _conv_bn_act(input_channels, 16, slope=negative_slope),
            nn.MaxPool2d(2),
            _conv_bn_act(16, 32, slope=negative_slope),
            nn.MaxPool2d(2),
            _conv_bn_act(32, channels, slope=negative_slope),
            nn.MaxPool2d(2),
            _conv_bn_act(channels, channels, slope=negative_slope),
        )

        # Appearance branch (per window): 64 -> 8, one attention logit per cell
        self.appearance = nn.Sequential(
            _conv_bn_act(input_channels, 8, stride=2, slope=negative_slope),
            _conv_bn_act(8, 16, stride=2, slope=negative_slope),
            _conv_bn_act(16, 16, stride=2, slope=negative_slope),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Temporal encoder
        self.temporal = nn.Sequential(
            *[TemporalBlock(channels, d, dropout=dropout, slope=negative_slope) for d in dilations]
        )
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

        self.last_attention: Optional[torch.Tensor] = None

    def attention_mask(self, appearance: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) mean frame -> (B,1,h,w) mask that averages to 1 over the cells."""
        a = appearance - appearance.mean(dim=(1, 2, 3), keepdim=True)
        a = a / (a.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        mask = torch.sigmoid(self.appearance(a))
        return mask / (mask.mean(dim=(2, 3), keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor, appearance: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feat = self.spatial(frames)  # (B*T, C', h', w')
        _, cf, hf, wf = feat.shape
        feat = feat.reshape(b, t, cf, hf, wf)

        if appearance is None:  # no appearance given: uniform pooling
            mask = torch.ones(b, 1, hf, wf, device=x.device, dtype=x.dtype)
        else:
            mask = self.attention_mask(appearance)
        self.last_attention = mask.detach()

        pooled = (feat * mask.unsqueeze(1)).mean(dim=(3, 4))  # (B, T, C')
        seq = pooled.permute(0, 2, 1)  # (B, C', T)
        seq = self.temporal(seq)
        return self.head(seq).squeeze(1)  # (B, T)


# ==============================================================================
# REGISTRY
# ==============================================================================
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "v2": rPPGCNN_v2,
    "v3": rPPGCNN_v3,
}
MODEL_NAMES = sorted(MODEL_REGISTRY)


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {MODEL_NAMES}")
    return MODEL_REGISTRY[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_model(model: nn.Module, x: torch.Tensor, meta: Dict, device: torch.device) -> torch.Tensor:
    """Forward a DataLoader batch (x, meta) through any registered model."""
    appearance = meta.get("mean_frame")
    if appearance is not None:
        appearance = appearance.to(device)
    return model(x.to(device), appearance=appearance)


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
