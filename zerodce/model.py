from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ZeroDCEConfig:
    in_channels: int = 3
    features: int = 32
    iterations: int = 8


class ZeroDCE(nn.Module):
    """
    Zero-DCE / DCE-Net from the paper:
    - 7 convolution layers
    - symmetric concatenation
    - no pooling / no batch norm
    - last layer outputs 24 channels = 8 x 3 curve maps
    """

    def __init__(self, config: ZeroDCEConfig | None = None):
        super().__init__()
        self.config = config or ZeroDCEConfig()
        f = self.config.features
        self.iterations = self.config.iterations
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(self.config.in_channels, f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(f * 2, f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(f * 2, f, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(f * 2, self.iterations * 3, kernel_size=3, stride=1, padding=1, bias=True)

    def _apply_curve(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return x + a * (x * x - x)

    def forward(self, x: torch.Tensor):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))

        curve_maps = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        alphas = torch.chunk(curve_maps, self.iterations, dim=1)

        enhanced = x
        intermediate = None
        for i, alpha in enumerate(alphas):
            enhanced = self._apply_curve(enhanced, alpha)
            if i == 3:
                intermediate = enhanced

        if intermediate is None:
            intermediate = enhanced
        return intermediate, enhanced, curve_maps


def weights_init(module: nn.Module) -> None:
    """Match the official repo's simple normal initialization."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
