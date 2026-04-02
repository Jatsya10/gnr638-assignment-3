from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    return x.mean(dim=1, keepdim=True)


class SpatialConsistencyLoss(nn.Module):
    """
    Preserve the relative contrast between neighboring regions.
    Implemented on 4x4 pooled grayscale regions.
    """

    def __init__(self, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, enhanced: torch.Tensor, input_img: torch.Tensor) -> torch.Tensor:
        y = _to_gray(enhanced)
        i = _to_gray(input_img)

        y_pool = F.avg_pool2d(y, kernel_size=self.patch_size, stride=self.patch_size)
        i_pool = F.avg_pool2d(i, kernel_size=self.patch_size, stride=self.patch_size)

        def neighbor_loss(t: torch.Tensor) -> torch.Tensor:
            losses = []
            if t.shape[2] > 1:
                diff_v = torch.abs(t[:, :, 1:, :] - t[:, :, :-1, :])
                losses.append(diff_v)
            if t.shape[3] > 1:
                diff_h = torch.abs(t[:, :, :, 1:] - t[:, :, :, :-1])
                losses.append(diff_h)
            if not losses:
                return torch.zeros((), device=t.device, dtype=t.dtype)
            return torch.cat([x.reshape(x.shape[0], -1) for x in losses], dim=1)

        dy_y = neighbor_loss(y_pool)
        dy_i = neighbor_loss(i_pool)

        # Same shape after concatenation; keep it differentiable and simple.
        if dy_y.numel() == 0:
            return torch.zeros((), device=enhanced.device, dtype=enhanced.dtype)
        return torch.mean((dy_y - dy_i) ** 2)


class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size: int = 16, target: float = 0.6):
        super().__init__()
        self.patch_size = patch_size
        self.target = target

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        y = _to_gray(enhanced)
        y_pool = F.avg_pool2d(y, kernel_size=self.patch_size, stride=self.patch_size)
        return torch.mean(torch.abs(y_pool - self.target))


class ColorConstancyLoss(nn.Module):
    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        mean_rgb = enhanced.mean(dim=(2, 3))
        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        return torch.mean((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2)


class IlluminationSmoothnessLoss(nn.Module):
    def forward(self, curve_maps: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(curve_maps[:, :, :, 1:] - curve_maps[:, :, :, :-1])
        dy = torch.abs(curve_maps[:, :, 1:, :] - curve_maps[:, :, :-1, :])
        return torch.mean(dx) + torch.mean(dy)


@dataclass
class LossWeights:
    tv: float = 200.0
    spa: float = 1.0
    color: float = 5.0
    exposure: float = 10.0


class ZeroDCELoss(nn.Module):
    def __init__(
        self,
        exposure_patch: int = 16,
        exposure_target: float = 0.6,
        spatial_patch: int = 4,
        weights: LossWeights | None = None,
    ):
        super().__init__()
        self.spatial = SpatialConsistencyLoss(spatial_patch)
        self.exposure = ExposureControlLoss(exposure_patch, exposure_target)
        self.color = ColorConstancyLoss()
        self.tv = IlluminationSmoothnessLoss()
        self.weights = weights or LossWeights()

    def forward(self, enhanced: torch.Tensor, input_img: torch.Tensor, curve_maps: torch.Tensor):
        loss_spa = self.spatial(enhanced, input_img)
        loss_exp = self.exposure(enhanced)
        loss_col = self.color(enhanced)
        loss_tv = self.tv(curve_maps)

        total = (
            self.weights.tv * loss_tv
            + self.weights.spa * loss_spa
            + self.weights.color * loss_col
            + self.weights.exposure * loss_exp
        )

        parts = {
            "total": total.detach(),
            "spa": loss_spa.detach(),
            "exp": loss_exp.detach(),
            "col": loss_col.detach(),
            "tv": loss_tv.detach(),
        }
        return total, parts
