from __future__ import annotations

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    paths: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
    paths.sort()
    return paths


class LowLightFolderDataset(Dataset):
    """
    Better training dataset for 5-epoch toy runs:
    - random resized crop instead of plain resize
    - optional random horizontal flip
    - works on any folder of images
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        augment_flip: bool = True,
        random_crop: bool = True,
        crop_scale_max: float = 1.35,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.augment_flip = augment_flip
        self.random_crop = random_crop
        self.crop_scale_max = crop_scale_max
        self.files = list_images(self.root)

        if not self.files:
            raise ValueError(f"No image files found in: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def _random_training_view(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        if not self.random_crop:
            return img.resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)

        # Upscale first so we can crop different patches from the same image.
        min_side = min(w, h)
        base_scale = max(self.image_size / float(min_side), 1.0)
        extra_scale = random.uniform(1.0, self.crop_scale_max)
        scale = base_scale * extra_scale

        new_w = max(self.image_size, int(round(w * scale)))
        new_h = max(self.image_size, int(round(h * scale)))
        img = img.resize((new_w, new_h), RESAMPLE_BILINEAR)

        if new_w == self.image_size and new_h == self.image_size:
            return img

        left = random.randint(0, max(0, new_w - self.image_size))
        top = random.randint(0, max(0, new_h - self.image_size))
        return img.crop((left, top, left + self.image_size, top + self.image_size))

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self._random_training_view(img)

        if self.augment_flip and random.random() < 0.5:
            img = img.transpose(FLIP_LEFT_RIGHT)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor