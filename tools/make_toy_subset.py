from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zerodce.dataset import list_images  # noqa: E402


def luminance_score(path: Path) -> float:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(lum.mean())


def pick_balanced(paths: list[Path], scores: list[float], max_images: int, bins: int) -> list[Path]:
    order = np.argsort(scores)
    paths = [paths[i] for i in order]

    edges = np.linspace(0, len(paths), bins + 1, dtype=int)
    per_bin = max(1, math.ceil(max_images / bins))

    chosen: list[Path] = []
    for b in range(bins):
        segment = paths[edges[b] : edges[b + 1]]
        if not segment:
            continue
        if len(segment) <= per_bin:
            chosen.extend(segment)
        else:
            idx = np.linspace(0, len(segment) - 1, per_bin, dtype=int)
            chosen.extend(segment[i] for i in idx)

    # remove duplicates while preserving order
    unique: list[Path] = []
    seen: set[Path] = set()
    for p in chosen:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    if len(unique) > max_images:
        unique = unique[:max_images]

    if len(unique) < max_images:
        remaining = [p for p in paths if p not in seen]
        need = max_images - len(unique)
        if remaining:
            idx = np.linspace(0, len(remaining) - 1, min(need, len(remaining)), dtype=int)
            unique.extend(remaining[i] for i in idx)

    return unique[:max_images]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a balanced toy subset for Zero-DCE training.")
    p.add_argument("--src", type=str, required=True, help="Source folder containing the full training images.")
    p.add_argument("--dst", type=str, required=True, help="Destination folder for the toy subset.")
    p.add_argument("--max_images", type=int, default=256)
    p.add_argument("--bins", type=int, default=4, help="Brightness bins for balanced sampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    images = list_images(src)
    if not images:
        raise ValueError(f"No images found in {src}")

    scores = [luminance_score(p) for p in images]
    chosen = pick_balanced(images, scores, args.max_images, args.bins)

    manifest_path = dst / "toy_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "luminance"])
        writer.writeheader()

        for p in chosen:
            rel = p.resolve().relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)

            writer.writerow({"file": rel.as_posix(), "luminance": luminance_score(p)})

    print(f"Copied {len(chosen)} images to {dst}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()