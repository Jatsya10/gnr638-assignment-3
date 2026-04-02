from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .dataset import list_images
from .model import ZeroDCE, ZeroDCEConfig
from .utils import ensure_dir, load_image, save_image_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhance images with a trained Zero-DCE checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/enhanced")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image_size", type=int, default=None, help="Optional resize before inference.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = ensure_dir(args.output_dir)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ZeroDCE(ZeroDCEConfig(iterations=8)).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    input_root = Path(args.input_dir).resolve()
    images = list_images(input_root)
    if not images:
        raise ValueError(f"No images found in {args.input_dir}")

    with torch.no_grad():
        for img_path in images:
            tensor = load_image(img_path)
            if args.image_size is not None:
                # simple resize at inference time if requested
                import torch.nn.functional as F
                tensor = tensor.unsqueeze(0)
                tensor = F.interpolate(tensor, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
                tensor = tensor.squeeze(0)

            tensor = tensor.unsqueeze(0).to(device)
            _, enhanced, _ = model(tensor)

            rel = img_path.resolve().relative_to(input_root)
            save_image_tensor(enhanced.squeeze(0).cpu(), out_dir / rel)

    print(f"Saved enhanced images to: {out_dir}")


if __name__ == "__main__":
    main()
