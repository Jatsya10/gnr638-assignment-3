# (imports unchanged)
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import csv
from skimage.metrics import structural_similarity as ssim

from .dataset import list_images
from .utils import ensure_dir, psnr, write_json


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def collect_images(root: Path):
    root = root.resolve()
    out = {}
    for p in list_images(root):
        p = p.resolve()
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.name
        out[rel] = p
    return out


# 🔥 NEW METRICS FUNCTION
def compute_all_metrics(a, b):
    mse = float(np.mean((a - b) ** 2))
    psnr_val = psnr(a, b)
    ssim_val = ssim(a, b, channel_axis=2, data_range=1.0)
    mae = float(np.mean(np.abs(a - b)))
    return mse, psnr_val, ssim_val, mae


# 🔥 VISUAL PANEL (kept)
def make_panel(original, official, ours):

    # convert to uint8
    orig = (original * 255).astype(np.uint8)
    off = (official * 255).astype(np.uint8)
    ours = (ours * 255).astype(np.uint8)

    i_orig = Image.fromarray(orig)
    i_off = Image.fromarray(off)
    i_ours = Image.fromarray(ours)

    # ensure same size
    h = max(i_orig.height, i_off.height, i_ours.height)

    def pad(im):
        if im.height == h:
            return im
        return im.resize((im.width, h))

    i_orig, i_off, i_ours = pad(i_orig), pad(i_off), pad(i_ours)

    # create panel
    label_h = 30
    panel = Image.new("RGB", (i_orig.width * 3, h + label_h), (20, 20, 20))

    panel.paste(i_orig, (0, label_h))
    panel.paste(i_off, (i_orig.width, label_h))
    panel.paste(i_ours, (i_orig.width * 2, label_h))

    # draw labels
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    draw.text((10, 5), "ORIGINAL", fill=(255, 255, 255), font=font)
    draw.text((i_orig.width + 10, 5), "OFFICIAL", fill=(255, 255, 255), font=font)
    draw.text((i_orig.width * 2 + 10, 5), "OURS", fill=(255, 255, 255), font=font)

    return panel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", required=True)
    p.add_argument("--official", required=True)
    p.add_argument("--original", required=True)  # NEW
    p.add_argument("--report_dir", default="comparison_report")
    p.add_argument("--save_panels", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    ours_dir = Path(args.ours)
    off_dir = Path(args.official)
    orig_dir = Path(args.original)

    report_dir = ensure_dir(args.report_dir)
    visual_dir = ensure_dir(report_dir / "visuals")

    ours_imgs = collect_images(ours_dir)
    off_imgs = collect_images(off_dir)
    orig_imgs = collect_images(orig_dir)

    common = sorted(set(ours_imgs) & set(off_imgs) & set(orig_imgs))

    if not common:
        raise ValueError("No matching files across all folders")

    rows = []

    stats = {
        "ours_vs_official": [],
        "ours_vs_original": [],
        "official_vs_original": []
    }

    for rel in common:
        ours = load_rgb(ours_imgs[rel])
        off = load_rgb(off_imgs[rel])
        orig = load_rgb(orig_imgs[rel])

        # --- compute all ---
        m1, p1, s1, a1 = compute_all_metrics(ours, off)
        m2, p2, s2, a2 = compute_all_metrics(ours, orig)
        m3, p3, s3, a3 = compute_all_metrics(off, orig)

        stats["ours_vs_official"].append((m1, p1, s1, a1))
        stats["ours_vs_original"].append((m2, p2, s2, a2))
        stats["official_vs_original"].append((m3, p3, s3, a3))

        rows.append({
            "file": rel,

            # ours vs official
            "psnr_ours_vs_official": p1,
            "ssim_ours_vs_official": s1,
            "mae_ours_vs_official": a1,

            # ours vs original
            "psnr_ours_vs_original": p2,
            "ssim_ours_vs_original": s2,
            "mae_ours_vs_original": a2,

            # official vs original
            "psnr_official_vs_original": p3,
            "ssim_official_vs_original": s3,
            "mae_official_vs_original": a3,
        })

        if args.save_panels:
            panel = make_panel(orig, off, ours)
            out_path = visual_dir / (Path(rel).with_suffix(".png"))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(out_path)

    # 🔥 summary
    def avg(idx, key):
        return float(np.mean([x[idx] for x in stats[key]]))

    summary = {
        "count": len(common),

        "ours_vs_official": {
            "psnr": avg(1, "ours_vs_official"),
            "ssim": avg(2, "ours_vs_official"),
            "mae": avg(3, "ours_vs_official"),
        },

        "ours_vs_original": {
            "psnr": avg(1, "ours_vs_original"),
            "ssim": avg(2, "ours_vs_original"),
            "mae": avg(3, "ours_vs_original"),
        },

        "official_vs_original": {
            "psnr": avg(1, "official_vs_original"),
            "ssim": avg(2, "official_vs_original"),
            "mae": avg(3, "official_vs_original"),
        },
    }

    # save csv
    with (report_dir / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    write_json(report_dir / "summary.json", summary)

    print("\n=== FINAL COMPARISON ===")
    print(summary)
    print(f"\nSaved to: {report_dir}")


if __name__ == "__main__":
    main()