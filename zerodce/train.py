from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LowLightFolderDataset
from .losses import ZeroDCELoss, LossWeights
from .model import ZeroDCE, ZeroDCEConfig, weights_init
from .utils import append_csv_row, ensure_dir, set_seed, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Zero-DCE from scratch on a folder of low-light images.")
    p.add_argument("--data_dir", type=str, default="data/toy_train_data", help="Folder containing training images.")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1143)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="runs/zerodce")
    p.add_argument("--save_every_epochs", type=int, default=1)

    p.add_argument("--augment_flip", dest="augment_flip", action="store_true")
    p.add_argument("--no_augment_flip", dest="augment_flip", action="store_false")
    p.set_defaults(augment_flip=True)

    p.add_argument("--random_crop", dest="random_crop", action="store_true")
    p.add_argument("--no_random_crop", dest="random_crop", action="store_false")
    p.set_defaults(random_crop=True)

    p.add_argument("--crop_scale_max", type=float, default=1.35)

    # Loss weights: start close to the paper, but a slightly higher exposure weight
    # often helps a 5-epoch toy run brighten faster.
    p.add_argument("--tv_weight", type=float, default=150.0)
    p.add_argument("--spa_weight", type=float, default=1.0)
    p.add_argument("--color_weight", type=float, default=5.0)
    p.add_argument("--exp_weight", type=float, default=12.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    save_dir = ensure_dir(args.save_dir)
    ckpt_dir = ensure_dir(save_dir / "checkpoints")
    sample_dir = ensure_dir(save_dir / "samples")

    dataset = LowLightFolderDataset(
        args.data_dir,
        image_size=args.image_size,
        augment_flip=args.augment_flip,
        random_crop=args.random_crop,
        crop_scale_max=args.crop_scale_max,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device(args.device)

    model = ZeroDCE(ZeroDCEConfig(iterations=8))
    model.apply(weights_init)
    model = model.to(device)

    criterion = ZeroDCELoss(
        weights=LossWeights(
            tv=args.tv_weight,
            spa=args.spa_weight,
            color=args.color_weight,
            exposure=args.exp_weight,
        )
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.1,
    )

    config_dump = vars(args).copy()
    config_dump["num_images"] = len(dataset)
    write_json(save_dir / "config.json", config_dump)

    log_path = save_dir / "train_log.csv"
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "spa": 0.0, "exp": 0.0, "col": 0.0, "tv": 0.0}
        count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        last_batch = None
        last_enhanced = None

        for batch_idx, batch in enumerate(pbar, start=1):
            batch = batch.to(device, non_blocking=True)

            inter, enhanced, curve_maps = model(batch)
            loss, parts = criterion(enhanced, batch, curve_maps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            bs = batch.size(0)
            count += bs
            global_step += 1
            last_batch = batch
            last_enhanced = enhanced

            for k in running:
                running[k] += float(parts[k]) * bs

            pbar.set_postfix({k: f"{running[k] / count:.4f}" for k in running})

            if batch_idx % 25 == 0:
                append_csv_row(
                    log_path,
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "batch": batch_idx,
                        "total": float(parts["total"]),
                        "spa": float(parts["spa"]),
                        "exp": float(parts["exp"]),
                        "col": float(parts["col"]),
                        "tv": float(parts["tv"]),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    ["epoch", "step", "batch", "total", "spa", "exp", "col", "tv", "lr"],
                )

        scheduler.step()

        if epoch % args.save_every_epochs == 0:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": vars(args),
            }
            torch.save(ckpt, ckpt_dir / f"zerodce_epoch_{epoch:03d}.pt")

        if last_batch is not None and last_enhanced is not None:
            from .utils import save_image_tensor

            with torch.no_grad():
                save_image_tensor(last_batch[0].cpu(), sample_dir / f"epoch_{epoch:03d}_input.png")
                save_image_tensor(last_enhanced[0].cpu(), sample_dir / f"epoch_{epoch:03d}_enhanced.png")

        epoch_summary = {k: running[k] / max(count, 1) for k in running}
        append_csv_row(
            log_path,
            {
                "epoch": epoch,
                "step": global_step,
                "batch": -1,
                "total": epoch_summary["total"],
                "spa": epoch_summary["spa"],
                "exp": epoch_summary["exp"],
                "col": epoch_summary["col"],
                "tv": epoch_summary["tv"],
                "lr": optimizer.param_groups[0]["lr"],
            },
            ["epoch", "step", "batch", "total", "spa", "exp", "col", "tv", "lr"],
        )

        print(f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in epoch_summary.items()))

    print(f"Done. Checkpoints: {ckpt_dir}")
    print(f"Logs: {log_path}")


if __name__ == "__main__":
    main()