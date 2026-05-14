"""
Train UNet segmentation on BraTS 2021 unhealthy slices.

Usage:
    python train.py \
        --data_root /mnt/apple/k66/minhdd/data/brats2021 \
        --split_file /mnt/apple/k66/minhdd/data/brats2021/preprocessed_split_old_val251.json \
        --output_dir ./output_unet \
        --epochs 50 --batch_size 16
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import BraTSSegDataset
from model import UNet


# ── Loss ─────────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce   = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        flat_p = prob.view(prob.size(0), -1)
        flat_t = targets.view(targets.size(0), -1)
        inter = (flat_p * flat_t).sum(1)
        dice = 1 - (2 * inter + self.smooth) / (flat_p.sum(1) + flat_t.sum(1) + self.smooth)
        return bce + dice.mean()


# ── Metrics ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics(loader, model, device, threshold=0.5):
    model.eval()
    dice_sum = iou_sum = n = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        pred = (torch.sigmoid(model(imgs)) > threshold).float()
        flat_p = pred.view(pred.size(0), -1)
        flat_t = masks.view(masks.size(0), -1)
        inter = (flat_p * flat_t).sum(1)
        union = flat_p.sum(1) + flat_t.sum(1) - inter
        dice_sum += (2 * inter / (flat_p.sum(1) + flat_t.sum(1) + 1e-6)).sum().item()
        iou_sum  += (inter / (union + 1e-6)).sum().item()
        n += pred.size(0)
    model.train()
    return dice_sum / n, iou_sum / n


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  default="/mnt/apple/k66/minhdd/data/brats2021")
    p.add_argument("--split_file", default="/mnt/apple/k66/minhdd/data/brats2021/preprocessed_split_old_val251.json")
    p.add_argument("--output_dir", default="./output_unet")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--base_ch",    type=int,   default=64)
    p.add_argument("--num_workers",type=int,   default=4)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--save_every", type=int,   default=5,
                   help="Save a numbered checkpoint every N epochs (0 = disabled)")
    p.add_argument("--resume",     default="")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = BraTSSegDataset(args.data_root, args.split_file, split="train")
    val_ds   = BraTSSegDataset(args.data_root, args.split_file, split="val")
    print(f"Train slices: {len(train_ds)}   Val slices: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Small val subset (~100 samples) for quick mid-train checks
    rng = torch.Generator().manual_seed(42)
    quick_val_idx = torch.randperm(len(val_ds), generator=rng)[:min(100, len(val_ds))].tolist()
    quick_val_loader = DataLoader(Subset(val_ds, quick_val_idx),
                                  batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

    model = UNet(in_channels=4, base_ch=args.base_ch,
                 gradient_checkpointing=args.gradient_checkpointing).to(device)
    if args.gradient_checkpointing:
        print("Gradient checkpointing: ON")
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_dice = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"Resumed from epoch {start_epoch}  best_dice={best_dice:.4f}")

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    csv_exists = os.path.isfile(csv_path)
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if not csv_exists:
        csv_w.writerow(["epoch", "train_loss", "val_dice", "val_iou", "lr"])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{args.epochs}", leave=False,
                    dynamic_ncols=True)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()

        avg_loss = total_loss / len(train_ds)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Quick val every 5 epochs, full val at end
        do_full_val = (epoch + 1 == args.epochs)
        do_quick_val = ((epoch + 1) % 5 == 0) and not do_full_val
        val_dice = val_iou = float("nan")
        if do_full_val:
            val_dice, val_iou = compute_metrics(val_loader, model, device)
            print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}  "
                  f"lr={lr_now:.2e}  t={elapsed:.0f}s  [FULL VAL]")
        elif do_quick_val:
            val_dice, val_iou = compute_metrics(quick_val_loader, model, device)
            print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}  "
                  f"lr={lr_now:.2e}  t={elapsed:.0f}s  [quick val n=100]")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"lr={lr_now:.2e}  t={elapsed:.0f}s")

        def _fmt(v): return f"{v:.6f}" if v == v else ""
        csv_w.writerow([epoch + 1, f"{avg_loss:.6f}",
                        _fmt(val_dice), _fmt(val_iou), f"{lr_now:.2e}"])
        csv_f.flush()

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.output_dir, "checkpoint_last.pth"))
        if val_dice == val_dice and val_dice > best_dice:  # not nan
            best_dice = val_dice
            ckpt["best_dice"] = best_dice
            torch.save(ckpt, os.path.join(args.output_dir, "checkpoint_best.pth"))
            print(f"  ↑ New best val_dice={best_dice:.4f}")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(ckpt, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1:03d}.pth"))

    csv_f.close()
    print(f"\nTraining done. Best val_dice={best_dice:.4f}")
    print(f"Checkpoints in {args.output_dir}/")


if __name__ == "__main__":
    main()
