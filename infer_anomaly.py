"""
UNet segmentation inference on BraTS 2021 test set.

Pipeline:
    * Forward each unhealthy slice through the trained UNet → binary mask.
    * Metrics: DICE, IoU, AUROC per slice; pooled best-DICE threshold sweep.
    * Healthy slices: just report they have no tumor (no metrics needed).
    * Output: metrics.csv, summary.txt, optional PNG per sample.

Usage:
    python infer_anomaly.py \
        --checkpoint ./output_unet/checkpoint_best.pth \
        --data_root  /mnt/apple/k66/minhdd/data/brats2021 \
        --split_file /mnt/apple/k66/minhdd/data/brats2021/preprocessed_split.json \
        --output_dir ./anomaly_results_unet \
        --split      test
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from dataset import BraTSSegDataset
from model import UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_root",   default="/mnt/apple/k66/minhdd/data/brats2021")
    p.add_argument("--split_file",  default="/mnt/apple/k66/minhdd/data/brats2021/preprocessed_split_old_val251.json")
    p.add_argument("--output_dir",  default="./anomaly_results_unet")
    p.add_argument("--split",       default="test")
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--threshold_steps", type=int, default=200)
    p.add_argument("--save_png",    action="store_true")
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def dice_score(pred, gt, smooth=1e-6):
    inter = (pred * gt).sum()
    return (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def iou_score(pred, gt, smooth=1e-6):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + smooth) / (union + smooth)


def save_png(arr_01, path):
    img = (arr_01 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet(in_channels=4, base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')+1}")

    ds = BraTSSegDataset(args.data_root, args.split_file, split=args.split)
    print(f"Evaluating {len(ds)} unhealthy slices from '{args.split}' split")

    all_probs = []
    all_gt    = []
    rows = []

    with torch.no_grad():
        for idx in range(len(ds)):
            img, mask = ds[idx]
            prob = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            gt   = mask[0].numpy()

            all_probs.append(prob.ravel())
            all_gt.append(gt.ravel().astype(np.uint8))

            pred_bin = (prob > args.threshold).astype(np.float32)
            d = dice_score(pred_bin, gt)
            i = iou_score(pred_bin, gt)

            try:
                auroc = roc_auc_score(gt.ravel().astype(int), prob.ravel())
            except ValueError:
                auroc = float("nan")

            rows.append({"idx": idx, "dice": d, "iou": i, "auroc": auroc})

            if args.save_png and idx < 200:
                out = Path(args.output_dir) / f"sample_{idx:04d}"
                out.mkdir(exist_ok=True)
                save_png(img[0].numpy(), str(out / "T1.png"))
                save_png(prob,            str(out / "prob_map.png"))
                save_png(pred_bin,        str(out / "pred_mask.png"))
                save_png(gt,              str(out / "gt_mask.png"))

            if (idx + 1) % 500 == 0:
                print(f"  {idx+1}/{len(ds)}")

    # ── Pooled threshold sweep ────────────────────────────────────────────────
    all_probs_flat = np.concatenate(all_probs)
    all_gt_flat    = np.concatenate(all_gt)
    thresholds = np.linspace(0, 1, args.threshold_steps + 1)
    best_dice, best_thr = 0.0, 0.5
    for thr in thresholds:
        pred = (all_probs_flat > thr).astype(np.float32)
        d = dice_score(pred, all_gt_flat)
        if d > best_dice:
            best_dice, best_thr = d, thr

    try:
        ap = average_precision_score(all_gt_flat, all_probs_flat)
    except Exception:
        ap = float("nan")
    try:
        auroc_pool = roc_auc_score(all_gt_flat, all_probs_flat)
    except Exception:
        auroc_pool = float("nan")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "dice", "iou", "auroc"])
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in r.items()})

    mean_dice  = np.mean([r["dice"]  for r in rows])
    mean_iou   = np.mean([r["iou"]   for r in rows])
    mean_auroc = np.nanmean([r["auroc"] for r in rows])

    summary = (
        f"UNet Segmentation — {args.split} split\n"
        f"Checkpoint:   {args.checkpoint}\n"
        f"Threshold:    {args.threshold}\n"
        f"Slices:       {len(rows)}\n"
        f"\n--- Per-slice (threshold={args.threshold}) ---\n"
        f"Mean DICE:    {mean_dice:.4f}\n"
        f"Mean IoU:     {mean_iou:.4f}\n"
        f"Mean AUROC:   {mean_auroc:.4f}\n"
        f"\n--- Pooled (all pixels) ---\n"
        f"Best DICE:    {best_dice:.4f}  (threshold={best_thr:.4f})\n"
        f"AP (AUPRC):   {ap:.4f}\n"
        f"AUROC:        {auroc_pool:.4f}\n"
    )
    print(summary)
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(summary)

    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
