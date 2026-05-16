"""
UNet segmentation inference on BraTS 2021 test set.

Pipeline:
    * Forward each unhealthy slice through the trained UNet → binary mask.
    * Metrics: DICE, IoU, AUROC per slice; pooled best-DICE threshold sweep.
    * Healthy slices: just report they have no tumor (no metrics needed).
    * Output: metrics.csv, summary.txt, PNG grids per sample (input + output).

Usage:
    python infer_anomaly.py \
        --checkpoint ./output_unet/checkpoint_best.pth \
        --data_root  /workspace/data/brats2021 \
        --split_file /workspace/preprocessed_split_train_val_test.json \
        --output_dir ./anomaly_results_unet \
        --split      test
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, roc_auc_score

from dataset import BraTSSegDataset
from model import UNet

MODALITY_NAMES = ["T1", "T1CE", "T2", "FLAIR"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_root",   default="/workspace/data/brats2021")
    p.add_argument("--split_file",  default="/workspace/preprocessed_split_train_val_test.json")
    p.add_argument("--output_dir",  default="./anomaly_results_unet")
    p.add_argument("--split",       default="test")
    p.add_argument("--base_ch",     type=int,   default=64)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--threshold_steps", type=int, default=200)
    p.add_argument("--no_save_png", action="store_true", help="disable PNG output")
    p.add_argument("--max_save",    type=int, default=-1,
                   help="max PNG grids to save (-1 = all)")
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def dice_score(pred, gt, smooth=1e-6):
    inter = (pred * gt).sum()
    return (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def iou_score(pred, gt, smooth=1e-6):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter + smooth) / (union + smooth)


def to_uint8(arr):
    return (arr * 255).clip(0, 255).astype(np.uint8)


def save_grid(img_4ch, prob, pred_bin, gt, slice_path, metrics, out_path):
    """Save a single-row grid: [T1 | T1CE | T2 | FLAIR | prob_map | pred_mask | gt_mask].
    Each tile is 256×256. A header row shows the slice path and metrics.
    """
    H, W = 256, 256
    HEADER = 20
    n_tiles = 7
    canvas = Image.new("RGB", (W * n_tiles, H + HEADER), color=(20, 20, 20))

    tiles = [
        Image.fromarray(to_uint8(img_4ch[c])).convert("RGB")
        for c in range(4)
    ]
    # prob map: grayscale → green tint
    prob_gray = to_uint8(prob)
    prob_rgb  = np.stack([np.zeros_like(prob_gray), prob_gray, np.zeros_like(prob_gray)], axis=-1)
    tiles.append(Image.fromarray(prob_rgb.astype(np.uint8)))
    # pred mask: white on black
    tiles.append(Image.fromarray(to_uint8(pred_bin)).convert("RGB"))
    # gt mask: red on black
    gt_u8  = to_uint8(gt)
    gt_rgb = np.stack([gt_u8, np.zeros_like(gt_u8), np.zeros_like(gt_u8)], axis=-1)
    tiles.append(Image.fromarray(gt_rgb.astype(np.uint8)))

    for i, tile in enumerate(tiles):
        canvas.paste(tile.resize((W, H), Image.NEAREST), (i * W, HEADER))

    # Header text
    draw = ImageDraw.Draw(canvas)
    label_names = MODALITY_NAMES + ["prob", "pred", "GT"]
    for i, name in enumerate(label_names):
        draw.text((i * W + 2, 2), name, fill=(200, 200, 200))

    d, iou, auroc = metrics
    header_txt = (
        f"{slice_path}   "
        f"DICE={d:.3f}  IoU={iou:.3f}  AUROC={auroc:.3f}"
    )
    draw.text((0, 10), header_txt, fill=(255, 220, 50))

    canvas.save(str(out_path))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    save_png = not args.no_save_png
    if save_png:
        png_dir = Path(args.output_dir) / "images"
        png_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet(in_channels=4, base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')+1}")

    ds = BraTSSegDataset(args.data_root, args.split_file, split=args.split)
    print(f"Evaluating {len(ds)} unhealthy slices from '{args.split}' split")
    if save_png:
        n_saved = len(ds) if args.max_save < 0 else args.max_save
        print(f"Saving PNG grids for up to {n_saved} slices → {png_dir}")

    all_probs = []
    all_gt    = []
    rows = []
    saved = 0

    with torch.no_grad():
        for idx in range(len(ds)):
            img, mask = ds[idx]
            prob = torch.sigmoid(model(img.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
            gt   = mask[0].numpy()

            all_probs.append(prob.ravel())
            all_gt.append(gt.ravel().astype(np.uint8))

            pred_bin = (prob > args.threshold).astype(np.float32)
            d = dice_score(pred_bin, gt)
            iou = iou_score(pred_bin, gt)

            try:
                auroc = roc_auc_score(gt.ravel().astype(int), prob.ravel())
            except ValueError:
                auroc = float("nan")

            slice_path = ds.samples[idx]["path"]
            rows.append({"idx": idx, "slice": slice_path,
                          "dice": d, "iou": iou, "auroc": auroc})

            if save_png and (args.max_save < 0 or saved < args.max_save):
                grid_path = png_dir / f"{idx:05d}.png"
                save_grid(
                    img.numpy(), prob, pred_bin, gt,
                    slice_path, (d, iou, auroc),
                    grid_path,
                )
                saved += 1

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
        w = csv.DictWriter(f, fieldnames=["idx", "slice", "dice", "iou", "auroc"])
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
