"""
BraTS 2021 slice-level segmentation dataset.
Reads the preprocessed_split.json produced by create_brats_split.py.
Each unhealthy slice has a paired *_seg.npy with shape (1, 256, 256).
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSSegDataset(Dataset):
    """
    Returns (image, mask) pairs where image is (4, 256, 256) float32 in [0,1]
    and mask is (1, 256, 256) float32 binary.

    Only unhealthy slices are used (they have ground-truth masks).
    """

    def __init__(self, data_root: str, split_file: str, split: str = "train"):
        with open(split_file) as f:
            data = json.load(f)

        # "test" falls back to "val" if absent (e.g. preprocessed_split_old_val251.json)
        if split not in data and split == "test" and "val" in data:
            split = "val"
        entries = data[split]
        # keep only unhealthy slices (they have seg files)
        self.samples = [e for e in entries if e["path"].startswith("unhealthy/")]
        self.data_root = data_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel = self.samples[idx]["path"]          # unhealthy/BraTS2021_XXXXX/slice_YYY.npy
        img_path = os.path.join(self.data_root, rel)
        seg_path = img_path.replace(".npy", "_seg.npy")

        image = torch.from_numpy(np.load(img_path).astype(np.float32))  # (4,256,256)
        mask  = torch.from_numpy(np.load(seg_path).astype(np.float32))  # (1,256,256)
        return image, mask
