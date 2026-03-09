#!/usr/bin/env python3
"""
compose_3views.py
─────────────────
Recompose the 3-panel brainrender figure from existing PNGs.
Run this independently without re-rendering brainrender.

Usage:
    conda activate funsy
    python compose_3views.py
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from scipy import ndimage
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths

from shared_code.fun_utils import (
    load_cognitive_data,
    set_figure_params,
)

#%%
save_fig = set_figure_params(False)
timecourse_folder = "Timecourses_updated_03052024"

paths = get_paths(
    dataset_name="ines_abdallah",
    timecourse_folder=timecourse_folder,
    cognitive_data_file="ROIs.xlsx",
    anat_labels_file="41_Allen.txt",
)

bundle = load_timeseries_bundle(
    paths["preprocessed"] / "ts_and_meta_2m4m.npz",
    paths["preprocessed"] / "grouping_data_oip.pkl",
)
ts = bundle.ts
n_animals = bundle.n_animals
total_tr = bundle.total_tr
anat_labels = bundle.anat_labels
regions = bundle.n_regions
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables


#%%
OUTPUT_DIR = "/home/samy/Bureau/vscode/net_fluidity/figures/brainrender"

# ── Smart crop: keep only largest non-white region ─────────────────────────
def smart_crop(img, pad=40, threshold=0.92):
    gray = img[:, :, :3].mean(axis=2)
    mask = gray < threshold
    labeled, n = ndimage.label(mask)
    if n == 0:
        return img
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    main_mask = labeled == largest
    rows = np.any(main_mask, axis=1)
    cols = np.any(main_mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img[max(0, r0-pad):min(img.shape[0], r1+pad),
               max(0, c0-pad):min(img.shape[1], c1+pad)]

# ── Pad image to target size (centered, white background) ─────────────────
def pad_to_size(img, target_h, target_w):
    h, w = img.shape[:2]
    has_alpha = img.shape[2] == 4
    canvas = np.ones((target_h, target_w, img.shape[2]), dtype=img.dtype)
    r_off = (target_h - h) // 2
    c_off = (target_w - w) // 2
    canvas[r_off:r_off+h, c_off:c_off+w] = img
    return canvas

# ── Load and crop all views ────────────────────────────────────────────────
views = [
    ("sagittal", "Sagittal"),
    ("coronal",  "Coronal"),
    ("axial",    "Axial (dorsal)"),
]

cropped = {}
for view_name, _ in views:
    img = mpimg.imread(f"{OUTPUT_DIR}/brainrender_{view_name}.png")
    # Ensure RGBA
    if img.shape[2] == 3:
        alpha = np.ones((*img.shape[:2], 1), dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
    cropped[view_name] = smart_crop(img)

# Normalize to same canvas size
max_h = max(img.shape[0] for img in cropped.values())
max_w = max(img.shape[1] for img in cropped.values())

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

for ax, (view_name, title) in zip(axes, views):
    img_padded = pad_to_size(cropped[view_name], max_h, max_w)
    ax.imshow(img_padded)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    ax.axis('off')

# ── Legend ─────────────────────────────────────────────────────────────────
handles = [
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor=[0.2, 0.4, 0.9], markersize=14, label='DMN'),
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor=[0.9, 0.2, 0.2], markersize=14, label='Memory'),
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor=[0.6, 0.6, 0.6], markersize=14, label='Other'),
]
fig.legend(handles=handles, loc='lower center', ncol=3,
           fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out_path = f"{OUTPUT_DIR}/brainrender_3views_final.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")

# %%
