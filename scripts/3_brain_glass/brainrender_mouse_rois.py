#!/usr/bin/env python3
"""
brainrender_mouse_rois.py
─────────────────────────
Standalone script to render mouse brain ROIs in 3 classic views
(sagittal, coronal, axial) and save a 3-panel publication figure.

Usage:
    conda activate funsy
    python brainrender_mouse_rois.py

Requirements:
    pip install brainrender bg-atlasapi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from scipy import ndimage

# ── Offscreen rendering (no display needed) ────────────────────────────────
import vedo
vedo.settings.default_backend = "offscreen"

from bg_atlasapi import BrainGlobeAtlas
from brainrender import Scene, settings
from brainrender.actors import Points

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — adjust paths here
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR    = "/home/samy/Bureau/vscode/net_fluidity"
LABELS_FILE = f"{BASE_DIR}/allen_roi_labels_41.npy"
OUTPUT_DIR  = f"{BASE_DIR}/figures/brainrender"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load atlas and define ROI → Allen acronym mapping
# ══════════════════════════════════════════════════════════════════════════════
print("Loading atlas...")
atlas = BrainGlobeAtlas("allen_mouse_25um")

anat_labels = np.load(LABELS_FILE)
print(f"Loaded {len(anat_labels)} ROI labels")

mapping = {
    'PL ILA':       ['PL', 'ILA'],
    'PFC':          ['PL', 'ILA', 'ORB', 'ACA'],
    'ACA':          ['ACA'],
    'RSP':          ['RSP'],
    'TEa':          ['TEa'],
    'd HIP':        ['CA1', 'CA3'],   # dorsal split applied below
    'v HIP':        ['CA1', 'CA3'],   # ventral split applied below
    'd DG':         ['DG'],
    'v DG':         ['DG'],
    'PERI':         ['PERI'],
    'ENT':          ['ENT'],
    'SUB':          ['SUB'],
    'CLA':          ['CLA'],
    'ReRh':         ['RE', 'RH'],
    'THAL memory':  ['MD', 'CM'],
    'THAL sensory': ['VPL', 'VPM', 'LP'],
    'THAL motor':   ['VAL', 'VM'],
    'Habenula':     ['LH', 'MH'],
    'Hypo medial':  ['DMH', 'VMH'],
    'Hypo SNC':     ['ZI'],
    'Hypo PVN-ARC': ['PVH', 'ARH'],
    'Hypo lat':     ['LHA'],
    'Septum':       ['LSr', 'MS'],
    'Insula':       ['AI', 'AIp'],
    'AUD':          ['AUDp', 'AUDv'],
    'PIR':          ['PIR'],
    'Motor':        ['MOp', 'MOs'],
    'Somato':       ['SSp', 'SSs'],
    'VIS':          ['VISp', 'VISl'],
    'PTLp':         ['PTLp'],
    'ACB':          ['ACB'],
    'CP':           ['CP'],
    'Pallidum':     ['PAL'],
    'Amygdala':     ['BLA', 'CEA', 'MEA'],
    'VTA':          ['VTA'],
    'MBmot PAG':    ['PAG'],
    'MBmot':        ['SCm'],
    'MBbeh':        ['PPN'],
    'MBsen':        ['SCsg'],
    'SN':           ['SNr', 'SNc'],
    'Pons':         ['PB'],
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute right-hemisphere centroids
# ══════════════════════════════════════════════════════════════════════════════
def get_centroid_right_hemi(acronyms, atlas, ventral=None):
    """Centroid using only right hemisphere voxels (ML > midpoint)."""
    coords = []
    ml_mid = atlas.annotation.shape[2] / 2

    for acr in acronyms:
        try:
            mask = atlas.get_structure_mask(acr)
            voxels = np.array(np.where(mask)).T  # (N,3): AP, DV, ML
            voxels = voxels[voxels[:, 2] > ml_mid]
            if len(voxels) > 0:
                coords.append(voxels)
        except Exception as e:
            print(f"  [skip] {acr}: {e}")

    if not coords:
        return None

    all_vox = np.concatenate(coords, axis=0)

    if ventral is not None:
        med_dv = np.median(all_vox[:, 1])
        all_vox = all_vox[all_vox[:, 1] >= med_dv] if ventral \
                  else all_vox[all_vox[:, 1] <  med_dv]

    return all_vox.mean(axis=0) * np.array(atlas.resolution)


centroids_path = f"{BASE_DIR}/allen_roi_centroids_41_rh.npy"

if os.path.exists(centroids_path):
    print("Loading cached right-hemisphere centroids...")
    centroid_array = np.load(centroids_path)
else:
    print("Computing right-hemisphere centroids (this takes ~2 min)...")
    centroids = {}
    for label, acronyms in mapping.items():
        print(f"  {label}...", end=" ", flush=True)
        if label == 'd HIP':
            centroids[label] = get_centroid_right_hemi(acronyms, atlas, ventral=False)
        elif label == 'v HIP':
            centroids[label] = get_centroid_right_hemi(acronyms, atlas, ventral=True)
        else:
            centroids[label] = get_centroid_right_hemi(acronyms, atlas)
        print("ok")

    centroid_array = np.array([centroids[l] for l in anat_labels])
    np.save(centroids_path, centroid_array)
    print(f"Saved centroids to {centroids_path}")

print(f"Centroid array shape: {centroid_array.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Network colors
# ══════════════════════════════════════════════════════════════════════════════
dmn_labels = ['PL ILA', 'PFC', 'ACA', 'RSP']
mem_labels = ['d HIP', 'v HIP', 'd DG', 'v DG', 'PERI', 'ENT',
              'SUB', 'ReRh', 'THAL memory']

def get_color(label):
    if label in dmn_labels: return [0.2, 0.4, 0.9]   # blue  — DMN
    if label in mem_labels: return [0.9, 0.2, 0.2]   # red   — Memory
    return                         [0.6, 0.6, 0.6]   # gray  — Other

node_colors = [get_color(l) for l in anat_labels]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Render 3 views
# ══════════════════════════════════════════════════════════════════════════════
# Camera: (position, focal_point, view_up)
# CCF coordinates in microns: (AP, DV, ML)
views = {
    "sagittal": dict(
        pos            = (3000, 3913, -49717),
        focal_point    = (7830, 4296, -5694),
        viewup         = (0.11, -0.99, -0.0),
        clipping_range = (31926, 59656),
        zoom           = 1.5,
    ),
    "coronal": dict(
        pos            = (-36455, 4631, -6154),
        focal_point    = (7830, 4296, -5694),
        viewup         = (-0.01, -1.0, -0.03),
        clipping_range = (29314, 60444),
        zoom           = 1.5,
    ),
    "axial": dict(
        pos            = (5980, -60225, 484),
        focal_point    = (7830, 4296, -5694),
        viewup         = (-1.0, 0.03, 0.03),
        clipping_range = (55000, 76301),
        zoom           = 1.5,
    ),
}

saved_paths = {}

# Global brainrender settings
settings.SHOW_AXES = False
settings.BACKGROUND_COLOR = "white"

for view_name, cam in views.items():
    print(f"Rendering {view_name}...")

    scene = Scene(atlas_name="allen_mouse_25um", title="")
    scene.add_brain_region("root", alpha=0.06, color="lightgray")
    # Style all brain region actors after adding
    for actor in scene.get_actors(br_class="brain region"):
        actor._mesh.lw(0.5)
    scene.add(Points(centroid_array, radius=200, colors=node_colors, alpha=1.0))

    # Build camera object and pass to render
    camera = {
        "pos":            cam["pos"],
        "focal_point":    cam["focal_point"],
        "viewup":         cam["viewup"],
        "clipping_range": cam["clipping_range"],
    }

    path = f"{OUTPUT_DIR}/brainrender_{view_name}.png"
    scene.render(
        interactive=False,
        camera=camera,
        zoom=cam["zoom"],
    )
    scene.screenshot(name=path, scale=3)
    saved_paths[view_name] = path
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Compose 3-panel figure
# ══════════════════════════════════════════════════════════════════════════════
print("Composing 3-panel figure...")

def smart_crop(img, pad=30):
    """Keep only the largest non-white region, crop to its bounding box."""
    gray = img[:, :, :3].mean(axis=2)
    mask = gray < 0.92
    labeled, n = ndimage.label(mask)
    if n == 0:
        return img
    sizes = ndimage.sum(mask, labeled, range(1, n+1))
    largest = np.argmax(sizes) + 1
    main_mask = labeled == largest
    rows = np.any(main_mask, axis=1)
    cols = np.any(main_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[max(0,rmin-pad):min(img.shape[0],rmax+pad),
               max(0,cmin-pad):min(img.shape[1],cmax+pad)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

from scipy import ndimage

for ax, (view_name, title) in zip(axes, [
    ("sagittal", "Sagittal"),
    ("coronal",  "Coronal"),
    ("axial",    "Axial (dorsal)"),
]):
    img = mpimg.imread(saved_paths[view_name])
    img_cropped = smart_crop(img)
    ax.imshow(img_cropped)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
    ax.axis('off')

# Legend
handles = [
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor=[0.2,0.4,0.9], markersize=14, label='DMN'),
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor=[0.9,0.2,0.2], markersize=14, label='Memory'),
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor=[0.6,0.6,0.6], markersize=14, label='Other'),
]
fig.legend(handles=handles, loc='lower center', ncol=3,
           fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
out_path = f"{OUTPUT_DIR}/brainrender_3views.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nDone! 3-panel figure saved to:\n  {out_path}")
