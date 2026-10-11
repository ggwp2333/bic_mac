"""Build the BIC-MAC head-crop cache that the head cVAE fine-tune trains on.

One compressed ``.npz`` per subject holding the head crop of the four inputs
(NAC-PET, MRI in/out phase, topogram), the per-crop normalisation statistics,
the CT target and the loss/scoring masks, plus the nominal (un-clamped) and
actual boxes. The box is anchored on the 3-D centroid of
``features/mri_face_mask.nii.gz``:

    CT box   = [centroid - NEG_VOX, centroid + POS_VOX]           (130, 170, 120)
    cond box = CT box grown by COND_MARGIN voxels on every side   (170, 210, 160)

The conditioning box is larger because the MRI/PET are not perfectly
registered to the CT. Near the skull crown the box runs off the top of the
volume for almost every subject; the trainer places each crop at
``actual_lo - nominal_lo`` in its canvas so anatomy stays aligned across
subjects, and the missing slices are padding that the loss mask ignores.

This is the cache-building cell of the original training notebook, unchanged
apart from the paths and the command line. ``finetune/train_vae_ddp.py`` and
``pretrain_synthrad/build_cache.py`` (which fits its SynthRAD anchor offset
against these boxes) both read it from ``head_cache_bicmac/`` by default.

Usage:
    python build_head_cache.py                    # 75 training subjects
    python build_head_cache.py --with-val         # + the 4 CT-free val subjects
"""

from __future__ import annotations

import argparse
import os
import time

import nibabel as nib
import numpy as np
from tqdm.auto import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BIC_MAC_DATA_ROOT",
                      os.path.join(BASE_DIR, "..", "..", "data"))
DATA_DIR = os.path.join(ROOT, "bic-mac-data", "train")
VAL_DIR = os.path.join(ROOT, "bic-mac-data", "val")
CACHE_DIR = os.path.join(BASE_DIR, "head_cache_bicmac")

# ---- crop geometry -----------------------------------------------------------
# +x=left  +y=anterior/face  +z=superior(up)
POS_VOX     = (65, 70, 110)
NEG_VOX     = (65, 100, 10)
COND_MARGIN = (20, 20, 20)      # extra voxels per side for MRI / NAC-PET / topogram

CT_KEY    = "ct-label/ct.nii.gz"
COND_KEYS = ["features/nacpet.nii.gz",
             "features/mri_combined_in_phase.nii.gz",
             "features/mri_combined_out_phase.nii.gz"]      # topogram handled separately (X,1,Z)
TOPO_KEY  = "features/topogram.nii.gz"
PMASK_KEY = "ct-label/prediction_mask.nii.gz"
BMASK_KEY = "ct-label/body_seg.nii.gz"
REF_KEY   = "features/nacpet.nii.gz"


def list_subjects(root):
    return sorted(s for s in os.listdir(root)
                  if s.startswith("sub-") and os.path.isdir(os.path.join(root, s)))


def mask_centroid(subject_dir):
    """3-D centroid (voxel coords) of the face mask + the volume shape."""
    fm = nib.load(os.path.join(subject_dir, "features", "mri_face_mask.nii.gz")).get_fdata() > 0
    c = []
    for ax in range(3):
        proj = fm.sum(axis=tuple(i for i in range(3) if i != ax)).astype(np.float64)
        c.append((np.arange(fm.shape[ax]) * proj).sum() / max(proj.sum(), 1))
    return np.array(c), fm.shape


def boxes_for(subject_dir):
    """Return nominal (un-clamped) and actual CT / cond boxes as (3,2) int arrays."""
    center, shape = mask_centroid(subject_dir)
    ci = np.round(center).astype(np.int64)
    ct_nom   = np.stack([ci - np.array(NEG_VOX), ci + np.array(POS_VOX)], axis=1)
    cond_nom = ct_nom + np.stack([-np.array(COND_MARGIN), np.array(COND_MARGIN)], axis=1)
    def clamp(box):
        out = box.copy()
        for a in range(3):
            out[a, 0] = max(int(box[a, 0]), 0)
            out[a, 1] = min(int(box[a, 1]), shape[a])
        return out
    return ct_nom, clamp(ct_nom), cond_nom, clamp(cond_nom)


def _sl(box): return tuple(slice(int(b[0]), int(b[1])) for b in box)


def _stats(vol, nonzero):
    """MONAI NormalizeIntensityd stats: (subtrahend, divisor)."""
    v = vol[vol != 0] if nonzero else vol
    if v.size == 0:
        return 0.0, 1.0
    sub = float(v.mean()) if nonzero else 0.0     # nacpet uses subtrahend=[0], like the body network
    div = float(v.std())
    return sub, (div if div > 1e-8 else 1.0)


def cache_subject(name, root, with_ct=True, overwrite=False):
    out_path = os.path.join(CACHE_DIR, f"{name}.npz")
    if os.path.exists(out_path) and not overwrite:
        return out_path
    sub_dir = os.path.join(root, name)
    ct_nom, ct_box, cond_nom, cond_box = boxes_for(sub_dir)
    cond_sl, ct_sl = _sl(cond_box), _sl(ct_box)

    cond = np.stack([nib.load(os.path.join(sub_dir, k)).get_fdata(dtype=np.float32)[cond_sl]
                     for k in COND_KEYS], 0)                                   # (3, X, Y, Z)
    topo = nib.load(os.path.join(sub_dir, TOPO_KEY)).get_fdata(dtype=np.float32)
    topo = topo[cond_sl[0], :, cond_sl[2]]                                     # (X, 1, Z) -> may squeeze
    topo = topo.reshape(cond.shape[1], 1, cond.shape[3])

    norm = np.array([_stats(cond[0], nonzero=False),      # nacpet: subtrahend 0, /std
                     _stats(cond[1], nonzero=True),       # MRI in : z-score over nonzero
                     _stats(cond[2], nonzero=True),       # MRI out: z-score over nonzero
                     _stats(topo,    nonzero=True)], dtype=np.float32)         # (4, 2)

    data = dict(cond=cond.astype(np.float16), topo=topo.astype(np.float16), norm=norm,
                ct_box=ct_box.astype(np.int32),   ct_box_nom=ct_nom.astype(np.int32),
                cond_box=cond_box.astype(np.int32), cond_box_nom=cond_nom.astype(np.int32),
                affine=nib.load(os.path.join(sub_dir, REF_KEY)).affine.astype(np.float32))
    if with_ct:
        data["ct"]    = nib.load(os.path.join(sub_dir, CT_KEY)).get_fdata(dtype=np.float32)[ct_sl].astype(np.float16)
        data["pmask"] = (nib.load(os.path.join(sub_dir, PMASK_KEY)).get_fdata()[ct_sl] > 0).astype(np.uint8)
        data["bmask"] = (nib.load(os.path.join(sub_dir, BMASK_KEY)).get_fdata()[ct_sl] > 0).astype(np.uint8)
    np.savez_compressed(out_path, **data)
    return out_path


def main() -> None:
    global CACHE_DIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", default=DATA_DIR)
    ap.add_argument("--val-dir", default=VAL_DIR)
    ap.add_argument("--out-dir", default=CACHE_DIR)
    ap.add_argument("--with-val", action="store_true",
                    help="also cache the CT-free challenge validation subjects "
                         "(the trainer skips records without a CT)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    CACHE_DIR = a.out_dir
    os.makedirs(CACHE_DIR, exist_ok=True)

    t0 = time.time()
    subs = list_subjects(a.train_dir)
    for s in tqdm(subs, desc="cache train"):
        cache_subject(s, a.train_dir, with_ct=True, overwrite=a.overwrite)
    print(f"cached {len(subs)} subjects in {time.time() - t0:.0f}s -> {CACHE_DIR}")

    if a.with_val:
        for s in tqdm(list_subjects(a.val_dir), desc="cache val"):
            cache_subject(s, a.val_dir, with_ct=False, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
