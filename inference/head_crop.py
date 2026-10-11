"""Head crop geometry + conditioning canvas for the head cVAE.

Training reads one precomputed .npz per subject
(`training/head/build_head_cache.py`) and builds the network canvas from it. A
container gets one subject at a time and no cache directory, so the same steps
run inline here; the arithmetic is unchanged.
"""

import math

import nibabel as nib
import numpy as np

# ---- crop geometry (verbatim: notebook Cell 1) -----------------------------
# +x = left, +y = anterior/face, +z = superior
POS_VOX = (65, 70, 110)
NEG_VOX = (65, 100, 10)
COND_MARGIN = (20, 20, 20)

CT_SIZE = tuple(p + n for p, n in zip(POS_VOX, NEG_VOX))            # (130, 170, 120)
COND_SIZE = tuple(s + 2 * m for s, m in zip(CT_SIZE, COND_MARGIN))  # (170, 210, 160)


def _up16(v):
    return int(math.ceil(v / 16) * 16)


PAD_SHAPE = tuple(_up16(s) for s in COND_SIZE)                      # (176, 224, 160)
PAD_LO = np.array([(P - C) // 2 for P, C in zip(PAD_SHAPE, COND_SIZE)])

COND_FILES = (
    "nacpet.nii.gz",
    "mri_combined_in_phase.nii.gz",
    "mri_combined_out_phase.nii.gz",
)
TOPO_FILE = "topogram.nii.gz"
FACE_MASK_FILE = "mri_face_mask.nii.gz"
REF_FILE = "nacpet.nii.gz"

# CT normalization, identical to baseline_v3 transforms.py
HU_MIN, HU_MAX = -1000.0, 2000.0


def norm_to_hu(norm):
    return norm.astype(np.float32) * (HU_MAX - HU_MIN) + HU_MIN


def mask_centroid(features_dir):
    """3-D centroid (voxel coords) of the face mask, plus the volume shape."""
    fm = nib.load(str(features_dir / FACE_MASK_FILE)).get_fdata() > 0
    c = []
    for ax in range(3):
        proj = fm.sum(axis=tuple(i for i in range(3) if i != ax)).astype(np.float64)
        c.append((np.arange(fm.shape[ax]) * proj).sum() / max(proj.sum(), 1))
    return np.array(c), fm.shape


def boxes_for(features_dir):
    """Nominal (un-clamped) and actual CT / cond boxes, each (3, 2) int."""
    center, shape = mask_centroid(features_dir)
    ci = np.round(center).astype(np.int64)
    ct_nom = np.stack([ci - np.array(NEG_VOX), ci + np.array(POS_VOX)], axis=1)
    cond_nom = ct_nom + np.stack([-np.array(COND_MARGIN), np.array(COND_MARGIN)], axis=1)

    def clamp(box):
        out = box.copy()
        for a in range(3):
            out[a, 0] = max(int(box[a, 0]), 0)
            out[a, 1] = min(int(box[a, 1]), shape[a])
        return out

    return ct_nom, clamp(ct_nom), cond_nom, clamp(cond_nom)


def _sl(box):
    return tuple(slice(int(b[0]), int(b[1])) for b in box)


def _stats(vol, nonzero):
    """MONAI NormalizeIntensityd statistics: (subtrahend, divisor)."""
    v = vol[vol != 0] if nonzero else vol
    if v.size == 0:
        return 0.0, 1.0
    sub = float(v.mean()) if nonzero else 0.0   # nacpet uses subtrahend=[0]
    div = float(v.std())
    return sub, (div if div > 1e-8 else 1.0)


def _place_offset(actual_lo, nominal_lo, pad_lo):
    return np.asarray(actual_lo) - np.asarray(nominal_lo) + np.asarray(pad_lo)


def _paste(canvas, arr, lo, channel=False):
    """Paste `arr` into `canvas` at voxel offset `lo`, clipping the overhang."""
    spatial = canvas.shape[1:] if channel else canvas.shape
    src, dst = [], []
    for a in range(3):
        s0 = int(lo[a])
        s1 = s0 + arr.shape[a + 1 if channel else a]
        d0, d1 = max(s0, 0), min(s1, spatial[a])
        if d1 <= d0:
            return
        src.append(slice(d0 - s0, d1 - s0))
        dst.append(slice(d0, d1))
    if channel:
        canvas[(slice(None), *dst)] = arr[(slice(None), *src)]
    else:
        canvas[tuple(dst)] = arr[tuple(src)]


def build_conditioning(features_dir):
    """Return (cond canvas [4, *PAD_SHAPE], ct_box, ct_lo, affine).

    Mirrors `cache_subject(..., with_ct=False)` followed by `HeadDataset.build`
    with no jitter and no flip.
    """
    ct_nom, ct_box, cond_nom, cond_box = boxes_for(features_dir)
    cond_sl = _sl(cond_box)

    cond = np.stack(
        [
            nib.load(str(features_dir / name)).get_fdata(dtype=np.float32)[cond_sl]
            for name in COND_FILES
        ],
        0,
    )
    topo = nib.load(str(features_dir / TOPO_FILE)).get_fdata(dtype=np.float32)
    topo = topo[cond_sl[0], :, cond_sl[2]]
    topo = topo.reshape(cond.shape[1], 1, cond.shape[3])

    norm = np.array(
        [
            _stats(cond[0], nonzero=False),   # nacpet:  subtrahend 0, / std
            _stats(cond[1], nonzero=True),    # MRI in:  z-score over nonzero
            _stats(cond[2], nonzero=True),    # MRI out: z-score over nonzero
            _stats(topo, nonzero=True),
        ],
        dtype=np.float32,
    )

    # The cache stored float16; keep that round-trip so the container's inputs
    # match the values the model was trained and validated on.
    cond = cond.astype(np.float16).astype(np.float32)
    topo = topo.astype(np.float16).astype(np.float32)

    topo_full = np.repeat(topo, cond.shape[2], axis=1)[None]
    stack = np.concatenate([cond, topo_full], 0)
    sub = norm[:, 0][:, None, None, None]
    div = norm[:, 1][:, None, None, None]
    stack = (stack - sub) / div

    canvas = np.zeros((4, *PAD_SHAPE), np.float32)
    cond_lo = _place_offset(cond_box[:, 0], cond_nom[:, 0], PAD_LO)
    _paste(canvas, stack, cond_lo, channel=True)

    ct_lo = _place_offset(ct_box[:, 0], cond_nom[:, 0], PAD_LO)
    affine = nib.load(str(features_dir / REF_FILE)).affine.astype(np.float32)
    return canvas, ct_box, ct_lo, affine


def extract_ct_patch(canvas_hu, ct_box, ct_lo):
    """Cut the CT box back out of a network-canvas-sized prediction."""
    size = ct_box[:, 1] - ct_box[:, 0]
    sl = tuple(slice(int(lo), int(lo) + int(sz)) for lo, sz in zip(ct_lo, size))
    return canvas_hu[sl].astype(np.float32)
