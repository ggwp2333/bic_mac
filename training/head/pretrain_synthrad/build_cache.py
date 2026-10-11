"""
Build a head-VAE cache from SynthRAD2025 head-and-neck, in the exact `.npz`
format `rithihas_brain/train_vae_ddp.py` already reads.

Writing a compatible cache rather than a new dataset class is deliberate: the
model, losses, augmentation and training loop that produced `v1_DA` stay
untouched, so a pretrained checkpoint drops straight back into the existing
head pipeline.

**What SynthRAD supplies and what it does not.** The head VAE conditions on
`[nacpet, mri_in_phase, mri_out_phase, topogram]` over a head crop. SynthRAD
head-and-neck gives one Dixon phase (`Source Head and Neck MRCAT IP` or
`t T1 3D TFE mDIXON gd`) and a CT — no PET, no topogram, no second phase. The MR
goes into the `mri_in_phase` slot and the other three channels are zero, exactly
as QUADRA's PET-only cases are handled in `pretrain_v2`. 91 of the 156 HN cases
are T1 Dixon; the rest are dropped (`synthrad_metadata.py` classifies them).

Two geometric problems have to be solved before any of that matters.

**Resolution.** SynthRAD is 1x1x3 mm, BIC-MAC is 1.523x1.523x2.0 mm, and the
crop box is defined in *voxels*. Without resampling, a 130x170x95 box would
cover a completely different piece of anatomy. Every volume is resampled onto
the BIC-MAC voxel size first.

**The anchor.** BIC-MAC anchors the box on the centroid of
`features/mri_face_mask.nii.gz`. SynthRAD has no face mask, so the anchor has to
come from the CT. The rule below --

    anchor = [x_mid, y_anterior - 70, z_top - 110] + OFFSET

-- was **fitted against the 79 cached BIC-MAC boxes**, not guessed: the naive
version (offset zero) is biased by -35 voxels in y and -40 in z, because the
real box deliberately extends above the skull and anterior of the face. Run
`--validate-anchor` to reproduce the fit and see the residuals. They are of the
same order as the +/-10 voxel shift augmentation the training already applies,
which is the standard this has to meet, not exactness.

Usage:
    python build_cache.py --validate-anchor      # reproduce the calibration
    python build_cache.py --limit 2 --dry-run    # inspect two cases
    python build_cache.py --workers 8            # build the cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SRC = BASE_DIR.parent
SHARED = Path(os.environ.get("BIC_MAC_DATA_ROOT", BASE_DIR.parents[2] / "data"))  # holds bic-mac-data/, quadra-hc-data/, synthrad2025-data/

SYNTHRAD = SHARED / "synthrad2025-data" / "synthRAD2025_Task1_Train"
HN_DIR = SYNTHRAD / "Task1" / "HN"
BICMAC_TRAIN = SHARED / "bic-mac-data" / "train"
BICMAC_CACHE = BASE_DIR.parent / "head_cache_bicmac"   # written by ../build_head_cache.py

# --- geometry, copied verbatim from rithihas_brain/train_vae_ddp.py ---------
POS_VOX = (65, 70, 110)
NEG_VOX = (65, 100, 10)
COND_MARGIN = (20, 20, 20)
CT_SIZE = tuple(p + n for p, n in zip(POS_VOX, NEG_VOX))        # (130, 170, 120)
COND_SIZE = tuple(s + 2 * m for s, m in zip(CT_SIZE, COND_MARGIN))  # (170, 210, 160)

# BIC-MAC voxel size; SynthRAD is resampled onto this before cropping.
TARGET_SPACING = (1.5234375, 1.5234375, 2.0)

# Fitted by --validate-anchor against the 79 cached BIC-MAC boxes. Replace only
# by re-running that mode; these are measurements, not tuning knobs.
ANCHOR_OFFSET = np.array([-2, 37, 38])

CHANNELS = ("nacpet", "mri_in_phase", "mri_out_phase", "topogram")
MR_SLOT = 1                      # SynthRAD's Dixon phase goes here


# ------------------------------------------------------------------ anchoring
def head_anchor_from_body(body: np.ndarray) -> np.ndarray:
    """CT-only stand-in for the face-mask centroid, in voxels of the (x,y,z) grid.

    +x = left, +y = anterior, +z = superior, matching BIC-MAC's LAS volumes.
    """
    zs = np.where(body.any(axis=(0, 1)))[0]
    if not len(zs):
        raise ValueError("empty body mask")
    z_top = int(zs.max())
    head = body[:, :, max(z_top - 150, 0): z_top + 1]
    ys = np.where(head.any(axis=(0, 2)))[0]
    xs = np.where(head.any(axis=(1, 2)))[0]
    y_ant = int(ys.max())
    x_mid = int(round((xs.min() + xs.max()) / 2))
    return np.array([x_mid, y_ant - POS_VOX[1], z_top - POS_VOX[2]]) + ANCHOR_OFFSET


def boxes_from_anchor(anchor: np.ndarray, shape) -> dict:
    """Nominal and clamped ct/cond boxes, in the convention the cache uses."""
    ct_lo = anchor - np.array(NEG_VOX)
    ct_hi = anchor + np.array(POS_VOX)
    cond_lo = ct_lo - np.array(COND_MARGIN)
    cond_hi = ct_hi + np.array(COND_MARGIN)

    def clamp(lo, hi):
        return np.stack([np.maximum(lo, 0), np.minimum(hi, np.array(shape))], 1).astype(np.int32)

    return {
        "ct_box": clamp(ct_lo, ct_hi),
        "ct_box_nom": np.stack([ct_lo, ct_hi], 1).astype(np.int32),
        "cond_box": clamp(cond_lo, cond_hi),
        "cond_box_nom": np.stack([cond_lo, cond_hi], 1).astype(np.int32),
    }


def validate_anchor(limit: int = 0) -> dict:
    """Fit and report the offset against the cached BIC-MAC boxes."""
    import glob
    import nibabel as nib

    rows, subs = [], []
    files = sorted(glob.glob(str(BICMAC_CACHE / "sub-*.npz")))
    if limit:
        files = files[:limit]
    for f in files:
        sub = Path(f).stem
        p = BICMAC_TRAIN / sub / "ct-label" / "body_seg.nii.gz"
        if not p.is_file():
            continue
        truth = np.load(f)["cond_box_nom"][:, 0] + np.array(COND_MARGIN) + np.array(NEG_VOX)
        body = np.asanyarray(nib.load(p).dataobj) > 0
        # measure with the offset removed, so the printed mean IS the offset
        raw = head_anchor_from_body(body) - ANCHOR_OFFSET
        rows.append(raw - truth)
        subs.append(sub)
    r = np.array(rows, dtype=float)
    out = {
        "n": len(r),
        "fitted_offset": (-r.mean(0)).round(1).tolist(),
        "offset_in_use": ANCHOR_OFFSET.tolist(),
        "residual_sd_voxels": r.std(0).round(1).tolist(),
        "residual_sd_mm": (r.std(0) * np.array(TARGET_SPACING)).round(1).tolist(),
        "aug_shift_voxels": [10, 10, 10],
    }
    return out


# ------------------------------------------------------------------ resampling
def load_mha_las(path: Path, spacing=TARGET_SPACING, order=1):
    """Read a .mha and return (x, y, z) on the BIC-MAC voxel size, LAS-oriented.

    SimpleITK delivers LPS with index order (z, y, x); BIC-MAC volumes are LAS
    with (x, y, z). The transpose fixes the index order and the y flip converts
    posterior-positive to anterior-positive.
    """
    import SimpleITK as sitk
    from scipy import ndimage

    img = sitk.ReadImage(str(path))
    arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0)).astype(np.float32)
    src = np.array(img.GetSpacing(), dtype=float)           # (x, y, z)
    zoom = src / np.array(spacing, dtype=float)
    if not np.allclose(zoom, 1.0, atol=1e-3):
        arr = ndimage.zoom(arr, zoom, order=order, mode="nearest")
    return arr[:, ::-1, :]                                   # LPS -> LAS in y


def crop(vol: np.ndarray, box: np.ndarray, fill: float) -> np.ndarray:
    """Crop `vol` to `box`, padding with `fill` wherever the box leaves the volume."""
    size = box[:, 1] - box[:, 0]
    out = np.full(tuple(size), fill, dtype=np.float32)
    src = [slice(max(box[a, 0], 0), min(box[a, 1], vol.shape[a])) for a in range(3)]
    dst = [slice(src[a].start - box[a, 0], src[a].stop - box[a, 0]) for a in range(3)]
    if all(s.stop > s.start for s in src):
        out[tuple(dst)] = vol[tuple(src)]
    return out


def build_case(case_dir: str, out_path: str) -> tuple[str, str]:
    case = Path(case_dir)
    ct = load_mha_las(case / "ct.mha", order=1)
    mr = load_mha_las(case / "mr.mha", order=1)
    mask_p = case / "mask.mha"
    if mask_p.is_file():
        body = load_mha_las(mask_p, order=0) > 0.5
    else:
        body = ct > -300
    if mr.shape != ct.shape:
        raise ValueError(f"{case.name}: mr {mr.shape} != ct {ct.shape}")

    anchor = head_anchor_from_body(body)
    boxes = boxes_from_anchor(anchor, ct.shape)

    # --- how much of the nominal box actually exists in this volume ---
    nom = boxes["ct_box_nom"]
    have = boxes["ct_box"]
    covered = float(np.prod(np.maximum(have[:, 1] - have[:, 0], 0)) /
                    np.prod(nom[:, 1] - nom[:, 0]))

    cb = boxes["cond_box"]
    cond = np.zeros((3, *(cb[:, 1] - cb[:, 0])), dtype=np.float16)
    mr_crop = crop(mr, cb, 0.0)
    cond[MR_SLOT] = mr_crop.astype(np.float16)
    topo = np.zeros((cb[0, 1] - cb[0, 0], 1, cb[2, 1] - cb[2, 0]), dtype=np.float16)

    # Normalisation stats, in the (sub, div) form the dataset applies as
    # (x - sub) / div. Channel 0/2/3 are all-zero here, so div must be 1 rather
    # than a std of 0. The MR channel gets nonzero mean/std, matching
    # NormalizeIntensityd(nonzero=True) in the challenge transforms.
    norm = np.zeros((4, 2), dtype=np.float32)
    norm[:, 1] = 1.0
    pos = mr_crop[mr_crop > 0]
    if pos.size:
        norm[MR_SLOT] = (float(pos.mean()), float(pos.std()) or 1.0)

    tb = boxes["ct_box"]
    ct_c = crop(ct, tb, -1000.0).astype(np.float16)
    bm = crop(body.astype(np.float32), tb, 0.0) > 0.5
    record = {
        "cond": cond, "topo": topo, "norm": norm,
        "ct": ct_c, "pmask": bm.astype(np.uint8), "bmask": bm.astype(np.uint8),
        "affine": np.eye(4, dtype=np.float32),
        "coverage": np.float32(covered),
        **{k: v for k, v in boxes.items()},
    }
    np.savez_compressed(out_path, **record)
    return case.name, f"coverage={covered:.2f} mr_mean={norm[MR_SLOT, 0]:.1f}"


def _worker(args):
    try:
        return build_case(*args)
    except Exception as e:
        return args[0], f"ERROR {type(e).__name__}: {e}"


def discover_dixon_cases() -> list[str]:
    """HN cases whose MR is T1 Dixon, per the SynthRAD acquisition sheets."""
    sys.path.insert(0, str(BASE_DIR))
    from synthrad_metadata import load_metadata

    md = load_metadata(str(SYNTHRAD))
    keep = []
    for d in sorted(os.listdir(HN_DIR)):
        p = HN_DIR / d
        if not p.is_dir() or not (p / "ct.mha").is_file():
            continue
        if md.get(d, {}).get("sequence_family") == "T1 Dixon":
            keep.append(str(p))
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=BASE_DIR / "head_cache_synthrad")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-anchor", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=0.6,
                    help="drop cases whose head box is mostly outside the volume")
    a = ap.parse_args()

    if a.validate_anchor:
        out = validate_anchor()
        print(json.dumps(out, indent=2))
        print("\nfitted_offset is what ANCHOR_OFFSET should equal; residual_sd_mm "
              "is the placement jitter this rule leaves, to be read against the "
              "+/-10 voxel shift augmentation already in training.")
        return

    cases = discover_dixon_cases()
    if a.limit:
        cases = cases[: a.limit]
    print(f"{len(cases)} SynthRAD HN T1-Dixon cases")
    if a.dry_run:
        for c in cases[:5]:
            print(" ", c)
        return

    a.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, str(a.out_dir / f"{Path(c).name}.npz")) for c in cases]
    done, failed = 0, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            name, msg = f.result()
            if msg.startswith("ERROR"):
                failed.append((name, msg))
            else:
                done += 1
            print(f"[{i}/{len(jobs)}] {name}: {msg}", flush=True)

    # prune by coverage, so a half-empty head box never reaches training
    dropped = []
    for p in sorted(a.out_dir.glob("*.npz")):
        cov = float(np.load(p)["coverage"])
        if cov < a.min_coverage:
            dropped.append((p.name, round(cov, 2)))
            p.unlink()
    kept = sorted(x.stem for x in a.out_dir.glob("*.npz"))
    with open(a.out_dir / "subjects.json", "w") as f:
        json.dump({"subjects": kept, "n": len(kept),
                   "dropped_low_coverage": dropped,
                   "failed": failed,
                   "anchor_offset": ANCHOR_OFFSET.tolist(),
                   "target_spacing": list(TARGET_SPACING)}, f, indent=2)
    print(f"\nbuilt {done}, failed {len(failed)}, dropped for coverage "
          f"{len(dropped)}, kept {len(kept)}")
    print(f"wrote {a.out_dir / 'subjects.json'}")
    if failed:
        for n, m in failed[:5]:
            print("  FAILED", n, m)


if __name__ == "__main__":
    main()
