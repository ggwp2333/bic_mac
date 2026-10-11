"""
Build the CT-only pretraining corpus: BIC-MAC + Vienna QUADRA_HC.

Writes `corpus.json`, a manifest of every CT with its body mask, its dataset of
origin, its bone-calibration gain and its train/val assignment. Nothing else in
this module reads the datasets directly, so the corpus is defined in exactly one
place and can be inspected without loading a single volume.

Two things happen here that are not bookkeeping:

**Body masks for QUADRA_HC.** BIC-MAC ships `ct-label/body_seg.nii.gz`;
QUADRA_HC ships none. The mask is derived with the threshold + 3-D
largest-component + hole-fill rule validated in
`visualization/external_data_survey/` at **Dice 0.9917** against BIC-MAC's own
`body_seg` (see `profiles/bodymask_equivalence.json`). The erosion step is not
optional: without it the per-slice hole fill floods the air inside the scanner
couch shell and a quarter of the "in-body" voxels come back below -500 HU.

**Grouping by subject for the val split.** QUADRA_HC's 96 scans are 48 subjects
imaged twice, 38 +/- 10 days apart. Splitting a subject's test and retest across
train and val would leak, so the split is over *subjects*, never scans.

Usage:
    python prepare_corpus.py                 # writes corpus.json
    python prepare_corpus.py --workers 16    # body-mask derivation is the cost
    python prepare_corpus.py --dry-run       # list what it would do
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SHARED = Path(os.environ.get("BIC_MAC_DATA_ROOT", BASE_DIR.parents[2] / "data"))  # holds bic-mac-data/, quadra-hc-data/, synthrad2025-data/


def _configured_output_dir() -> Path:
    """Read output_dir from config.yaml so the masks cannot land somewhere the
    trainer will not look. Hardcoding it here once already produced a manifest
    pointing at a directory that no longer existed."""
    import yaml

    try:
        with open(BASE_DIR / "config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return BASE_DIR / "outputs"
    out = Path(cfg.get("output_dir", "outputs"))
    return out if out.is_absolute() else BASE_DIR / out

BICMAC_TRAIN = SHARED / "bic-mac-data" / "train"
QUADRA_ROOT = SHARED / "quadra-hc-data" / "QUADRA_HC" / "Imaging Data"

# Cortical bone HU depends on tube voltage. QUADRA_HC's low-dose CT is acquired
# at 140 kVp while the challenge evaluator applies the Carney parameters for
# 120 kVp, and the measured tails differ accordingly: QUADRA cortical bone
# averages 57 HU lower with a sixth as many voxels above 1200 HU.
#
# The gain below was fitted by least squares on 49 quantiles (2%..98%) of the
# two pooled in-body bone histograms, under the one-parameter model
#     HU_calibrated = pivot + gain * (HU_quadra - pivot)
# It matches BIC-MAC's bone quantiles to within 17 HU worst case, 7 HU mean.
# Refit with visualization/external_data_survey if either corpus changes.
BONE_PIVOT_HU = 300.0
BONE_GAIN_QUADRA = 1.2028


def derive_body_mask(hu: np.ndarray, thresh: float = -300.0,
                     erode: int = 3) -> np.ndarray:
    """Threshold -> erode -> largest 3-D component -> dilate back -> fill holes.

    Ported from visualization/external_data_survey/survey_common.py so this
    trainer has no dependency on the analysis tree. See that module for why
    each step is there.
    """
    from scipy import ndimage

    raw = hu > thresh
    core = ndimage.binary_erosion(raw, iterations=erode)
    lab, n = ndimage.label(core)
    if n == 0:
        lab, n = ndimage.label(raw)
        if n == 0:
            return raw
        sizes = ndimage.sum(raw, lab, range(1, n + 1))
        core = lab == (int(np.argmax(sizes)) + 1)
    else:
        sizes = ndimage.sum(core, lab, range(1, n + 1))
        core = lab == (int(np.argmax(sizes)) + 1)
        core = ndimage.binary_dilation(core, iterations=erode) & raw

    out = np.zeros_like(core)
    for z in range(core.shape[2]):
        if core[:, :, z].any():
            out[:, :, z] = ndimage.binary_fill_holes(core[:, :, z])
    return out


def _make_quadra_mask(ct_path: str, mask_path: str) -> tuple[str, str, int]:
    img = nib.load(ct_path)
    ct = img.get_fdata(dtype=np.float32)
    mask = derive_body_mask(ct).astype(np.uint8)
    out = nib.Nifti1Image(mask, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, mask_path)
    return ct_path, mask_path, int(mask.sum())


def discover_bicmac() -> list[dict]:
    cases = []
    for sub in sorted(os.listdir(BICMAC_TRAIN)):
        ct = BICMAC_TRAIN / sub / "ct-label" / "ct.nii.gz"
        seg = BICMAC_TRAIN / sub / "ct-label" / "body_seg.nii.gz"
        if not ct.is_file() or not seg.is_file():
            continue
        cases.append({
            "case_id": f"bicmac/{sub}",
            "dataset": "BIC-MAC",
            "subject": sub,
            "ct": str(ct),
            "body_mask": str(seg),
            "bone_gain": 1.0,
        })
    return cases


def discover_quadra(mask_dir: Path) -> list[dict]:
    cases = []
    if not QUADRA_ROOT.is_dir():
        return cases
    for sub in sorted(os.listdir(QUADRA_ROOT)):
        sub_dir = QUADRA_ROOT / sub
        if not sub_dir.is_dir():
            continue
        for tp in sorted(os.listdir(sub_dir)):
            tp_dir = sub_dir / tp
            if not tp_dir.is_dir():
                continue
            ct = [
                f for f in sorted(os.listdir(tp_dir))
                if re.search(r"CT-AC\.nii(\.gz)?$", f)
            ]
            if not ct:
                continue
            cases.append({
                "case_id": f"quadra/{sub}_{tp}",
                "dataset": "QUADRA_HC",
                "subject": sub,          # test and retest share this
                "ct": str(tp_dir / ct[0]),
                "body_mask": str(mask_dir / f"{sub}_{tp}_body_mask.nii.gz"),
                "bone_gain": BONE_GAIN_QUADRA,
            })
    return cases


def assign_split(cases: list[dict], n_val_quadra: int, n_val_bicmac: int) -> None:
    """Hold out whole *subjects*, deterministically, and mark them in place."""
    val_subjects: set[tuple[str, str]] = set()
    for dataset, n in (("QUADRA_HC", n_val_quadra), ("BIC-MAC", n_val_bicmac)):
        subjects = sorted({c["subject"] for c in cases if c["dataset"] == dataset})
        if n > len(subjects):
            raise ValueError(
                f"asked for {n} held-out {dataset} subjects, only "
                f"{len(subjects)} exist"
            )
        # every k-th subject, so the held-out set spans the cohort rather than
        # taking a contiguous block of acquisition order
        if n:
            step = len(subjects) / n
            picked = {subjects[min(int(i * step), len(subjects) - 1)]
                      for i in range(n)}
            val_subjects |= {(dataset, s) for s in picked}
    for c in cases:
        c["split"] = "val" if (c["dataset"], c["subject"]) in val_subjects else "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=BASE_DIR / "corpus.json")
    ap.add_argument("--mask-dir", type=Path, default=None,
                    help="defaults to <output_dir from config.yaml>/corpus/body_masks")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val-quadra-subjects", type=int, default=6,
                    help="held-out QUADRA subjects (both timepoints each)")
    ap.add_argument("--val-bicmac-subjects", type=int, default=0,
                    help="default 0: every BIC-MAC CT is worth pretraining on, "
                         "and the downstream run validates on Codabench")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.mask_dir is None:
        a.mask_dir = _configured_output_dir() / "corpus" / "body_masks"
    print(f"mask dir: {a.mask_dir}")

    cases = discover_bicmac() + discover_quadra(a.mask_dir)
    if not cases:
        raise SystemExit("no cases found; check the dataset paths")
    assign_split(cases, a.val_quadra_subjects, a.val_bicmac_subjects)

    n_by = {}
    for c in cases:
        n_by[(c["dataset"], c["split"])] = n_by.get((c["dataset"], c["split"]), 0) + 1
    print("corpus:")
    for k in sorted(n_by):
        print(f"  {k[0]:12s} {k[1]:5s} {n_by[k]:4d} scans")
    print(f"  {'TOTAL':12s} {'':5s} {len(cases):4d} scans")

    missing = [c for c in cases if not Path(c["body_mask"]).is_file()]
    print(f"body masks to derive: {len(missing)}")
    if a.dry_run:
        return

    if missing:
        a.mask_dir.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futures = [
                ex.submit(_make_quadra_mask, c["ct"], c["body_mask"])
                for c in missing
            ]
            for i, fut in enumerate(as_completed(futures), 1):
                _, path, n = fut.result()
                print(f"  [{i}/{len(missing)}] {Path(path).name}  "
                      f"{n * 1.5234 ** 2 * 2.0 / 1e6:.1f} L", flush=True)

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bone_calibration": {
            "pivot_hu": BONE_PIVOT_HU,
            "gain_by_dataset": {"BIC-MAC": 1.0, "QUADRA_HC": BONE_GAIN_QUADRA},
            "note": "fitted on 49 bone-histogram quantiles; see module docstring",
        },
        "body_mask": {
            "BIC-MAC": "ct-label/body_seg.nii.gz (shipped)",
            "QUADRA_HC": "derived, Dice 0.9917 vs shipped on BIC-MAC",
        },
        "n_cases": len(cases),
        "cases": sorted(cases, key=lambda c: c["case_id"]),
    }
    with open(a.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
