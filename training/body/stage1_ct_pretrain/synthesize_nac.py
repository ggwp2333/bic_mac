"""
Turn attenuation-corrected PET into a NAC-PET-like image, so QUADRA_HC's PET can
enter the input space the challenge model actually consumes.

**The problem.** The model reads non-attenuation-corrected PET. Every public
whole-body PET corpus, QUADRA_HC included, ships the attenuation-*corrected*
image. The survey measured how far apart those are, within subject, on the 8
BIC-MAC subjects that carry both: the AC/NAC ratio runs monotonically from 0.61
at the skin to 1.91 beyond 120 mm depth, a 3.1x swing, with voxelwise r = 0.83
after mean normalisation. Feeding AC-PET in as if it were NAC-PET is not an
option.

**The approach.** Attenuation correction multiplies each line of response by
exp(+L), where L is the line integral of mu through the body. Undoing it is a
projection-space operation, not a voxelwise one. So:

    L(view, plane, t) = Radon(mu)                       [dimensionless]
    W(x)              = BP(exp(-L)) / BP(1)             [attenuation weighting]
    NAC_hat(x)        = AC(x) * W(x)

`W` is the mean survival probability over the rays that pass through voxel x.
This is a **first-order** inversion: it applies the correct per-ray attenuation
and then averages in image space, where a true inversion would divide in
sinogram space and reconstruct. It is deliberately not an FBP round-trip --
a ramp filter on a divided sinogram produces streaks that a pretraining input
does not need, and the unfiltered normalised back-projection is smooth by
construction.

Because it is an approximation, **it must be validated before it is used**:
`--validate` runs the identical pipeline on BIC-MAC's AC-PET, where the real
NAC-PET exists, and reports what the synthesis recovers. That number is the
go/no-go.

Geometry follows visualization/acf_pet_diagnostic/surrogate_projector.py, whose
per-plane profile tracks STIR's at r = 0.9995: the direct segment is a stack of
independent 2-D parallel-beam Radon transforms, one per image plane.

Usage:
    python synthesize_nac.py --validate                 # the go/no-go, 8 subjects
    python synthesize_nac.py --quadra --workers 1       # write all 96
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
SHARED = Path(os.environ.get("BIC_MAC_DATA_ROOT", BASE_DIR.parents[2] / "data"))  # holds bic-mac-data/, quadra-hc-data/, synthrad2025-data/
BICMAC_TRAIN = SHARED / "bic-mac-data" / "train"

N_VIEWS = 50              # matches the challenge sinogram's view count
VOXEL_XY_MM = 1.52344     # CT grid in-plane spacing

# Carney bilinear HU -> mu at 511 keV, 120 kVp (must match the evaluator)
_A, _B, _BP, _LOW = 5.10e-5, 4.71e-2, 1047.0, 9.6e-5


def hu_to_mu(hu: torch.Tensor) -> torch.Tensor:
    hu1000 = hu + 1000.0
    mu = torch.where(hu1000 < _BP, _LOW * hu1000, _A * hu1000 + _B)
    return mu.clamp_min(0.0)


def _rot_grid(theta: float, z: int, y: int, x: int, dtype, device):
    c, s = float(np.cos(theta)), float(np.sin(theta))
    mat = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=dtype, device=device)
    return F.affine_grid(mat.expand(z, 2, 3), (z, 1, y, x), align_corners=False)


def radon_stack(vol: torch.Tensor, n_views: int = N_VIEWS) -> torch.Tensor:
    """(Z, Y, X) -> (n_views, Z, X). Line integrals in cm."""
    z, y, x = vol.shape
    out = torch.empty((n_views, z, x), dtype=vol.dtype, device=vol.device)
    src = vol.unsqueeze(1)
    for i in range(n_views):
        grid = _rot_grid(np.pi * i / n_views, z, y, x, vol.dtype, vol.device)
        out[i] = F.grid_sample(src, grid, align_corners=False,
                               padding_mode="zeros")[:, 0].sum(dim=1) * VOXEL_XY_MM / 10.0
    return out


def backproject(sino: torch.Tensor, y: int) -> torch.Tensor:
    """(n_views, Z, X) -> (Z, Y, X). Unfiltered adjoint of `radon_stack`.

    Each view's profile is smeared back across the plane and rotated home. No
    ramp filter: this is used only to average a bounded quantity (exp(-L) in
    [0, 1]) over rays, where filtering would add streaks and nothing else.
    """
    n_views, z, x = sino.shape
    out = torch.zeros((z, y, x), dtype=sino.dtype, device=sino.device)
    for i in range(n_views):
        smear = sino[i].unsqueeze(1).expand(z, y, x).unsqueeze(1)   # (Z,1,Y,X)
        grid = _rot_grid(-np.pi * i / n_views, z, y, x, sino.dtype, sino.device)
        out += F.grid_sample(smear, grid, align_corners=False,
                             padding_mode="zeros")[:, 0]
    return out / n_views


def attenuation_weighting(ct_hu: np.ndarray, device="cuda",
                          chunk: int = 96) -> np.ndarray:
    """Mean ray survival probability per voxel, from a CT in HU.

    Input and output are (X, Y, Z) as nibabel delivers them; the projector works
    in (Z, Y, X), so the axes are moved on the way in and back on the way out.
    Planes are processed in chunks to bound memory.
    """
    vol = torch.from_numpy(np.ascontiguousarray(
        np.transpose(ct_hu, (2, 1, 0)))).float()                    # (Z, Y, X)
    z, y, x = vol.shape
    out = torch.empty_like(vol)
    ones = None
    for lo in range(0, z, chunk):
        hi = min(lo + chunk, z)
        mu = hu_to_mu(vol[lo:hi].to(device))
        L = radon_stack(mu)                                          # (V, z, X)
        surv = torch.exp(-L)
        num = backproject(surv, y)
        if ones is None or ones.shape[0] != hi - lo:
            ones = backproject(torch.ones_like(surv), y)
        out[lo:hi] = (num / ones.clamp_min(1e-6)).cpu()
        del mu, L, surv, num
    return np.ascontiguousarray(np.transpose(out.numpy(), (2, 1, 0)))


def resample_like(vol: np.ndarray, src_img, ref_img, order: int = 1) -> np.ndarray:
    from scipy import ndimage
    m = np.linalg.inv(src_img.affine) @ ref_img.affine
    idx = np.indices(ref_img.shape[:3], dtype=np.float32).reshape(3, -1)
    src_idx = m[:3, :3] @ idx + m[:3, 3:4]
    return ndimage.map_coordinates(vol, src_idx, order=order, mode="constant",
                                   cval=0).reshape(ref_img.shape[:3])


# ---------------------------------------------------------------- validation
def validate(subjects, device="cuda") -> dict:
    """Synthesize NAC from BIC-MAC's own AC-PET and score it against the real one."""
    from scipy import ndimage

    rows = []
    for sub in subjects:
        d = BICMAC_TRAIN / sub
        ct_img = nib.load(d / "ct-label" / "ct.nii.gz")
        ct = ct_img.get_fdata(dtype=np.float32)
        body = np.asanyarray(nib.load(d / "ct-label" / "body_seg.nii.gz").dataobj) > 0
        nac_img = nib.load(d / "features" / "nacpet.nii.gz")
        nac_true = nac_img.get_fdata(dtype=np.float32)

        ac_img = nib.load(d / "pet-label" / "pet.nii.gz")
        ac = resample_like(ac_img.get_fdata(dtype=np.float32), ac_img, ct_img)

        W = attenuation_weighting(ct, device=device)
        nac_hat = ac * W

        # both are arbitrary-unit; compare on shape, scaled to in-body mean 1
        t = nac_true[body]
        h = nac_hat[body]
        a = ac[body]
        t_n = t / max(float(t.mean()), 1e-9)
        h_n = h / max(float(h.mean()), 1e-9)
        a_n = a / max(float(a.mean()), 1e-9)

        zooms = [float(z) for z in ct_img.header.get_zooms()[:3]]
        depth = ndimage.distance_transform_edt(body, sampling=zooms).astype(np.float32)[body]

        row = {
            "subject": sub,
            "r_synth_vs_true": float(np.corrcoef(h_n, t_n)[0, 1]),
            "r_ac_vs_true": float(np.corrcoef(a_n, t_n)[0, 1]),
            "mae_synth": float(np.abs(h_n - t_n).mean()),
            "mae_ac": float(np.abs(a_n - t_n).mean()),
        }
        # does the synthesis reproduce the measured depth profile?
        prof = {}
        for lo, hi in ((0, 10), (20, 30), (40, 60), (80, 120), (120, 400)):
            sel = (depth >= lo) & (depth < hi)
            if sel.sum() > 1000:
                prof[f"{lo}-{hi}"] = {
                    "true_ac_over_nac": float(a_n[sel].mean() / max(t_n[sel].mean(), 1e-9)),
                    "synth_ac_over_nac": float(a_n[sel].mean() / max(h_n[sel].mean(), 1e-9)),
                }
        row["depth_profile"] = prof
        rows.append(row)
        print(f"  {sub}: r {row['r_ac_vs_true']:.3f} (AC) -> "
              f"{row['r_synth_vs_true']:.3f} (synth)   "
              f"MAE {row['mae_ac']:.4f} -> {row['mae_synth']:.4f}", flush=True)

    summary = {
        "n": len(rows),
        "r_ac_vs_true_mean": float(np.mean([r["r_ac_vs_true"] for r in rows])),
        "r_synth_vs_true_mean": float(np.mean([r["r_synth_vs_true"] for r in rows])),
        "mae_ac_mean": float(np.mean([r["mae_ac"] for r in rows])),
        "mae_synth_mean": float(np.mean([r["mae_synth"] for r in rows])),
        "subjects": rows,
    }
    summary["mae_reduction_pct"] = 100 * (
        1 - summary["mae_synth_mean"] / max(summary["mae_ac_mean"], 1e-12)
    )
    return summary


# ---------------------------------------------------------------- QUADRA pass
def synthesize_quadra(manifest_path: Path, out_dir: Path, device="cuda",
                      limit: int = 0) -> None:
    with open(manifest_path) as f:
        corpus = json.load(f)
    cases = [c for c in corpus["cases"] if c["dataset"] == "QUADRA_HC"]
    if limit:
        cases = cases[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, c in enumerate(cases, 1):
        dst = out_dir / f"{Path(c['case_id']).name}_pseudonac.nii.gz"
        if dst.is_file():
            print(f"  [{i}/{len(cases)}] {dst.name} exists", flush=True)
            continue
        ct_img = nib.load(c["ct"])
        ct = ct_img.get_fdata(dtype=np.float32)
        pet_path = str(c["ct"]).replace("_CT-AC.nii.gz", "_PT-SUV.nii.gz")
        if not os.path.exists(pet_path):
            print(f"  [{i}/{len(cases)}] no PET beside {c['ct']}", flush=True)
            continue
        ac_img = nib.load(pet_path)
        ac = resample_like(ac_img.get_fdata(dtype=np.float32), ac_img, ct_img)
        W = attenuation_weighting(ct, device=device)
        nac_hat = (ac * W).astype(np.float32)
        img = nib.Nifti1Image(nac_hat, ct_img.affine, ct_img.header)
        img.set_data_dtype(np.float32)
        nib.save(img, dst)
        print(f"  [{i}/{len(cases)}] {dst.name}  mean={nac_hat.mean():.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--quadra", action="store_true")
    ap.add_argument("--n", type=int, default=8, help="subjects for --validate")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--corpus", type=Path, default=BASE_DIR / "corpus.json")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.out_dir is None:
        import yaml
        with open(BASE_DIR / "config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        out = Path(cfg.get("output_dir", "outputs"))
        a.out_dir = (out if out.is_absolute() else BASE_DIR / out) / "pseudo_nac"

    if a.validate:
        subs = [s for s in sorted(os.listdir(BICMAC_TRAIN))
                if (BICMAC_TRAIN / s / "pet-label" / "pet.nii.gz").is_file()][: a.n]
        print(f"validating on {len(subs)} subjects with both AC and NAC PET")
        summary = validate(subs, device=a.device)
        dst = a.out_dir.parent / "pseudo_nac_validation.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nr  vs real NAC : {summary['r_ac_vs_true_mean']:.3f} (raw AC) "
              f"-> {summary['r_synth_vs_true_mean']:.3f} (synthesised)")
        print(f"MAE vs real NAC: {summary['mae_ac_mean']:.4f} -> "
              f"{summary['mae_synth_mean']:.4f} "
              f"({summary['mae_reduction_pct']:+.1f}%)")
        print(f"wrote {dst}")

    if a.quadra:
        synthesize_quadra(a.corpus, a.out_dir, device=a.device, limit=a.limit)


if __name__ == "__main__":
    main()
