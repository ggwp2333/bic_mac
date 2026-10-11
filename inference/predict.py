"""Container entrypoint: one subject in, one pseudo-CT out.

The challenge harness runs this as

    docker run --rm --gpus all --network none --memory 128g \
      -v <subject>/features:/data/features:ro -v <out>:/data/output <image>

so the defaults are the mounted paths and nothing else is required. Both are
overridable for local testing.

Pipeline:

    1. body network: whole-body sliding-window inference -> body CT (HU)
    2. head cVAE on a face-mask-centred crop, prior mean -> head CT (HU)
    3. paste the head patch into the body and blend in mu-space at the patch's
       inferior seam                                     -> /data/output/ct.nii.gz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from blend import blend_head_into_body
from dataset import get_subject_features
from head_crop import build_conditioning, extract_ct_patch, norm_to_hu
from head_model import build_head_vae
from transforms import get_inference_transforms
from unet import build_model as build_body_model

BASE_DIR = Path(__file__).resolve().parent
# Inside the container the weights sit next to this file; in a repository
# checkout they are one level up, in weights/.
_LOCAL_WEIGHTS = BASE_DIR / "weights"
WEIGHTS_DIR = Path(os.environ.get(
    "WEIGHTS_DIR", _LOCAL_WEIGHTS if _LOCAL_WEIGHTS.is_dir() else BASE_DIR.parent / "weights"))
# Final submission weights (Codabench 890240).
# Body: two-stage pretrained width-96 U-Net (206.3M params), fine-tune epoch 92 -
#       the best single body checkpoint, CT mu-MAE 0.0056968 on the val leaderboard.
# Head: SynthRAD-pretrained cVAE (166.1M params), all-data fine-tune epoch 70.
BODY_CKPT = WEIGHTS_DIR / "body_finetune_e0092.pth"
HEAD_CKPT = WEIGHTS_DIR / "head_finetune_e070.pth"

INPUT_KEYS = ("nacpet", "mri_combined_in_phase", "mri_combined_out_phase", "topogram")
BODY_BASE_CHANNELS = 96

# Matches the body trainer's leaderboard predictions exactly (train.py:
# VAL_SW_BATCH, VAL_OVERLAP, VAL_MODE and config.yaml patch_size).
PATCH_SIZE = (192, 192, 192)
OVERLAP = 0.5
SW_MODE = "gaussian"
# Window batch: 1 reproduces the leaderboard run bit-for-bit; >1 is the same
# arithmetic with more windows in flight, and is what keeps us inside the
# 5-minute-per-subject budget on an A40.
SW_BATCH = int(os.environ.get("SW_BATCH", "1"))


def log(message):
    print(message, flush=True)


def predict_body_hu(features_dir, device):
    """Whole-body pseudo-CT in HU from the body network, plus the NAC-PET affine."""
    # unet.py defaults to width 64; base_channels=96 is what makes the
    # state_dict load.
    model = build_body_model(in_channels=len(INPUT_KEYS), base_channels=BODY_BASE_CHANNELS).to(device)
    model.load_state_dict(
        torch.load(BODY_CKPT, map_location=device, weights_only=True)
    )
    model.eval()

    data = get_inference_transforms(INPUT_KEYS)(get_subject_features(str(features_dir)))
    x = data["input"].unsqueeze(0).to(device)
    autocast = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()
    with torch.no_grad(), autocast:
        prediction = sliding_window_inference(
            x, PATCH_SIZE, SW_BATCH, model, overlap=OVERLAP, mode=SW_MODE,
            progress=False, sw_device=device, device="cpu",
        )
    body_hu = (prediction.float().numpy()[0, 0] * 3000.0 - 1000.0).astype(np.float32)
    affine = data["nacpet"].meta["affine"].numpy()

    del model, prediction, x, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return body_hu, affine


def predict_head_hu(features_dir, device):
    """Head-VAE pseudo-CT patch in HU (prior mean), plus its full-volume box."""
    checkpoint = torch.load(HEAD_CKPT, map_location=device, weights_only=False)
    model = build_head_vae(checkpoint["cfg"], in_channels=len(INPUT_KEYS)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    log(f"[head] VAE checkpoint epoch {checkpoint.get('epoch')}")

    cond, ct_box, ct_lo, affine = build_conditioning(features_dir)
    x = torch.from_numpy(cond)[None].to(device)
    autocast = (torch.amp.autocast("cuda") if device.type == "cuda"
                else nullcontext())
    with torch.no_grad(), autocast:
        prediction = model(x, ct=None)[0].float()
    canvas_hu = norm_to_hu(prediction[0, 0].cpu().numpy())

    del model, checkpoint, x, prediction
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return extract_ct_patch(canvas_hu, ct_box, ct_lo), ct_box, affine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_dir", default="/data/features")
    parser.add_argument("--output_ct", default="/data/output/ct.nii.gz")
    parser.add_argument("--save_parts", default=None,
                        help="debugging: also write body/head volumes here")
    args = parser.parse_args()

    started = time.time()
    features_dir = Path(args.features_dir)
    if not features_dir.is_dir():
        sys.exit(f"features directory not found: {features_dir}")
    for missing in [f for f in ("nacpet.nii.gz", "mri_face_mask.nii.gz")
                    if not (features_dir / f).is_file()]:
        sys.exit(f"required input missing: {features_dir / missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[setup] device={device}  sw_batch={SW_BATCH}  weights={WEIGHTS_DIR}")
    if device.type == "cuda":
        log(f"[setup] gpu={torch.cuda.get_device_name(0)}")

    t0 = time.time()
    body_hu, affine = predict_body_hu(features_dir, device)
    log(f"[body] {body_hu.shape} in {time.time() - t0:.1f}s  "
        f"HU[{body_hu.min():.0f}, {body_hu.max():.0f}]")

    t0 = time.time()
    head_hu, ct_box, head_affine = predict_head_hu(features_dir, device)
    log(f"[head] VAE patch {head_hu.shape} box={ct_box.tolist()} "
        f"in {time.time() - t0:.1f}s  HU[{head_hu.min():.0f}, {head_hu.max():.0f}]")

    if not np.allclose(affine, head_affine, atol=1e-4):
        sys.exit("[blend] affine mismatch between the two paths")

    hybrid_hu, stats = blend_head_into_body(body_hu, head_hu, ct_box, affine)
    if not np.isfinite(hybrid_hu).all():
        sys.exit("[blend] hybrid contains NaN or Inf")
    log(f"[blend] seam z={stats['seam_z']}  "
        f"transition~{stats['transition_slices']} slices  "
        f"HU[{hybrid_hu.min():.0f}, {hybrid_hu.max():.0f}] "
        f"mean={hybrid_hu.mean():.1f}")

    # Shape and affine must match features/nacpet.nii.gz exactly, so the
    # reference header is copied rather than rebuilt.
    reference = nib.load(str(features_dir / "nacpet.nii.gz"))
    if hybrid_hu.shape != reference.shape:
        sys.exit(f"[out] shape {hybrid_hu.shape} != nacpet {reference.shape}")
    header = reference.header.copy()
    header.set_data_dtype(np.float32)

    out_path = Path(args.output_ct)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(hybrid_hu, reference.affine, header), str(out_path))
    log(f"[out] wrote {out_path} in {time.time() - started:.1f}s total")

    if args.save_parts:
        parts = Path(args.save_parts)
        parts.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(body_hu, reference.affine, header),
                 str(parts / "body_ct.nii.gz"))
        nib.save(nib.Nifti1Image(head_hu, reference.affine, header),
                 str(parts / "head_vae_patch_ct.nii.gz"))
        (parts / "blend_stats.json").write_text(json.dumps(stats, indent=2))
        log(f"[out] parts -> {parts}")


if __name__ == "__main__":
    main()
