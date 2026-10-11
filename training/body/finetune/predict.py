import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
import nibabel as nib
import numpy as np

from monai.inferers import sliding_window_inference
from unet import build_model
from dataset import get_subject_features
from transforms import DEFAULT_INPUT_KEYS, get_inference_transforms


CONFIG_PATH = Path(__file__).parent / "config.yaml"
PATCH_SIZE = (192, 192, 192)
SW_BATCH = 1 # You can increase this if to speed up inference at the cost of VRAM
OVERLAP = 0.5


def parse_patch_size(value):
    parts = tuple(int(v) for v in value.replace(",", " ").split())
    if len(parts) == 1:
        parts = parts * 3
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("patch size must be one integer or three integers")
    if any(value <= 0 for value in parts):
        raise argparse.ArgumentTypeError("patch dimensions must be positive")
    return parts


def normalized_ct_to_mu(pred_norm):
    """Convert normalized network CT output to 511-keV attenuation in fp32."""
    hu1000 = pred_norm.float() * 3000.0
    mu = torch.where(
        hu1000 < 1047.0,
        9.6e-5 * hu1000,
        5.10e-5 * hu1000 + 4.71e-2,
    )
    return mu.clamp_min_(0.0)


def mu_to_hu(mu):
    """Invert the challenge Carney conversion for a submission in HU."""
    mu = np.asarray(mu, dtype=np.float32)
    high_branch_start = 5.10e-5 * 1047.0 + 4.71e-2
    hu1000 = np.where(
        mu < high_branch_start,
        mu / 9.6e-5,
        (mu - 4.71e-2) / 5.10e-5,
    )
    return hu1000 - 1000.0


def infer_once(x, model, patch_size, sw_batch, overlap, mode, device, progress):
    return sliding_window_inference(
        x,
        patch_size,
        sw_batch,
        model,
        overlap=overlap,
        mode=mode,
        progress=progress,
        sw_device=device,
        device="cpu",
    )


def predict(features_dir, out_path, patch_size=PATCH_SIZE, sw_batch=SW_BATCH,
            overlap=OVERLAP, mode="gaussian", checkpoint=None, tta="none"):

    if sw_batch < 1:
        raise ValueError("sw_batch must be at least 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if tta not in {"none", "lr"}:
        raise ValueError("tta must be 'none' or 'lr'")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    input_keys = tuple(cfg.get("input_keys", DEFAULT_INPUT_KEYS))
    transforms = get_inference_transforms(input_keys)
    output_dir = Path(cfg.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir
    model_path = (
        Path(checkpoint)
        if checkpoint is not None
        else output_dir / "checkpoints" / "last_model.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    print(f"Input channels ({len(input_keys)}): {list(input_keys)}")
    print(f"Loading checkpoint: {model_path}")
    # Width must match the trained checkpoint; taking the builder default here
    # is how a width-96 checkpoint silently fails to load.
    base_channels = int(cfg.get("base_channels", 96))
    print(f"Model width: {base_channels}")
    model = build_model(
        in_channels=len(input_keys), base_channels=base_channels
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    subject = get_subject_features(features_dir)
    data = transforms(subject)

    x = data["input"].unsqueeze(0).to(device)
    print(
        "Sliding window inference: "
        f"patch_size={patch_size}, sw_batch={sw_batch}, overlap={overlap}, "
        f"mode={mode}, tta={tta}"
    )
    autocast_context = (
        torch.amp.autocast("cuda") if use_cuda else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        predictions = [
            infer_once(
                x, model, patch_size, sw_batch, overlap, mode, device, progress=True
            )
        ]
        if tta == "lr":
            # Tensor spatial axes are [X, Y, Z]; NIfTI X is left-right here.
            # Undo the flip on the prediction before averaging.
            print("TTA pass: left-right flip")
            flipped_prediction = infer_once(
                torch.flip(x, dims=(2,)),
                model,
                patch_size,
                sw_batch,
                overlap,
                mode,
                device,
                progress=True,
            )
            predictions.append(torch.flip(flipped_prediction, dims=(2,)))

        if len(predictions) == 1:
            # Preserve byte-for-byte-equivalent behavior for the default path.
            pred_hu = predictions[0].float().numpy()[0, 0] * 3000.0 - 1000.0
        else:
            # The challenge and the reconstruction both operate on mu, so TTA
            # predictions are averaged in attenuation space rather than HU.
            mean_mu = torch.stack(
                [normalized_ct_to_mu(prediction) for prediction in predictions]
            ).mean(dim=0)
            pred_hu = mu_to_hu(mean_mu.numpy()[0, 0])

    del predictions
    if use_cuda:
        torch.cuda.empty_cache()

    if not np.isfinite(pred_hu).all():
        raise RuntimeError("prediction contains NaN or infinite values")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    affine = data["nacpet"].meta["affine"].numpy()

    print(
        f"Output HU: min={float(pred_hu.min()):.1f}, "
        f"max={float(pred_hu.max()):.1f}, mean={float(pred_hu.mean()):.1f}"
    )
    print("Saving...")
    nib.save(
        nib.Nifti1Image(pred_hu.astype(np.float32, copy=False), affine),
        out_path,
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline-v3 sliding-window inference, optionally averaging "
            "the original and left-right-flipped predictions in mu-space."
        )
    )
    parser.add_argument("--features_dir", required=True, help="Path to the subject's features/ folder")
    parser.add_argument("--output_ct",    required=True, help="Path to save the predicted pseudo-CT (e.g. './psudeo-ct.nii.gz')")
    parser.add_argument("--patch_size", type=parse_patch_size, default=PATCH_SIZE, help="Sliding-window patch size. Use one value or three values, e.g. 192 or '192 192 192'.")
    parser.add_argument("--sw_batch", type=int, default=SW_BATCH, help="Sliding-window batch size.")
    parser.add_argument("--overlap", type=float, default=OVERLAP, help="Sliding-window overlap fraction.")
    parser.add_argument("--mode", choices=["constant", "gaussian"], default="gaussian", help="Sliding-window blending mode.")
    parser.add_argument(
        "--tta",
        choices=["none", "lr"],
        default="none",
        help="Test-time augmentation: none or left-right flip averaging in mu-space.",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to load (default: <config output_dir>/checkpoints/last_model.pth).")
    args = parser.parse_args()

    predict(args.features_dir, args.output_ct, args.patch_size, args.sw_batch,
            args.overlap, args.mode, args.checkpoint, args.tta)
