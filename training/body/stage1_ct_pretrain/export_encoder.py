"""
Turn a pretraining checkpoint into an initialisation for the 4-channel model.

The pretraining network reads one channel (a corrupted CT); the downstream
network reads four (NAC-PET, Dixon in/out, topogram). Every weight transfers
except the stem — `enc1.block.conv1.weight` and `enc1.block.skip.weight`, whose
input dimension is the channel count. That is the known and stated limit of this
arm: it buys a decoder and a deep encoder that know whole-body CT anatomy, not a
stem that knows the challenge's inputs.

Two ways to fill the stem:

``--stem reinit`` (default)
    Fresh Kaiming init, the honest choice. Channel 0 of the downstream model is
    NAC-PET, not CT, so a pretrained CT filter has no claim on it.

``--stem replicate``
    Copy the pretrained single-channel filter to all four inputs, divided by the
    channel count so the pre-activation scale is preserved. Sometimes converges
    faster; it does assume the four inputs are interchangeable, which they are
    not. Worth trying only as a second arm.

The output is a full state dict for the 4-channel model, so the downstream
trainer can load it with ``strict=True``.

Usage:
    python export_encoder.py --checkpoint outputs_pretrain/checkpoints/epoch_0200.pth
    python export_encoder.py --checkpoint ... --in-channels 4 --stem replicate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unet import build_model, describe_model

BASE_DIR = Path(__file__).resolve().parent
STEM_KEYS = ("enc1.block.conv1.weight", "enc1.block.skip.weight")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=BASE_DIR / "outputs_pretrain" / "pretrained_init.pth")
    ap.add_argument("--in-channels", type=int, default=4)
    ap.add_argument("--base-channels", type=int, default=None,
                    help="defaults to the width recorded in the checkpoint")
    ap.add_argument("--stem", choices=("reinit", "replicate"), default="reinit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    path = args.checkpoint
    if not path.is_absolute():
        path = BASE_DIR / path
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    source = checkpoint.get("model_state_dict", checkpoint)
    saved_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    base_channels = args.base_channels or int(saved_cfg.get("base_channels", 96))
    target_model = build_model(
        in_channels=args.in_channels, base_channels=base_channels
    )
    target = target_model.state_dict()

    transferred, skipped, mismatched = [], [], []
    for key, value in target.items():
        if key not in source:
            skipped.append(key)
            continue
        src = source[key]
        if key in STEM_KEYS:
            if args.stem == "replicate":
                if src.shape[1] != 1:
                    raise ValueError(
                        f"--stem replicate needs a 1-channel source stem, "
                        f"got {tuple(src.shape)} for {key}"
                    )
                target[key] = (
                    src.repeat(1, args.in_channels, 1, 1, 1) / args.in_channels
                )
                transferred.append(f"{key} (replicated x{args.in_channels})")
            else:
                skipped.append(f"{key} (reinit)")
            continue
        if src.shape != value.shape:
            mismatched.append(f"{key}: {tuple(src.shape)} vs {tuple(value.shape)}")
            continue
        target[key] = src
        transferred.append(key)

    if mismatched:
        raise ValueError(
            "checkpoint does not match the target architecture:\n  "
            + "\n  ".join(mismatched)
            + "\nCheck --base-channels."
        )

    target_model.load_state_dict(target, strict=True)
    n_src = sum(v.numel() for k, v in source.items())
    n_moved = sum(
        target[k].numel() for k in target
        if k in source and k not in STEM_KEYS and source[k].shape == target[k].shape
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": target_model.state_dict(),
        "pretrained_from": str(path),
        "pretrained_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "in_channels": args.in_channels,
        "base_channels": base_channels,
        "stem_policy": args.stem,
        "transferred_parameters": n_moved,
        "source_parameters": n_src,
    }
    torch.save(payload, args.out)

    print(f"source     : {path}")
    print(f"target     : in_channels={args.in_channels}, width={base_channels}, "
          f"{describe_model(target_model)}")
    n_stem = sum(
        target[k].numel() for k in STEM_KEYS if k in target
    ) if args.stem == "reinit" else 0
    print(f"transferred: {n_moved:,} of {n_src:,} parameters; "
          f"{n_stem:,} left to fresh init")
    print("             the stem is a rounding error in parameter count and "
          "not in function —")
    print("             it is the only layer that ever sees the challenge's "
          "actual inputs")
    print(f"stem       : {args.stem} ({', '.join(k for k in STEM_KEYS)})")
    if skipped:
        print(f"not copied : {len(skipped)} tensors — {skipped[:4]}"
              + (" ..." if len(skipped) > 4 else ""))
    print(f"wrote      : {args.out}")
    print(json.dumps({
        "next": "point the downstream trainer's --resume at this file with "
                "--resume-epoch 0, and set its base_channels to "
                f"{base_channels}",
    }, indent=2))


if __name__ == "__main__":
    main()
