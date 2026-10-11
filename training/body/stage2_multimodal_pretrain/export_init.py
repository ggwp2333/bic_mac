"""
Wrap a stage-2 checkpoint as a `finetune_v1` initialisation.

Unlike stage 1, no surgery is needed: this model *is* the downstream model —
4 channels, width 96, same architecture — so the whole state dict transfers,
stem included. That is the point of stage 2. This script exists to verify that
claim rather than to perform a conversion, and to stamp provenance onto the
file the fine-tune will read.

Usage:
    python export_init.py --checkpoint outputs/checkpoints/epoch_0120.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unet import build_model, describe_model

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=BASE_DIR / "outputs" / "stage2_init.pth")
    ap.add_argument("--in-channels", type=int, default=4)
    ap.add_argument("--base-channels", type=int, default=None)
    a = ap.parse_args()

    path = a.checkpoint if a.checkpoint.is_absolute() else BASE_DIR / a.checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    saved = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    base_channels = a.base_channels or int(saved.get("base_channels", 96))
    model = build_model(in_channels=a.in_channels, base_channels=base_channels)

    # strict=True is the assertion: if the stem or anything else disagrees with
    # the downstream architecture, this raises rather than silently shipping a
    # partially-initialised model.
    model.load_state_dict(state, strict=True)
    moved = sum(v.numel() for v in state.values())

    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "pretrained_from": str(path),
        "pretrained_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "in_channels": a.in_channels,
        "base_channels": base_channels,
        "stage": 2,
        "stage1_init": saved.get("init_checkpoint"),
        "transferred_parameters": moved,
    }, a.out)

    print(f"source     : {path}")
    print(f"target     : in_channels={a.in_channels}, width={base_channels}, "
          f"{describe_model(model)}")
    print(f"transferred: {moved:,} parameters — the entire network, stem included")
    print(f"wrote      : {a.out}")
    print(json.dumps({
        "next": [
            "in ../finetune/config.yaml set:",
            "  arm: pretrained_stage2",
            f"  init_checkpoint: ../stage2_multimodal_pretrain/{a.out.relative_to(BASE_DIR)}",
            "  output_dir: outputs_stage2",
            "keep modality dropout ON downstream if it is added there — the "
            "stem was trained with it",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
