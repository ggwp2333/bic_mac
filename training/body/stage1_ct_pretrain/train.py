"""Self-supervised CT pretraining on BIC-MAC + Vienna QUADRA_HC (plan.md §3.12).

The task is masked reconstruction: blank half of a CT patch in 16-voxel cubes
and predict it back, scored in mu at 511 keV with the same tissue weighting the
downstream run uses. No PET, no MRI, no topogram, no attenuation modelling — the
point of this arm is that it buys the external anatomy without depending on any
of the input-synthesis work that Option A needs.

Single GPU:
    python prepare_corpus.py
    python train.py

Four GPUs:
    torchrun --standalone --nproc_per_node=4 train.py

Then transfer the weights:
    python export_encoder.py --checkpoint outputs_pretrain/checkpoints/epoch_0200.pth

**What this run cannot tell you.** Held-out reconstruction loss is a pretext
metric. It says the network learned to inpaint CT; it does not say the fine-tune
will score better. The only readout that counts is the downstream Codabench run
against a from-scratch control trained at the same width — see the README.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from monai.data import CacheDataset, DataLoader
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from dataset import corpus_summary, get_dataset
from losses import blended_umap_mae_loss, validate_tissue_loss_config
from transforms import get_transforms, get_validation_transforms
from unet import build_model, describe_model

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
EXPERIMENT = "pretrain_v1_ct_masked_reconstruction"
torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        help="checkpoint to resume from; relative paths resolve from pretrain_v1/",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        help="last completed epoch; normally inferred from epoch_XXXX.pth",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="cap optimisation steps per epoch; for smoke tests only",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="override config epochs; for smoke tests only",
    )
    return parser.parse_args()


def load_config() -> dict:
    with open(CONFIG_PATH) as file:
        return yaml.safe_load(file) or {}


def resolve_from_base(path_value) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def infer_resume_epoch(checkpoint_path: Path, requested: int | None,
                       training_log: Path) -> int:
    if requested is not None:
        return requested
    stem = checkpoint_path.stem
    if stem.startswith("epoch_") and stem[6:].isdigit():
        return int(stem[6:])
    if training_log.is_file():
        with open(training_log, newline="") as file:
            rows = list(csv.DictReader(file))
        completed = [int(r["epoch"]) for r in rows if r.get("epoch", "").isdigit()]
        if completed:
            return max(completed)
    raise ValueError("cannot infer the completed epoch; pass --resume-epoch")


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def reduce_means(sums: dict, count: int, device, distributed: bool) -> dict:
    names = sorted(sums)
    tensor = torch.tensor(
        [sums[n] for n in names] + [float(count)],
        dtype=torch.float64, device=device,
    )
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total = tensor[-1].clamp_min(1.0)
    return {n: float(tensor[i] / total) for i, n in enumerate(names)}


def loss_mask_for(batch_body: torch.Tensor, batch_corrupt: torch.Tensor,
                  corrupted_only: bool, inside_body_only: bool) -> torch.Tensor:
    """Which voxels the reconstruction is scored on."""
    mask = torch.ones_like(batch_body, dtype=torch.bool)
    if inside_body_only:
        mask &= batch_body.bool()
    if corrupted_only:
        mask &= batch_corrupt.bool()
    return mask


def save_training_plot(path: Path, history: dict[str, list]) -> None:
    plt.figure(figsize=(7, 4))
    for name, values in history.items():
        xs = [i + 1 for i, v in enumerate(values) if v is not None and not (
            isinstance(v, float) and math.isnan(v))]
        ys = [v for v in values if v is not None and not (
            isinstance(v, float) and math.isnan(v))]
        if xs:
            plt.plot(xs, ys, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(r"$\mu$-MAE on scored voxels (cm$^{-1}$)")
    plt.title("CT masked-reconstruction pretraining")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def build_loader(cases, transform, cfg, distributed, world_size, rank,
                 use_cuda, shuffle):
    dataset = CacheDataset(
        data=cases,
        transform=transform,
        cache_rate=float(cfg.get("cache_rate", 0.0)),
        num_workers=int(cfg["num_workers"]),
        copy_cache=False,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                           shuffle=shuffle, drop_last=False)
        if distributed else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(cfg["num_workers"]),
        pin_memory=use_cuda,
        persistent_workers=int(cfg["num_workers"]) > 0,
    )
    return loader, sampler


def main() -> None:
    args = parse_args()
    if args.resume_epoch is not None and args.resume is None:
        raise ValueError("--resume-epoch requires --resume")

    cfg = load_config()
    if cfg.get("experiment") != EXPERIMENT:
        raise ValueError(f"config.yaml must set experiment: {EXPERIMENT}")

    epochs = int(args.epochs or cfg["epochs"])
    tissue_cfg = validate_tissue_loss_config(cfg["tissue_loss"])
    patch_size = list(cfg["patch_size"])
    if any(int(s) % 16 for s in patch_size):
        raise ValueError(
            f"patch_size {patch_size} must be a multiple of 16 — the encoder "
            "pools four times"
        )

    output_dir = resolve_from_base(cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    plot_dir = output_dir / "plots"
    training_log = log_dir / "train_log.csv"

    resume_path = resolve_from_base(args.resume) if args.resume else None
    completed_epoch = (
        infer_resume_epoch(resume_path, args.resume_epoch, training_log)
        if resume_path is not None else 0
    )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    if not 0 <= completed_epoch < epochs:
        raise ValueError(f"resume epoch must lie in [0, {epochs - 1}]")

    distributed, rank, world_size, local_rank = init_distributed()
    is_main = rank == 0
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available() else torch.device("cpu")
    )
    use_cuda = device.type == "cuda"

    manifest = resolve_from_base(cfg["corpus"])
    if not manifest.is_file():
        raise FileNotFoundError(
            f"corpus manifest not found at {manifest} — run prepare_corpus.py first"
        )
    train_cases = get_dataset(manifest, split="train",
                             datasets=list(cfg["train_datasets"]))
    val_cases = get_dataset(manifest, split="val",
                            datasets=list(cfg["train_datasets"]))

    train_transform = get_transforms(
        patch_size,
        int(cfg["train_num_samples"]),
        float(cfg["mask_ratio"]),
        int(cfg["block_size"]),
        float(cfg.get("noise_std", 0.0)),
    )
    val_transform = get_validation_transforms(
        patch_size,
        int(cfg.get("val_num_samples", 2)),
        float(cfg["mask_ratio"]),
        int(cfg["block_size"]),
        float(cfg.get("noise_std", 0.0)),
    )

    train_loader, train_sampler = build_loader(
        train_cases, train_transform, cfg, distributed, world_size, rank,
        use_cuda, shuffle=True,
    )
    val_loader = None
    if val_cases:
        val_loader, _ = build_loader(
            val_cases, val_transform, cfg, False, 1, 0, use_cuda, shuffle=False,
        )

    bare_model = build_model(
        in_channels=int(cfg.get("in_channels", 1)),
        base_channels=int(cfg["base_channels"]),
    ).to(device)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        bare_model.load_state_dict(
            checkpoint.get("model_state_dict", checkpoint), strict=True
        )
    model = bare_model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            bare_model, device_ids=[local_rank] if use_cuda else None
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if completed_epoch:
        resumed_lr = float(cfg["learning_rate"]) * (
            1.0 + math.cos(math.pi * completed_epoch / epochs)
        ) / 2.0
        for group in optimizer.param_groups:
            group["lr"] = resumed_lr
        scheduler.last_epoch = completed_epoch
        scheduler._step_count = completed_epoch + 1
        scheduler._last_lr = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    accumulation_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    corrupted_only = bool(cfg.get("loss_on_corrupted_only", True))
    inside_body_only = bool(cfg.get("loss_inside_body_only", True))
    checkpoint_every = int(cfg.get("checkpoint_every_epochs", 10))
    validate_every = int(cfg.get("validate_every_epochs", 2))
    val_seed = int(cfg.get("val_seed", 1234))

    if is_main:
        for directory in (checkpoint_dir, log_dir, plot_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if not training_log.exists() or not training_log.stat().st_size:
            with open(training_log, "w", newline="") as file:
                csv.writer(file).writerow(
                    ["epoch", "train_total", "train_plain_mu",
                     "train_weighted_mu", "val_total", "val_plain_mu",
                     "learning_rate"]
                )
        print(f"Device: {device} (world_size={world_size})")
        print(f"Train corpus: {corpus_summary(train_cases)}")
        print(f"Val corpus:   {corpus_summary(val_cases) if val_cases else 'none'}")
        print(f"Model: width {cfg['base_channels']}, "
              f"{describe_model(bare_model)}, in_channels={cfg.get('in_channels', 1)}")
        print(f"Pretext: mask {100 * float(cfg['mask_ratio']):.0f}% in "
              f"{cfg['block_size']}-voxel blocks, patch {patch_size}, "
              f"loss on {'corrupted' if corrupted_only else 'all'} voxels"
              f"{' inside body' if inside_body_only else ''}")
        print(f"Effective patch batch: "
              f"{int(cfg['batch_size']) * int(cfg['train_num_samples']) * accumulation_steps * world_size}")
    if distributed:
        dist.barrier()

    history: dict[str, list] = {"train_total": [], "val_total": []}
    if training_log.is_file() and completed_epoch:
        with open(training_log, newline="") as file:
            for row in csv.DictReader(file):
                if not row.get("epoch", "").isdigit():
                    continue
                if int(row["epoch"]) > completed_epoch:
                    continue
                history["train_total"].append(float(row["train_total"]))
                history["val_total"].append(
                    float(row["val_total"]) if row.get("val_total") else float("nan")
                )

    def run_batch(batch, train: bool):
        target = batch["ct"].to(device, non_blocking=True)
        body = batch["body_mask"].to(device, non_blocking=True)
        corrupt = batch["corrupt_mask"].to(device, non_blocking=True)
        inputs = batch["input"].to(device, non_blocking=True)
        mask = loss_mask_for(body, corrupt, corrupted_only, inside_body_only)
        autocast_context = (
            torch.amp.autocast("cuda") if use_cuda else nullcontext()
        )
        with autocast_context:
            prediction = model(inputs)
            total, plain_mu, weighted_mu = blended_umap_mae_loss(
                prediction, target, mask, tissue_cfg
            )
        return total, plain_mu, weighted_mu, int(mask.sum())

    @torch.no_grad()
    def validate() -> dict:
        if val_loader is None:
            return {}
        model.eval()
        # Same crops and same blanked blocks every epoch, so the curve moves
        # only when the model does.
        val_transform.set_random_state(seed=val_seed)
        sums = {"val_total": 0.0, "val_plain_mu": 0.0}
        count = 0
        for batch in val_loader:
            n_micro = batch["ct"].shape[0]
            for i in range(n_micro):
                micro = {k: v[i:i + 1] for k, v in batch.items()
                         if hasattr(v, "shape")}
                total, plain_mu, _, n_voxels = run_batch(micro, train=False)
                if n_voxels == 0:
                    continue
                sums["val_total"] += float(total.detach())
                sums["val_plain_mu"] += float(plain_mu.detach())
                count += 1
        model.train()
        if count == 0:
            return {}
        return {k: v / count for k, v in sums.items()}

    for epoch_index in range(completed_epoch, epochs):
        epoch = epoch_index + 1
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_index)

        sums = {"total": 0.0, "plain_mu": 0.0, "weighted_mu": 0.0}
        count = 0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader) if is_main else train_loader

        for batch_index, batch in enumerate(progress):
            n_micro = batch["ct"].shape[0]
            step_now = (
                (batch_index + 1) % accumulation_steps == 0
                or (batch_index + 1) == len(train_loader)
            )
            for micro_index in range(n_micro):
                micro = {k: v[micro_index:micro_index + 1]
                         for k, v in batch.items() if hasattr(v, "shape")}
                sync_now = step_now and micro_index + 1 == n_micro
                sync_context = (
                    model.no_sync() if distributed and not sync_now else nullcontext()
                )
                with sync_context:
                    total, plain_mu, weighted_mu, n_voxels = run_batch(micro, True)
                    if n_voxels == 0:
                        # a crop entirely outside the body has nothing to score
                        continue
                    scaler.scale(
                        total / (accumulation_steps * n_micro)
                    ).backward()
                sums["total"] += float(total.detach())
                sums["plain_mu"] += float(plain_mu.detach())
                sums["weighted_mu"] += float(weighted_mu.detach())
                count += 1

            if step_now:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                    break
            if is_main:
                progress.set_description(
                    f"μ-MAE {sums['plain_mu'] / max(count, 1):.5f}"
                )

        means = reduce_means(sums, count, device, distributed)
        scheduler.step()

        val_means = validate() if (
            is_main and validate_every and epoch % validate_every == 0
        ) else {}

        if is_main:
            history["train_total"].append(means["total"])
            history["val_total"].append(
                val_means.get("val_total", float("nan"))
            )
            print(
                f"Epoch {epoch:03d}: train μ-MAE={means['plain_mu']:.6f} "
                f"total={means['total']:.6f}"
                + (f" | val μ-MAE={val_means['val_plain_mu']:.6f}"
                   if val_means else "")
            )
            with open(training_log, "a", newline="") as file:
                csv.writer(file).writerow([
                    epoch,
                    f"{means['total']:.8f}",
                    f"{means['plain_mu']:.8f}",
                    f"{means['weighted_mu']:.8f}",
                    f"{val_means['val_total']:.8f}" if val_means else "",
                    f"{val_means['val_plain_mu']:.8f}" if val_means else "",
                    f"{optimizer.param_groups[0]['lr']:.12g}",
                ])
            save_training_plot(plot_dir / "training_curve.png", history)

            payload = {
                "epoch": epoch,
                "model_state_dict": bare_model.state_dict(),
                "config": {
                    "base_channels": int(cfg["base_channels"]),
                    "in_channels": int(cfg.get("in_channels", 1)),
                    "experiment": EXPERIMENT,
                    "patch_size": patch_size,
                    "mask_ratio": float(cfg["mask_ratio"]),
                    "block_size": int(cfg["block_size"]),
                },
                "train_metrics": means,
                "val_metrics": val_means,
            }
            if cfg.get("keep_last_checkpoint", True):
                torch.save(payload, checkpoint_dir / "last.pth")
            if checkpoint_every and epoch % checkpoint_every == 0:
                torch.save(payload, checkpoint_dir / f"epoch_{epoch:04d}.pth")

        if distributed:
            dist.barrier()

    if is_main:
        summary = {
            "experiment": EXPERIMENT,
            "epochs": epochs,
            "train_scans": len(train_cases),
            "val_scans": len(val_cases),
            "base_channels": int(cfg["base_channels"]),
            "final_train_total": history["train_total"][-1] if history["train_total"] else None,
            "final_val_total": next(
                (v for v in reversed(history["val_total"])
                 if v is not None and not math.isnan(v)), None
            ),
        }
        with open(log_dir / "summary.json", "w") as file:
            json.dump(summary, file, indent=2)
        print(json.dumps(summary, indent=2))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
