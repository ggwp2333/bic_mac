"""Stage-2 pretraining: supervised multimodal CT synthesis (plan.md §3.13).

Four real input channels in, CT out, over 171 scans — 75 BIC-MAC with all four
channels and 96 QUADRA_HC contributing pseudo-NAC PET only. Modality dropout
throughout, so the missing channels are ordinary rather than a special case.

Two things distinguish this from stage 1 (`pretrain_v1`):

  * **The stem transfers.** Stage 1 read one channel (a corrupted CT), so the
    layer that sees the challenge's actual inputs was thrown away on export.
    This model is the downstream model, exactly, and its whole state dict loads
    into `finetune_v1` with strict=True.
  * **It is supervised, not self-supervised.** The target is the CT the model is
    actually asked to predict, which is why the objective is the same mu-space
    tissue-weighted MAE the downstream run uses.

    python prepare_corpus.py
    torchrun --standalone --nproc_per_node=4 train.py
    python export_init.py --checkpoint outputs/checkpoints/epoch_0120.pth

**What the held-out number here is not.** It is measured on 6 held-out QUADRA
subjects with PET only, on a corpus this model was pretrained for. It says the
pretraining converged; it does not predict the leaderboard. Only the downstream
`finetune_v1` run against the already-banked stage-1 arm does that.
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
EXPERIMENT = "pretrain_v2_multimodal_supervised"
torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path,
                   help="checkpoint to continue from; relative to pretrain_v2/")
    p.add_argument("--resume-epoch", type=int)
    p.add_argument("--epochs", type=int, help="override config; smoke tests only")
    p.add_argument("--max-steps-per-epoch", type=int, default=0,
                   help="cap optimisation steps per epoch; smoke tests only")
    return p.parse_args()


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def resolve_from_base(value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else BASE_DIR / p


def infer_resume_epoch(path: Path, requested, training_log: Path) -> int:
    if requested is not None:
        return requested
    stem = path.stem
    if stem.startswith("epoch_") and stem[6:].isdigit():
        return int(stem[6:])
    if training_log.is_file():
        rows = list(csv.DictReader(open(training_log, newline="")))
        done = [int(r["epoch"]) for r in rows if r.get("epoch", "").isdigit()]
        if done:
            return max(done)
    raise ValueError("cannot infer the completed epoch; pass --resume-epoch")


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return True, rank, world, local


def reduce_means(sums: dict, count: int, device, distributed: bool) -> dict:
    names = sorted(sums)
    t = torch.tensor([sums[n] for n in names] + [float(count)],
                     dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total = t[-1].clamp_min(1.0)
    return {n: float(t[i] / total) for i, n in enumerate(names)}


def save_training_plot(path: Path, history: dict) -> None:
    plt.figure(figsize=(7, 4))
    for name, values in history.items():
        xs = [i + 1 for i, v in enumerate(values)
              if v is not None and not (isinstance(v, float) and math.isnan(v))]
        ys = [v for v in values
              if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if xs:
            plt.plot(xs, ys, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(r"$\mu$-MAE in the scored region (cm$^{-1}$)")
    plt.title("Stage-2 multimodal pretraining")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def build_loader(cases, transform, cfg, distributed, world, rank, use_cuda, shuffle):
    ds = CacheDataset(data=cases, transform=transform,
                      cache_rate=float(cfg.get("cache_rate", 0.0)),
                      num_workers=int(cfg["num_workers"]), copy_cache=False)
    sampler = (DistributedSampler(ds, num_replicas=world, rank=rank,
                                  shuffle=shuffle, drop_last=False)
               if distributed else None)
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]),
                        shuffle=shuffle and sampler is None, sampler=sampler,
                        num_workers=int(cfg["num_workers"]), pin_memory=use_cuda,
                        persistent_workers=int(cfg["num_workers"]) > 0)
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
        raise ValueError(f"patch_size {patch_size} must be a multiple of 16 — "
                         "the encoder pools four times")

    output_dir = resolve_from_base(cfg["output_dir"])
    checkpoint_dir, log_dir, plot_dir = (output_dir / "checkpoints",
                                         output_dir / "logs",
                                         output_dir / "plots")
    training_log = log_dir / "train_log.csv"

    resume_path = resolve_from_base(args.resume) if args.resume else None
    completed = (infer_resume_epoch(resume_path, args.resume_epoch, training_log)
                 if resume_path is not None else 0)
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    if not 0 <= completed < epochs:
        raise ValueError(f"resume epoch must lie in [0, {epochs - 1}]")

    distributed, rank, world, local_rank = init_distributed()
    is_main = rank == 0
    device = (torch.device("cuda", local_rank) if torch.cuda.is_available()
              else torch.device("cpu"))
    use_cuda = device.type == "cuda"

    manifest = resolve_from_base(cfg["corpus"])
    if not manifest.is_file():
        raise FileNotFoundError(f"corpus not found at {manifest} — "
                                "run prepare_corpus.py first")
    train_cases = get_dataset(manifest, "train", list(cfg["train_datasets"]))
    val_cases = get_dataset(manifest, "val", list(cfg["train_datasets"]))

    md = cfg.get("modality_dropout", {})
    train_tf = get_transforms(
        patch_size, int(cfg["train_num_samples"]),
        float(cfg.get("mask_ratio", 0.25)), int(cfg.get("block_size", 16)),
        float(md.get("prob", 0.5)), float(md.get("per_channel_prob", 0.25)),
    )
    val_tf = get_validation_transforms(patch_size, int(cfg.get("val_num_samples", 2)))

    train_loader, train_sampler = build_loader(
        train_cases, train_tf, cfg, distributed, world, rank, use_cuda, True)
    val_loader = None
    if val_cases:
        val_loader, _ = build_loader(
            val_cases, val_tf, cfg, False, 1, 0, use_cuda, False)

    bare_model = build_model(in_channels=int(cfg.get("in_channels", 4)),
                             base_channels=int(cfg["base_channels"])).to(device)

    init_ckpt = cfg.get("init_checkpoint")
    if init_ckpt and resume_path is None:
        init_path = resolve_from_base(init_ckpt)
        if not init_path.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {init_path}")
        payload = torch.load(init_path, map_location=device, weights_only=True)
        bare_model.load_state_dict(
            payload.get("model_state_dict", payload), strict=True)
        if is_main:
            print(f"Initialised from stage 1: {init_path}")
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        bare_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)

    model = bare_model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            bare_model, device_ids=[local_rank] if use_cuda else None)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg["learning_rate"]), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if completed:
        lr = float(cfg["learning_rate"]) * (
            1.0 + math.cos(math.pi * completed / epochs)) / 2.0
        for g in optimizer.param_groups:
            g["lr"] = lr
        scheduler.last_epoch = completed
        scheduler._step_count = completed + 1
        scheduler._last_lr = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    accum = max(1, int(cfg.get("grad_accum_steps", 1)))
    ckpt_every = int(cfg.get("checkpoint_every_epochs", 10))
    val_every = int(cfg.get("validate_every_epochs", 2))
    val_seed = int(cfg.get("val_seed", 1234))

    if is_main:
        for d in (checkpoint_dir, log_dir, plot_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not training_log.exists() or not training_log.stat().st_size:
            with open(training_log, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["epoch", "train_total", "train_plain_mu", "train_weighted_mu",
                     "val_total", "val_plain_mu", "learning_rate"])
        print(f"Device: {device} (world_size={world})")
        print(f"Train: {corpus_summary(train_cases)}")
        print(f"Val:   {corpus_summary(val_cases) if val_cases else 'none'}")
        print(f"Model: width {cfg['base_channels']}, {describe_model(bare_model)}, "
              f"in_channels={cfg.get('in_channels', 4)}")
        print(f"Modality dropout: prob={md.get('prob', 0.5)}, "
              f"per_channel={md.get('per_channel_prob', 0.25)} | "
              f"block mask {cfg.get('mask_ratio', 0.25)} @ {cfg.get('block_size', 16)}")
        print(f"Effective patch batch: "
              f"{int(cfg['batch_size']) * int(cfg['train_num_samples']) * accum * world}")
    if distributed:
        dist.barrier()

    history: dict = {"train_total": [], "val_total": []}
    if training_log.is_file() and completed:
        for row in csv.DictReader(open(training_log, newline="")):
            if row.get("epoch", "").isdigit() and int(row["epoch"]) <= completed:
                history["train_total"].append(float(row["train_total"]))
                history["val_total"].append(
                    float(row["val_total"]) if row.get("val_total") else float("nan"))

    def run_batch(batch):
        target = batch["ct"].to(device, non_blocking=True)
        mask = batch["loss_mask"].to(device, non_blocking=True).bool()
        inputs = batch["input"].to(device, non_blocking=True)
        ctx = torch.amp.autocast("cuda") if use_cuda else nullcontext()
        with ctx:
            pred = model(inputs)
            total, plain, weighted = blended_umap_mae_loss(
                pred, target, mask, tissue_cfg)
        return total, plain, weighted, int(mask.sum())

    @torch.no_grad()
    def validate() -> dict:
        if val_loader is None:
            return {}
        model.eval()
        val_tf.set_random_state(seed=val_seed)
        sums = {"val_total": 0.0, "val_plain_mu": 0.0}
        n = 0
        for batch in val_loader:
            for i in range(batch["ct"].shape[0]):
                micro = {k: v[i:i + 1] for k, v in batch.items()
                         if hasattr(v, "shape")}
                total, plain, _, nv = run_batch(micro)
                if nv == 0:
                    continue
                sums["val_total"] += float(total.detach())
                sums["val_plain_mu"] += float(plain.detach())
                n += 1
        model.train()
        return {k: v / n for k, v in sums.items()} if n else {}

    for epoch_index in range(completed, epochs):
        epoch = epoch_index + 1
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_index)
        sums = {"total": 0.0, "plain_mu": 0.0, "weighted_mu": 0.0}
        count = steps = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader) if is_main else train_loader

        for batch_index, batch in enumerate(progress):
            n_micro = batch["ct"].shape[0]
            step_now = ((batch_index + 1) % accum == 0
                        or (batch_index + 1) == len(train_loader))
            for i in range(n_micro):
                micro = {k: v[i:i + 1] for k, v in batch.items()
                         if hasattr(v, "shape")}
                sync_now = step_now and i + 1 == n_micro
                sync_ctx = (model.no_sync()
                            if distributed and not sync_now else nullcontext())
                with sync_ctx:
                    total, plain, weighted, nv = run_batch(micro)
                    if nv == 0:
                        continue
                    scaler.scale(total / (accum * n_micro)).backward()
                sums["total"] += float(total.detach())
                sums["plain_mu"] += float(plain.detach())
                sums["weighted_mu"] += float(weighted.detach())
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
                    f"μ-MAE {sums['plain_mu'] / max(count, 1):.5f}")

        means = reduce_means(sums, count, device, distributed)
        scheduler.step()
        val_means = validate() if (
            is_main and val_every and epoch % val_every == 0) else {}

        if is_main:
            history["train_total"].append(means["total"])
            history["val_total"].append(val_means.get("val_total", float("nan")))
            print(f"Epoch {epoch:03d}: train μ-MAE={means['plain_mu']:.6f} "
                  f"total={means['total']:.6f}"
                  + (f" | val μ-MAE={val_means['val_plain_mu']:.6f}"
                     if val_means else ""))
            with open(training_log, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, f"{means['total']:.8f}", f"{means['plain_mu']:.8f}",
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
                    "in_channels": int(cfg.get("in_channels", 4)),
                    "experiment": EXPERIMENT,
                    "patch_size": patch_size,
                    "modality_dropout": md,
                    "mask_ratio": float(cfg.get("mask_ratio", 0.25)),
                    "init_checkpoint": str(init_ckpt) if init_ckpt else None,
                },
                "train_metrics": means,
                "val_metrics": val_means,
            }
            if cfg.get("keep_last_checkpoint", True):
                torch.save(payload, checkpoint_dir / "last.pth")
            if ckpt_every and epoch % ckpt_every == 0:
                torch.save(payload, checkpoint_dir / f"epoch_{epoch:04d}.pth")
        if distributed:
            dist.barrier()

    if is_main:
        summary = {
            "experiment": EXPERIMENT, "epochs": epochs,
            "train_scans": len(train_cases), "val_scans": len(val_cases),
            "base_channels": int(cfg["base_channels"]),
            "final_train_total": history["train_total"][-1] if history["train_total"] else None,
            "final_val_total": next(
                (v for v in reversed(history["val_total"])
                 if v is not None and not math.isnan(v)), None),
        }
        with open(log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
