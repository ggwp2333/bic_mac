"""Train width-64 baseline-v7 on all 75 labels and evaluate late checkpoints.

Single GPU:
    python train.py

Four GPUs:
    torchrun --standalone --nproc_per_node=4 train.py

All labeled subjects are used for optimization. At epochs 205, 210, ..., 300,
the model is permanently checkpointed, all four unlabeled validation subjects
are predicted, and a CT-only Codabench submission is launched asynchronously.
The submission worker waits for scores and stores the returned JSON beside that
epoch's predictions. No in-sample metric is called validation and no
``best_model.pth`` is selected from training subjects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import yaml
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from dataset import get_dataset, get_inference_dataset
from losses import (
    blended_umap_mae_loss,
    tissue_weight_map,
    validate_tissue_loss_config,
)
from transforms import (
    DEFAULT_INPUT_KEYS,
    get_inference_transforms,
    get_transforms,
)
from unet import build_model, describe_model


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
SUBMIT_WORKER = BASE_DIR / "submit_checkpoint.py"
VAL_SW_BATCH = 1
VAL_OVERLAP = 0.5
VAL_MODE = "gaussian"
torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "Model or training-state checkpoint to resume from. Relative "
            "paths are resolved from this directory."
        ),
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        help=(
            "Last completed epoch. Normally inferred from epoch_XXXX.pth or "
            "outputs/logs/train_log.csv; use this to override inference."
        ),
    )
    return parser.parse_args()


def load_config() -> dict:
    with open(CONFIG_PATH) as file:
        return yaml.safe_load(file) or {}


def resolve_from_base(path_value) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def infer_resume_epoch(
    checkpoint_path: Path,
    requested_epoch: int | None,
    training_log: Path,
) -> int:
    if requested_epoch is not None:
        return requested_epoch
    stem = checkpoint_path.stem
    if stem.startswith("epoch_") and stem[6:].isdigit():
        return int(stem[6:])
    if training_log.is_file():
        with open(training_log, newline="") as file:
            rows = list(csv.DictReader(file))
        completed = [
            int(row["epoch"])
            for row in rows
            if row.get("epoch", "").isdigit()
        ]
        if completed:
            return max(completed)
    raise ValueError(
        "cannot infer the completed epoch; pass --resume-epoch explicitly"
    )


def load_training_history(
    training_log: Path,
    completed_epoch: int,
) -> dict[str, list[float]]:
    history = {"total": [], "plain_mu": [], "weighted_mu": []}
    if not training_log.is_file():
        return history
    with open(training_log, newline="") as file:
        for row in csv.DictReader(file):
            if not row.get("epoch", "").isdigit():
                continue
            if int(row["epoch"]) > completed_epoch:
                continue
            history["total"].append(float(row["train_total"]))
            history["plain_mu"].append(float(row["train_plain_mu"]))
            history["weighted_mu"].append(float(row["train_weighted_mu"]))
    return history


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def reduce_component_means(
    component_sums: dict[str, float],
    count: int,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    names = tuple(component_sums)
    packed = torch.tensor(
        [component_sums[name] for name in names] + [float(count)],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    denominator = packed[-1].clamp_min(1.0)
    return {
        name: float((packed[index] / denominator).item())
        for index, name in enumerate(names)
    }


def checkpoint_epoch(epoch: int, checkpoint_cfg: dict | None) -> bool:
    """Whether to keep a permanent checkpoint at this epoch.

    Deliberately separate from the submission schedule: keeping a checkpoint is
    free, submitting is not. Codabench quota is finite and §5.5 records that
    repeated leaderboard probing over four subjects overfits, so the two
    schedules must be able to differ.
    """
    if not checkpoint_cfg or not checkpoint_cfg.get("enabled", True):
        return False
    start = int(checkpoint_cfg["start_epoch"])
    every = int(checkpoint_cfg["every_epochs"])
    return epoch >= start and (epoch - start) % every == 0


def validate_checkpoint_config(config, epochs: int) -> dict | None:
    if config is None:
        return None
    result = dict(config)
    result["enabled"] = bool(result.get("enabled", True))
    start = int(result["start_epoch"])
    every = int(result["every_epochs"])
    if start < 1 or start > epochs:
        raise ValueError(f"checkpoint_schedule.start_epoch must lie in [1, {epochs}]")
    if every < 1:
        raise ValueError("checkpoint_schedule.every_epochs must be positive")
    return result


class SubmissionTrigger:
    """Loss-triggered submission schedule.

    A fixed `start_epoch` submits at the same epoch regardless of what the model
    is doing, which wastes quota on arms that converge at different rates — the
    stage-1 arm crosses a given training loss ~120 epochs before the from-scratch
    control does. This instead waits for the training loss to hold below a
    threshold for N consecutive epochs, then submits every `every_epochs` until
    `max_submissions` is reached.

    The decision is taken from the all-reduced epoch mean, so every rank computes
    the same answer. State is rebuilt from `train_log.csv` on resume rather than
    persisted separately, so there is one source of truth.
    """

    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        self.metric = str(config.get("metric", "plain_mu"))
        self.threshold = float(config.get("threshold", 0.006))
        self.consecutive = int(config.get("consecutive_epochs", 5))
        self.every = int(config.get("every_epochs", 5))
        self.max_submissions = int(config.get("max_submissions", 18))
        # Once the budget is spent there is nothing left to measure, and with a
        # deadline the remaining epochs are pure cost.
        self.stop_after_last = bool(config.get("stop_after_last_submission", True))
        if self.consecutive < 1 or self.every < 1 or self.max_submissions < 1:
            raise ValueError(
                "leaderboard_eval.trigger: consecutive_epochs, every_epochs and "
                "max_submissions must all be >= 1"
            )
        self.streak = 0
        self.trigger_epoch: int | None = None
        self.submitted = 0

    def span(self) -> int:
        """Epochs from the first submission to the last, inclusive."""
        return (self.max_submissions - 1) * self.every

    def update(self, epoch: int, value: float) -> bool:
        """Advance the state by one epoch and say whether to submit."""
        if not self.enabled:
            return False
        self.streak = self.streak + 1 if value < self.threshold else 0
        if self.trigger_epoch is None and self.streak >= self.consecutive:
            self.trigger_epoch = epoch
        if self.trigger_epoch is None:
            return False
        if self.submitted >= self.max_submissions:
            return False
        if (epoch - self.trigger_epoch) % self.every:
            return False
        self.submitted += 1
        return True

    def replay(self, history: dict, first_epoch: int = 1) -> None:
        """Rebuild state from a completed history, without submitting."""
        values = history.get(self.metric, [])
        for offset, value in enumerate(values):
            self.update(first_epoch + offset, float(value))


def scheduled_epoch(epoch: int, leaderboard_cfg: dict) -> bool:
    if not leaderboard_cfg["enabled"]:
        return False
    start = int(leaderboard_cfg["start_epoch"])
    every = int(leaderboard_cfg["every_epochs"])
    return epoch >= start and (epoch - start) % every == 0


def infer_ct_hu(
    subject: dict,
    model: torch.nn.Module,
    device: torch.device,
    patch_size,
    infer_transform,
    input_keys,
) -> tuple[np.ndarray, np.ndarray]:
    data = infer_transform({key: subject[key] for key in input_keys})
    inputs = data["input"].unsqueeze(0).to(device)
    autocast_context = (
        torch.amp.autocast("cuda")
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        prediction = sliding_window_inference(
            inputs,
            patch_size,
            VAL_SW_BATCH,
            model,
            overlap=VAL_OVERLAP,
            mode=VAL_MODE,
            progress=False,
            sw_device=device,
            device="cpu",
        )
    prediction_hu = (
        prediction.float().numpy()[0, 0] * 3000.0 - 1000.0
    ).astype(np.float32, copy=False)
    affine = data["nacpet"].meta["affine"].numpy()
    return prediction_hu, affine


def generate_validation_predictions(
    subjects: list[dict],
    model: torch.nn.Module,
    device: torch.device,
    patch_size,
    infer_transform,
    input_keys,
    prediction_dir: Path,
    rank: int,
) -> None:
    model.eval()
    for subject in subjects:
        name = subject["subject"]
        print(f"[rank {rank}] [leaderboard] predicting {name}", flush=True)
        prediction_hu, affine = infer_ct_hu(
            subject,
            model,
            device,
            patch_size,
            infer_transform,
            input_keys,
        )
        if not np.isfinite(prediction_hu).all():
            raise RuntimeError(f"{name}: prediction contains NaN or Inf")
        subject_dir = prediction_dir / name
        subject_dir.mkdir(parents=True, exist_ok=True)
        final_path = subject_dir / "ct.nii.gz"
        temporary_path = subject_dir / "ct.tmp.nii.gz"
        nib.save(nib.Nifti1Image(prediction_hu, affine), temporary_path)
        temporary_path.replace(final_path)
        print(f"[rank {rank}] [leaderboard] saved {final_path}", flush=True)


def append_submission_job(
    path: Path,
    epoch: int,
    event: str,
    process_id=None,
    return_code=None,
    prediction_dir=None,
    log_path=None,
) -> None:
    new_file = not path.exists() or not path.stat().st_size
    with open(path, "a", newline="") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(
                [
                    "epoch",
                    "event",
                    "pid",
                    "return_code",
                    "prediction_dir",
                    "log_path",
                ]
            )
        writer.writerow(
            [
                epoch,
                event,
                process_id,
                return_code,
                prediction_dir,
                log_path,
            ]
        )


def rebuild_score_ledger(
    prediction_root: Path,
    ledger_path: Path,
) -> None:
    rows = []
    for result_path in sorted(
        prediction_root.glob(
            "epoch_*/codabench_submission_*_result.json"
        )
    ):
        try:
            epoch = int(result_path.parent.name.split("_")[-1])
            with open(result_path, encoding="utf-8") as file:
                result = json.load(file)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            print(
                f"[leaderboard] warning: cannot parse {result_path}: {error}"
            )
            continue
        ct_scores = [
            score.get("score")
            for score in result.get("scores", [])
            if score.get("column_key") == "ct_mu_map_mae"
        ]
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(
                    prediction_root.parent
                    / "checkpoints"
                    / f"epoch_{epoch:04d}.pth"
                ),
                "submission_id": result.get("submission_id"),
                "status": result.get("status"),
                "ct_mu_map_mae": ct_scores[0] if ct_scores else "",
                "fetched_at_utc": result.get("fetched_at_utc"),
                "result_json": str(result_path),
            }
        )
    with open(ledger_path, "w", newline="") as file:
        fieldnames = (
            "epoch",
            "checkpoint",
            "submission_id",
            "status",
            "ct_mu_map_mae",
            "fetched_at_utc",
            "result_json",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["epoch"]))


def launch_submission(
    epoch: int,
    prediction_dir: Path,
    leaderboard_cfg: dict,
    zip_dir: Path,
    log_dir: Path,
    jobs_path: Path,
):
    command = [
        sys.executable,
        str(SUBMIT_WORKER),
        "--prediction-dir",
        str(prediction_dir),
        "--workdir",
        str(zip_dir),
        "--competition",
        str(int(leaderboard_cfg["competition_id"])),
        "--max-attempts",
        str(int(leaderboard_cfg.get("max_submit_attempts", 3))),
        "--retry-delay",
        str(int(leaderboard_cfg.get("retry_delay_seconds", 300))),
    ]
    phase_id = leaderboard_cfg.get("phase_id")
    if phase_id is not None:
        command.extend(["--phase", str(int(phase_id))])
    if bool(leaderboard_cfg.get("dry_run", False)):
        command.append("--dry-run")
    elif not bool(leaderboard_cfg.get("wait_for_score", True)):
        command.append("--no-wait")
    if bool(leaderboard_cfg.get("delete_after_score", False)):
        command.append("--delete-after-score")

    log_path = log_dir / f"epoch_{epoch:04d}_submission.log"
    log_handle = open(log_path, "w")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    append_submission_job(
        jobs_path,
        epoch,
        "launched",
        process.pid,
        prediction_dir=prediction_dir,
        log_path=log_path,
    )
    print(
        f"[leaderboard] epoch {epoch}: submission worker pid={process.pid}; "
        f"log={log_path}",
        flush=True,
    )
    return {
        "epoch": epoch,
        "process": process,
        "prediction_dir": prediction_dir,
        "log_path": log_path,
    }


def poll_submission_jobs(
    jobs: dict[int, dict],
    jobs_path: Path,
    prediction_root: Path,
    score_ledger_path: Path,
    *,
    block: bool = False,
) -> None:
    completed = []
    for epoch, job in list(jobs.items()):
        process = job["process"]
        return_code = process.wait() if block else process.poll()
        if return_code is None:
            continue
        event = "completed" if return_code == 0 else "failed"
        append_submission_job(
            jobs_path,
            epoch,
            event,
            process.pid,
            return_code,
            job["prediction_dir"],
            job["log_path"],
        )
        print(
            f"[leaderboard] epoch {epoch}: submission worker {event} "
            f"(exit={return_code}); log={job['log_path']}",
            flush=True,
        )
        completed.append(epoch)
    for epoch in completed:
        del jobs[epoch]
    rebuild_score_ledger(prediction_root, score_ledger_path)


def validate_leaderboard_config(
    config: dict,
    epochs: int,
) -> dict:
    required = (
        "enabled",
        "start_epoch",
        "every_epochs",
        "competition_id",
        "prediction_dir",
        "zip_dir",
        "log_dir",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"leaderboard_eval missing keys: {missing}")
    result = dict(config)
    result["enabled"] = bool(result["enabled"])
    start = int(result["start_epoch"])
    every = int(result["every_epochs"])
    if start < 1 or start > epochs:
        raise ValueError(
            f"leaderboard_eval.start_epoch must lie in [1, {epochs}]"
        )
    if every < 1:
        raise ValueError("leaderboard_eval.every_epochs must be positive")
    if (
        result["enabled"]
        and not bool(result.get("dry_run", False))
        and not bool(result.get("wait_for_score", True))
    ):
        raise ValueError(
            "wait_for_score must be true to retrieve the requested μ-MAE"
        )
    if bool(result.get("delete_after_score", False)) and (
        bool(result.get("dry_run", False))
        or not bool(result.get("wait_for_score", True))
    ):
        raise ValueError(
            "delete_after_score requires wait_for_score=true and dry_run=false"
        )
    if result["enabled"] and not bool(result.get("dry_run", False)):
        missing_credentials = [
            name
            for name in ("CODABENCH_USERNAME", "CODABENCH_PASSWORD")
            if not os.environ.get(name)
        ]
        if missing_credentials:
            raise RuntimeError(
                "automatic leaderboard evaluation requires environment "
                "variables: " + ", ".join(missing_credentials)
            )
    return result


def save_training_plot(
    path: Path,
    history: dict[str, list[float]],
) -> None:
    plt.figure()
    epochs = range(1, len(history["total"]) + 1)
    for name, values in history.items():
        plt.plot(epochs, values, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("All-75 training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.resume_epoch is not None and args.resume is None:
        raise ValueError("--resume-epoch requires --resume")
    cfg = load_config()
    if cfg.get("experiment") != "finetune_v1_width96":
        raise ValueError("config.yaml must set experiment: finetune_v1_width96")
    epochs = int(cfg["epochs"])
    tissue_cfg = validate_tissue_loss_config(cfg["tissue_loss"])
    leaderboard_cfg = validate_leaderboard_config(
        cfg["leaderboard_eval"], epochs
    )
    checkpoint_cfg = validate_checkpoint_config(
        cfg.get("checkpoint_schedule"), epochs
    )
    input_keys = tuple(cfg.get("input_keys", DEFAULT_INPUT_KEYS))
    infer_transform = get_inference_transforms(input_keys)
    output_dir = resolve_from_base(cfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    training_log = log_dir / "train_log.csv"
    resume_path = (
        resolve_from_base(args.resume) if args.resume is not None else None
    )
    completed_epoch = (
        infer_resume_epoch(resume_path, args.resume_epoch, training_log)
        if resume_path is not None
        else 0
    )
    if not 0 <= completed_epoch < epochs:
        raise ValueError(
            f"resume epoch must lie in [0, {epochs - 1}], "
            f"got {completed_epoch}"
        )
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

    distributed, rank, world_size, local_rank = init_distributed()
    is_main = rank == 0
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    use_cuda = device.type == "cuda"

    train_data = get_dataset(str(resolve_from_base(cfg["data_dir"])))
    validation_data = get_inference_dataset(
        str(resolve_from_base(cfg["validation_data_dir"]))
    )
    expected_train = int(cfg.get("expected_training_subjects", 75))
    expected_validation = int(cfg.get("expected_validation_subjects", 4))
    if len(train_data) != expected_train:
        raise ValueError(
            f"expected {expected_train} training subjects, "
            f"found {len(train_data)}"
        )
    if len(validation_data) != expected_validation:
        raise ValueError(
            f"expected {expected_validation} validation subjects, "
            f"found {len(validation_data)}"
        )
    local_validation_data = validation_data[rank::world_size]

    if is_main:
        schedule = [
            epoch
            for epoch in range(1, epochs + 1)
            if scheduled_epoch(epoch, leaderboard_cfg)
        ]
        print(f"Using device: {device} (world_size={world_size})")
        print(f"Training on all {len(train_data)} labeled subjects")
        print(
            f"External validation subjects: "
            f"{[subject['subject'] for subject in validation_data]}"
        )
        print(f"Input channels ({len(input_keys)}): {list(input_keys)}")
        print(f"Checkpoint/Codabench schedule: {schedule}")

    train_transform = get_transforms(
        cfg["patch_size"],
        cfg["train_num_samples"],
        input_keys,
    )
    train_dataset = CacheDataset(
        data=train_data,
        transform=train_transform,
        cache_rate=float(cfg.get("cache_rate", 0.0)),
        num_workers=int(cfg["num_workers"]),
        copy_cache=False,
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg["num_workers"]),
        pin_memory=use_cuda,
        persistent_workers=int(cfg["num_workers"]) > 0,
    )

    base_channels = int(cfg["base_channels"])
    bare_model = build_model(
        in_channels=len(input_keys), base_channels=base_channels
    ).to(device)
    if is_main:
        print(f"Arm: {cfg.get('arm', 'unspecified')} — width {base_channels}, "
              f"{describe_model(bare_model)}")

    # `init_checkpoint` is an *initialisation*, not a resumption: the weights are
    # loaded but the epoch counter, LR schedule and optimizer all start from
    # scratch, so the pretrained arm sees exactly the schedule the control does.
    # --resume is the separate mechanism for continuing an interrupted run.
    init_checkpoint = cfg.get("init_checkpoint")
    if init_checkpoint and resume_path is None:
        init_path = resolve_from_base(init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {init_path}")
        payload = torch.load(init_path, map_location=device, weights_only=True)
        state_dict = payload.get("model_state_dict", payload)
        bare_model.load_state_dict(state_dict, strict=True)
        if is_main:
            print(
                f"Initialised from {init_path}"
                + (f" (pretrained epoch {payload['pretrained_epoch']}, "
                   f"stem {payload.get('stem_policy', '?')})"
                   if isinstance(payload, dict) and "pretrained_epoch" in payload
                   else "")
            )
    elif init_checkpoint and is_main:
        print(
            f"Ignoring init_checkpoint because --resume was given: "
            f"{resume_path} carries the weights."
        )

    if resume_path is not None:
        checkpoint = torch.load(
            resume_path, map_location=device, weights_only=True
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        bare_model.load_state_dict(state_dict, strict=True)
    model = bare_model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            bare_model,
            device_ids=[local_rank] if use_cuda else None,
        )

    optimizer_lr = float(cfg["learning_rate"])
    scheduler_t_max = epochs
    scheduler_progress = completed_epoch
    continuation_cfg = cfg.get("continuation_schedule")
    using_continuation_schedule = False
    if continuation_cfg is not None:
        continuation_start = int(continuation_cfg["start_epoch"])
        continuation_lr = float(continuation_cfg["learning_rate"])
        if not 0 <= continuation_start < epochs:
            raise ValueError(
                "continuation_schedule.start_epoch must lie in "
                f"[0, {epochs - 1}]"
            )
        if continuation_lr <= 0:
            raise ValueError(
                "continuation_schedule.learning_rate must be positive"
            )
        if completed_epoch >= continuation_start and resume_path is not None:
            using_continuation_schedule = True
            optimizer_lr = continuation_lr
            scheduler_t_max = epochs - continuation_start
            scheduler_progress = completed_epoch - continuation_start

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max
    )
    if completed_epoch:
        resumed_lr = optimizer_lr * (
            1.0
            + math.cos(
                math.pi * scheduler_progress / scheduler_t_max
            )
        ) / 2.0
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = resumed_lr
        scheduler.last_epoch = scheduler_progress
        scheduler._step_count = scheduler_progress + 1
        scheduler._last_lr = [
            parameter_group["lr"]
            for parameter_group in optimizer.param_groups
        ]
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    ct_weight = float(cfg.get("ct_loss_weight", 0.0))
    umap_weight = float(cfg.get("umap_loss_weight", 1.0))
    accumulation_steps = max(1, int(cfg.get("grad_accum_steps", 1)))

    plot_dir = output_dir / "plots"
    prediction_root = output_dir / leaderboard_cfg["prediction_dir"]
    zip_dir = output_dir / leaderboard_cfg["zip_dir"]
    leaderboard_log_dir = output_dir / leaderboard_cfg["log_dir"]
    jobs_path = log_dir / "leaderboard_submission_jobs.csv"
    score_ledger_path = log_dir / "leaderboard_scores.csv"
    if is_main:
        for directory in (
            checkpoint_dir,
            log_dir,
            plot_dir,
            prediction_root,
            zip_dir,
            leaderboard_log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if not training_log.exists() or not training_log.stat().st_size:
            with open(training_log, "w", newline="") as file:
                csv.writer(file).writerow(
                    [
                        "epoch",
                        "train_total",
                        "train_plain_mu",
                        "train_weighted_mu",
                        "learning_rate",
                        "leaderboard_checkpoint",
                    ]
                )
        rebuild_score_ledger(prediction_root, score_ledger_path)
    if distributed:
        dist.barrier()

    def compute_loss_components(prediction, target, mask):
        _, plain_mu, weighted_mu = blended_umap_mae_loss(
            prediction, target, mask, tissue_cfg
        )
        return plain_mu, weighted_mu

    def microbatch_loss_fractions(target, mask):
        """Return weights that reproduce the full crop-batch reductions."""
        plain_denominators = mask.flatten(1).sum(dim=1).float()
        if tissue_cfg["enabled"]:
            weighted_denominators = []
            with torch.no_grad():
                for micro_index in range(target.shape[0]):
                    micro_target = target[micro_index : micro_index + 1]
                    micro_mask = mask[micro_index : micro_index + 1]
                    weights, _ = tissue_weight_map(
                        micro_target, tissue_cfg, micro_mask
                    )
                    weighted_denominators.append(
                        (
                            weights
                            * micro_mask.to(weights.dtype)
                        ).sum()
                    )
            weighted_denominators = torch.stack(weighted_denominators)
        else:
            weighted_denominators = plain_denominators
        return (
            plain_denominators
            / plain_denominators.sum().clamp_min(1.0),
            weighted_denominators
            / weighted_denominators.sum().clamp_min(1.0),
        )

    effective_batch = (
        int(cfg["batch_size"])
        * int(cfg["train_num_samples"])
        * accumulation_steps
        * world_size
    )
    if is_main:
        print(
            f"Effective patch batch: {effective_batch}; "
            f"loss weights CT={ct_weight}, μ={umap_weight}"
        )
        print("Starting all-75 training...")

    trigger = SubmissionTrigger(leaderboard_cfg.get("trigger", {}))
    history = load_training_history(training_log, completed_epoch)
    if trigger.enabled:
        # Resuming must not restart the streak or re-spend the submission budget.
        trigger.replay(history)
        if is_main:
            print(
                f"Submission trigger: {trigger.metric} < {trigger.threshold} for "
                f"{trigger.consecutive} consecutive epochs, then every "
                f"{trigger.every} for {trigger.max_submissions} submissions "
                f"(spans {trigger.span()} epochs)"
            )
            if trigger.trigger_epoch is not None:
                print(f"  already triggered at epoch {trigger.trigger_epoch}; "
                      f"{trigger.submitted} submissions spent")
            print("  start_epoch/every_epochs are ignored while the trigger is on")
    submission_jobs: dict[int, dict] = {}
    if is_main and resume_path is not None:
        print(
            f"Resuming weights from {resume_path} after epoch "
            f"{completed_epoch}; next epoch={completed_epoch + 1}, "
            f"lr={optimizer.param_groups[0]['lr']:.12g}",
            flush=True,
        )
        if using_continuation_schedule:
            print(
                "Using continuation cosine schedule: "
                f"start_epoch={continuation_start}, "
                f"initial_lr={continuation_lr:.12g}, "
                f"end_epoch={epochs}",
                flush=True,
            )
        print(
            "The source checkpoint has no optimizer/scaler state; "
            "AdamW moments and AMP scaling restart from defaults.",
            flush=True,
        )
    for epoch_index in range(completed_epoch, epochs):
        epoch = epoch_index + 1
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_index)
        sums = {"total": 0.0, "plain_mu": 0.0, "weighted_mu": 0.0}
        batch_count = len(train_loader)
        progress = tqdm(train_loader) if is_main else train_loader
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(progress):
            target = batch["ct"].to(device)
            mask = batch["prediction_mask"].bool().to(device)
            target[~mask] = 0
            plain_fractions, weighted_fractions = (
                microbatch_loss_fractions(target, mask)
            )
            microbatch_count = target.shape[0]
            step_now = (
                (batch_index + 1) % accumulation_steps == 0
                or (batch_index + 1) == batch_count
            )
            batch_means = {
                "total": 0.0,
                "plain_mu": 0.0,
                "weighted_mu": 0.0,
            }
            for micro_index in range(microbatch_count):
                inputs = batch["input"][
                    micro_index : micro_index + 1
                ].to(device)
                micro_target = target[micro_index : micro_index + 1]
                micro_mask = mask[micro_index : micro_index + 1]
                sync_now = (
                    step_now and micro_index + 1 == microbatch_count
                )
                sync_context = (
                    model.no_sync()
                    if distributed and not sync_now
                    else nullcontext()
                )
                autocast_context = (
                    torch.amp.autocast("cuda")
                    if use_cuda
                    else nullcontext()
                )
                with sync_context:
                    with autocast_context:
                        prediction = model(inputs)
                        plain_mu, weighted_mu = compute_loss_components(
                            prediction, micro_target, micro_mask
                        )
                        plain_contribution = (
                            plain_mu * plain_fractions[micro_index]
                        )
                        weighted_contribution = (
                            weighted_mu
                            * weighted_fractions[micro_index]
                        )
                        if tissue_cfg["enabled"]:
                            blend = tissue_cfg["blend"]
                            mu_contribution = (
                                (1.0 - blend) * plain_contribution
                                + blend * weighted_contribution
                            )
                        else:
                            mu_contribution = plain_contribution
                        total = umap_weight * mu_contribution
                        if ct_weight:
                            valid = micro_mask.to(plain_mu.dtype)
                            ct_l1 = (
                                (
                                    (prediction - micro_target).abs()
                                    * valid
                                ).sum()
                                / valid.sum().clamp_min(1.0)
                            )
                            total = total + (
                                ct_weight
                                * ct_l1
                                * plain_fractions[micro_index]
                            )
                    scaler.scale(total / accumulation_steps).backward()
                batch_means["total"] += float(total.detach().item())
                batch_means["plain_mu"] += float(
                    plain_contribution.detach().item()
                )
                batch_means["weighted_mu"] += float(
                    weighted_contribution.detach().item()
                )
                del inputs, prediction, total, plain_mu, weighted_mu

            if step_now:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for name, value in batch_means.items():
                sums[name] += value
            if is_main:
                progress.set_description(
                    f"loss {batch_means['total']:.5f} "
                    f"μ {batch_means['plain_mu']:.5f}"
                )

        means = reduce_component_means(
            sums, batch_count, device, distributed
        )
        scheduler.step()
        if trigger.enabled:
            # Decided from the all-reduced epoch mean, so all ranks agree.
            was_armed = trigger.trigger_epoch is not None
            is_leaderboard_epoch = trigger.update(epoch, means[trigger.metric])
            if is_main and not was_armed and trigger.trigger_epoch is not None:
                last = trigger.trigger_epoch + trigger.span()
                print(
                    f"[trigger] {trigger.metric} held below {trigger.threshold} "
                    f"for {trigger.consecutive} epochs; submitting at "
                    f"{trigger.trigger_epoch}, {trigger.trigger_epoch + trigger.every}, "
                    f"... {last} ({trigger.max_submissions} total)",
                    flush=True,
                )
                if last > epochs:
                    n_fit = (epochs - trigger.trigger_epoch) // trigger.every + 1
                    print(
                        f"[trigger] WARNING: the schedule needs epoch {last} but "
                        f"epochs={epochs}; only {n_fit} of "
                        f"{trigger.max_submissions} submissions will happen. "
                        f"Raise epochs to {last} to get them all.",
                        flush=True,
                    )
        else:
            is_leaderboard_epoch = scheduled_epoch(epoch, leaderboard_cfg)
        keep_checkpoint = is_leaderboard_epoch or checkpoint_epoch(
            epoch, checkpoint_cfg
        )

        if is_main:
            print(
                f"Epoch {epoch:03d}: total={means['total']:.6f} "
                f"plain-μ={means['plain_mu']:.6f} "
                f"weighted-μ={means['weighted_mu']:.6f}"
            )
            for name in history:
                history[name].append(means[name])
            state_dict = bare_model.state_dict()
            torch.save(state_dict, checkpoint_dir / "last_model.pth")
            if keep_checkpoint:
                checkpoint_path = (
                    checkpoint_dir / f"epoch_{epoch:04d}.pth"
                )
                torch.save(state_dict, checkpoint_path)
                print(
                    f"[{'leaderboard' if is_leaderboard_epoch else 'checkpoint'}]"
                    f" saved fixed checkpoint: {checkpoint_path}"
                )
            with open(
                log_dir / "train_log.csv", "a", newline=""
            ) as file:
                csv.writer(file).writerow(
                    [
                        epoch,
                        means["total"],
                        means["plain_mu"],
                        means["weighted_mu"],
                        optimizer.param_groups[0]["lr"],
                        int(is_leaderboard_epoch),
                    ]
                )
            save_training_plot(plot_dir / "loss_curve.png", history)
            poll_submission_jobs(
                submission_jobs,
                jobs_path,
                prediction_root,
                score_ledger_path,
            )

        if is_leaderboard_epoch:
            epoch_prediction_dir = (
                prediction_root / f"epoch_{epoch:04d}"
            )
            if is_main:
                epoch_prediction_dir.mkdir(
                    parents=True, exist_ok=True
                )
            if distributed:
                dist.barrier()
            generate_validation_predictions(
                local_validation_data,
                bare_model,
                device,
                cfg["patch_size"],
                infer_transform,
                input_keys,
                epoch_prediction_dir,
                rank,
            )
            if distributed:
                dist.barrier()
            if is_main:
                missing_outputs = [
                    subject["subject"]
                    for subject in validation_data
                    if not (
                        epoch_prediction_dir
                        / subject["subject"]
                        / "ct.nii.gz"
                    ).is_file()
                ]
                if missing_outputs:
                    raise RuntimeError(
                        "validation prediction generation incomplete: "
                        f"{missing_outputs}"
                    )
                submission_jobs[epoch] = launch_submission(
                    epoch,
                    epoch_prediction_dir,
                    leaderboard_cfg,
                    zip_dir,
                    leaderboard_log_dir,
                    jobs_path,
                )

        if (
            trigger.enabled
            and trigger.stop_after_last
            and trigger.trigger_epoch is not None
            and trigger.submitted >= trigger.max_submissions
        ):
            # Every rank sees the same trigger state (it is computed from the
            # all-reduced epoch mean), so they leave the loop together. Any
            # in-flight submissions are drained by the blocking poll below.
            if is_main:
                print(
                    f"[trigger] {trigger.submitted} submissions spent at epoch "
                    f"{epoch}; stopping early (epochs={epochs}).",
                    flush=True,
                )
            break

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        if submission_jobs:
            print(
                "[leaderboard] training complete; waiting for outstanding "
                "submission workers",
                flush=True,
            )
        poll_submission_jobs(
            submission_jobs,
            jobs_path,
            prediction_root,
            score_ledger_path,
            block=True,
        )
        print(f"[leaderboard] score ledger: {score_ledger_path}")


if __name__ == "__main__":
    main()
