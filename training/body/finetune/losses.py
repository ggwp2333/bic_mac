"""
Losses for pseudo-CT training.

The challenge is scored on the linear attenuation coefficient (mu, cm^-1)
derived from the predicted CT, not on the CT (HU) itself. This module provides
a differentiable port of the Carney bilinear HU -> mu model used by the
evaluation metric (see ../evaluation/metrics/ct_whole_body_mae.py::hu_to_mu),
so we can put an MAE loss directly on the generated umap.
"""

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

# --- Carney et al. 2006 bilinear HU -> mu @ 511 keV, 120 kVp -----------------
# These MUST match evaluation/metrics/ct_whole_body_mae.py::hu_to_mu
_A, _B, _BP = 5.10e-5, 4.71e-2, 1047.0   # high branch: a * hu1000 + b, breakpoint bp
_LOW_SLOPE = 9.6e-5                        # low branch:  9.6e-5 * hu1000

# The transforms normalize CT with ScaleIntensityRange(a_min=-1000, a_max=2000,
# b_min=0, b_max=1), i.e. norm = (HU + 1000) / 3000  =>  hu1000 = norm * 3000.
_HU1000_RANGE = 3000.0
_HU_OFFSET = -1000.0


def validate_tissue_loss_config(config: Mapping) -> dict:
    """Validate and return a normalized tissue-loss configuration."""
    required = (
        "enabled",
        "blend",
        "lung_air_max_hu",
        "bone_min_hu",
        "lung_air_weight",
        "soft_tissue_weight",
        "bone_weight",
        "interface_multiplier",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"tissue_loss is missing required keys: {missing}")

    result = {
        "enabled": bool(config["enabled"]),
        "blend": float(config["blend"]),
        "lung_air_max_hu": float(config["lung_air_max_hu"]),
        "bone_min_hu": float(config["bone_min_hu"]),
        "lung_air_weight": float(config["lung_air_weight"]),
        "soft_tissue_weight": float(config["soft_tissue_weight"]),
        "bone_weight": float(config["bone_weight"]),
        "interface_multiplier": float(config["interface_multiplier"]),
    }

    if (
        not math.isfinite(result["lung_air_max_hu"])
        or not math.isfinite(result["bone_min_hu"])
        or result["lung_air_max_hu"] >= result["bone_min_hu"]
    ):
        raise ValueError(
            "tissue_loss thresholds must satisfy "
            "lung_air_max_hu < bone_min_hu"
        )
    if not math.isfinite(result["blend"]) or not 0.0 <= result["blend"] <= 1.0:
        raise ValueError("tissue_loss.blend must be in [0, 1]")

    weight_keys = (
        "lung_air_weight",
        "soft_tissue_weight",
        "bone_weight",
        "interface_multiplier",
    )
    non_positive = [
        key
        for key in weight_keys
        if not math.isfinite(result[key]) or result[key] <= 0.0
    ]
    if non_positive:
        raise ValueError(
            f"tissue_loss weights must be positive; invalid: {non_positive}"
        )
    return result


def norm_ct_to_mu(norm_ct: torch.Tensor) -> torch.Tensor:
    """Differentiable Carney HU->mu applied to *normalized* CT in [0, 1].

    Mirrors hu_to_mu() in the evaluation metric exactly (including the
    non-negativity clip), but operates on the network's normalized output/target
    so no de-normalization round-trip is needed.
    """
    # Compute in fp32: mu ~ 0.1 and its error ~1e-4 loses precision in fp16.
    hu1000 = norm_ct.float() * _HU1000_RANGE
    mu = torch.where(hu1000 < _BP, _LOW_SLOPE * hu1000, _A * hu1000 + _B)
    return mu.clamp(min=0.0)


def norm_ct_to_hu(norm_ct: torch.Tensor) -> torch.Tensor:
    """Convert normalized CT back to HU in fp32."""
    return norm_ct.float() * _HU1000_RANGE + _HU_OFFSET


def tissue_class_map(
    target_norm: torch.Tensor,
    lung_air_max_hu: float,
    bone_min_hu: float,
) -> torch.Tensor:
    """Return integer HU classes: 0=lung/air, 1=soft tissue, 2=bone."""
    target_hu = norm_ct_to_hu(target_norm)
    classes = torch.ones_like(target_hu, dtype=torch.long)
    classes[target_hu < lung_air_max_hu] = 0
    classes[target_hu >= bone_min_hu] = 2
    return classes


def tissue_interface_mask(
    classes: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Find the one-voxel 3-D shell around valid tissue-class interfaces."""
    if classes.ndim != 5:
        raise ValueError(
            "tissue interface detection expects a 5-D [B,C,D,H,W] tensor"
        )
    classes_float = classes.float()
    if mask is None:
        max_values = min_values = classes_float
        valid = None
    else:
        valid = mask.bool()
        # Invalid voxels must not change an adjacent valid voxel's tissue
        # interface status. Class IDs are in [0, 2], so -1 and 3 are neutral
        # sentinels for max and min pooling, respectively.
        max_values = torch.where(valid, classes_float, -torch.ones_like(classes_float))
        min_values = torch.where(valid, classes_float, 3 * torch.ones_like(classes_float))

    padded_max = F.pad(max_values, (1, 1, 1, 1, 1, 1), mode="replicate")
    padded_min = F.pad(min_values, (1, 1, 1, 1, 1, 1), mode="replicate")
    local_max = F.max_pool3d(padded_max, kernel_size=3, stride=1)
    local_min = -F.max_pool3d(-padded_min, kernel_size=3, stride=1)
    interfaces = local_max.ne(local_min)
    return interfaces if valid is None else interfaces & valid


def tissue_weight_map(
    target_norm: torch.Tensor,
    tissue_config: Mapping,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build detached per-voxel tissue weights and the interface shell."""
    config = validate_tissue_loss_config(tissue_config)
    classes = tissue_class_map(
        target_norm.detach(),
        config["lung_air_max_hu"],
        config["bone_min_hu"],
    )

    weights = torch.full_like(
        target_norm, config["soft_tissue_weight"], dtype=torch.float32
    )
    weights.masked_fill_(classes == 0, config["lung_air_weight"])
    weights.masked_fill_(classes == 2, config["bone_weight"])

    interfaces = tissue_interface_mask(classes, mask)
    weights = weights * torch.where(
        interfaces,
        weights.new_tensor(config["interface_multiplier"]),
        weights.new_tensor(1.0),
    )
    return weights, interfaces


def umap_mae_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Patch-wise MAE between predicted and target umaps.

    Parameters
    ----------
    pred_norm, target_norm : normalized CT tensors (network output / target).
    mask : optional boolean tensor; when given, the MAE is averaged only over
           True voxels (the prediction region), ignoring bed/background.
    """
    pred_mu = norm_ct_to_mu(pred_norm)
    target_mu = norm_ct_to_mu(target_norm)
    ae = (pred_mu - target_mu).abs()

    if mask is None:
        return ae.mean()

    mask = mask.to(ae.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (ae * mask).sum() / denom


def tissue_weighted_umap_mae_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    tissue_config: Mapping,
) -> torch.Tensor:
    """HU-tissue-weighted patch μ-MAE, normalized by valid voxel weights."""
    pred_mu = norm_ct_to_mu(pred_norm)
    target_mu = norm_ct_to_mu(target_norm)
    ae = (pred_mu - target_mu).abs()

    weights, _ = tissue_weight_map(target_norm, tissue_config, mask)
    valid_weights = weights * mask.to(weights.dtype)
    denom = valid_weights.sum().clamp(min=1.0)
    return (ae * valid_weights).sum() / denom


def blended_umap_mae_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    tissue_config: Mapping,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (blended loss, plain μ-MAE, tissue-weighted μ-MAE)."""
    config = validate_tissue_loss_config(tissue_config)
    plain = umap_mae_loss(pred_norm, target_norm, mask)
    if not config["enabled"]:
        return plain, plain, plain

    weighted = tissue_weighted_umap_mae_loss(
        pred_norm, target_norm, mask, config
    )
    blend = config["blend"]
    total = (1.0 - blend) * plain + blend * weighted
    return total, plain, weighted
