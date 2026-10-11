"""
Transforms for CT-only self-supervised pretraining.

The pretext task is masked reconstruction: blank out cuboid blocks of a CT
patch, optionally add noise to what survives, and ask the network to put the
CT back. Everything else is kept deliberately identical to the downstream
recipe so the learned weights are useful there:

  * the same `ScaleIntensityRange(-1000, 2000 -> 0, 1)` normalisation, so the
    network's output scale and the Carney mu conversion in `losses.py` carry
    over unchanged;
  * the same fixed-size random patch sampling;
  * **no spatial augmentation.** The body path has none (plan.md §3.7) and
    adding it here would confound the pretraining question with the
    augmentation question, which is a separate live item. Left-right flip is
    excluded on principle -- whole-body anatomy is laterally asymmetric.

The one transform with no downstream counterpart is the bone calibration, which
is a per-case property read from the corpus manifest.
"""

from __future__ import annotations

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandCropByPosNegLabeld,
    RandomizableTransform,
    ScaleIntensityRanged,
)

CT_A_MIN = -1000.0
CT_A_MAX = 2000.0
# normalized value of -1000 HU, i.e. what "blanked to air" means after scaling
AIR_NORM = 0.0


class BoneCalibrationd(MapTransform):
    """Rescale bone HU by a per-case gain, pivoting at `pivot_hu`.

    QUADRA_HC's CT is acquired at a higher tube voltage than the challenge's,
    which compresses cortical bone by ~57 HU on average. Left uncorrected, the
    pretrained model would learn that compressed bone distribution and carry it
    into a fine-tune whose metric is dominated by exactly that class
    (plan.md §5.3: air and cortical bone each run 91% |bias|/MAE).

    Identity below the pivot, where the two corpora already agree, and identity
    for any case whose manifest gain is 1.0.
    """

    def __init__(self, key="ct", gain_key="bone_gain", pivot_hu=300.0):
        super().__init__([key])
        self.key = key
        self.gain_key = gain_key
        self.pivot = float(pivot_hu)

    def __call__(self, data):
        data = dict(data)
        gain = float(data.get(self.gain_key, 1.0))
        if gain == 1.0:
            return data
        ct = data[self.key]
        above = ct > self.pivot
        data[self.key] = ct + above * ((ct - self.pivot) * (gain - 1.0))
        return data


class RandBlockMaskd(RandomizableTransform, MapTransform):
    """Blank random cuboid blocks of the CT patch; emit input + corruption mask.

    Produces:
      ``input``        the corrupted CT, channel-first, ready for the network
      ``corrupt_mask`` True where the voxel was blanked

    Block masking rather than per-voxel masking is the point: a voxel-wise mask
    is solved by local interpolation and teaches nothing, whereas a block wide
    enough to swallow a structure forces the network to infer it from context.
    `block_size` should be read against the measured effective receptive field
    of r90 = 49 voxels (plan.md §5.3) -- blocks far beyond that are not
    recoverable and only add noise to the gradient.

    Noise is applied to the *surviving* voxels only, so the task stays
    "reconstruct what was removed" rather than "denoise everything".
    """

    def __init__(self, keys=("ct",), mask_ratio=0.5, block_size=16,
                 noise_std=0.0, fill_value=AIR_NORM, prob=1.0):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob)
        self.mask_ratio = float(mask_ratio)
        self.block_size = int(block_size)
        self.noise_std = float(noise_std)
        self.fill_value = float(fill_value)
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie in (0, 1)")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")

    def __call__(self, data):
        data = dict(data)
        ct = data["ct"]
        array = np.asarray(ct, dtype=np.float32)          # (C, X, Y, Z)
        spatial = array.shape[1:]

        grid = tuple(max(1, int(np.ceil(s / self.block_size))) for s in spatial)
        coarse = self.R.random(grid) < self.mask_ratio
        blocks = np.kron(coarse, np.ones((self.block_size,) * 3, dtype=bool))
        blocks = blocks[: spatial[0], : spatial[1], : spatial[2]]
        corrupt = np.broadcast_to(blocks[None], array.shape)

        corrupted = np.where(corrupt, np.float32(self.fill_value), array)
        if self.noise_std > 0:
            noise = self.R.normal(0.0, self.noise_std, array.shape).astype(np.float32)
            corrupted = np.where(corrupt, corrupted, corrupted + noise)

        data["input"] = corrupted
        data["corrupt_mask"] = corrupt.astype(np.uint8)
        return data


def _base(manifest_keys=("ct", "body_mask")):
    return [
        LoadImaged(keys=list(manifest_keys)),
        EnsureChannelFirstd(keys=list(manifest_keys)),
        BoneCalibrationd(),
        ScaleIntensityRanged(
            keys=["ct"], a_min=CT_A_MIN, a_max=CT_A_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
    ]


def get_transforms(patch_size, num_samples=2, mask_ratio=0.5, block_size=16,
                   noise_std=0.0):
    """Training pipeline: load, calibrate, normalise, crop, corrupt."""
    return Compose([
        *_base(),
        # Crops are centred on a body voxel, not sampled uniformly over the FOV.
        # The body is only ~10% of a 512x512x531 volume, so uniform sampling --
        # what the downstream recipe uses, where the whole-FOV context is part
        # of the task -- draws mostly air here, and a patch with no body has no
        # scored voxels at all and contributes nothing to the pretext gradient.
        RandCropByPosNegLabeld(
            keys=["ct", "body_mask"],
            label_key="body_mask",
            spatial_size=patch_size,
            pos=1.0,
            neg=0.0,
            num_samples=num_samples,
            allow_smaller=True,
        ),
        RandBlockMaskd(
            mask_ratio=mask_ratio,
            block_size=block_size,
            noise_std=noise_std,
        ),
        EnsureTyped(keys=["input", "ct", "body_mask", "corrupt_mask"]),
    ])


def get_validation_transforms(patch_size, num_samples=2, mask_ratio=0.5,
                              block_size=16, noise_std=0.0):
    """Same pipeline; the trainer re-seeds it every epoch so the held-out
    reconstruction loss is comparable across epochs rather than drifting with
    whatever crop and mask the RNG happened to draw."""
    return get_transforms(
        patch_size, num_samples, mask_ratio, block_size, noise_std
    )
