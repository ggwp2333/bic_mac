"""
Transforms for stage-2 pretraining: the model's real 4-channel input space.

**Every normalisation here is copied from `finetune_v1/transforms.py` and must
stay copied.** The entire value of stage 2 over stage 1 is that the stem —
`enc1.block.conv1` — transfers, and a stem only transfers if the input
statistics it saw during pretraining are the ones it will see downstream. If the
two normalisations drift apart, this module is worse than useless: it trains a
stem on a distribution that does not exist at fine-tune time.

Two things happen here that the downstream recipe does not do:

**Missing channels are filled with zeros, after normalisation.** QUADRA_HC has
PET and nothing else. Filling before normalising would ask
`NormalizeIntensity(nonzero=True)` to take statistics over an empty set. After
normalisation, a zero channel sits at the mean of the z-normalised MRI channels,
which is the right neutral value.

**Modality dropout.** Whole channels are zeroed at random. This is the load-
bearing idea of stage 2, not a detail:

  * it makes QUADRA's `[PET, --, --, --]` an ordinary sample rather than a
    special case, so the model never learns "PET-only means QUADRA means
    healthy-young-Viennese-anatomy";
  * without it, a model pretrained on a corpus where 96 of 171 scans have zero
    MRI learns to route around MRI, and the fine-tune has to unlearn that;
  * it is the §3.7 augmentation lever aimed straight at the +14–18%
    generalization gap §3.12 measured on the stage-1 arm.

Block masking survives from stage 1 but has changed role: there it *was* the
pretext task, here the task is real prediction and masking is cutout-style
augmentation. Hence the much lower default ratio.

Never left-right flip: whole-body anatomy is laterally asymmetric.
"""

from __future__ import annotations

import numpy as np
from monai.transforms import (
    Compose,
    ConcatItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandomizableTransform,
    ScaleIntensityRanged,
)

# Must equal finetune_v1/config.yaml `input_keys`, in order.
CHANNELS = ("nacpet", "mri_in_phase", "mri_out_phase", "topogram")
MAGNITUDE_KEYS = ("mri_in_phase", "mri_out_phase", "topogram")

CT_A_MIN, CT_A_MAX = -1000.0, 2000.0


class BoneCalibrationd(MapTransform):
    """QUADRA's 140 kVp CT onto the challenge's Carney-120 kVp bone scale.

    Identity below the pivot and for any case whose manifest gain is 1.0.
    See pretrain_v1/prepare_corpus.py for the fit.
    """

    def __init__(self, key="ct", gain_key="bone_gain", pivot_hu=300.0):
        super().__init__([key])
        self.key, self.gain_key, self.pivot = key, gain_key, float(pivot_hu)

    def __call__(self, data):
        data = dict(data)
        gain = float(data.get(self.gain_key, 1.0))
        if gain == 1.0:
            return data
        ct = data[self.key]
        data[self.key] = ct + (ct > self.pivot) * ((ct - self.pivot) * (gain - 1.0))
        return data


class BroadcastTopogramd(MapTransform):
    """(C, X, 1, Z) -> (C, X, Y, Z). Copied from finetune_v1/transforms.py."""

    def __init__(self, key="topogram", reference_key="ct"):
        super().__init__([key])
        self.key, self.reference_key = key, reference_key

    def __call__(self, data):
        data = dict(data)
        if self.key not in data:
            return data
        topo, ref = data[self.key], data[self.reference_key]
        if topo.ndim != 4 or ref.ndim != 4:
            raise ValueError(
                f"expected channel-first 3-D inputs, got topogram={topo.shape}, "
                f"{self.reference_key}={ref.shape}"
            )
        if topo.shape[1] != ref.shape[1] or topo.shape[3] != ref.shape[3] \
                or topo.shape[2] != 1:
            raise ValueError(
                f"topogram {topo.shape} cannot be broadcast onto {ref.shape}"
            )
        if hasattr(topo, "repeat_interleave"):
            data[self.key] = topo.repeat_interleave(ref.shape[2], dim=2)
        else:
            data[self.key] = np.repeat(topo, ref.shape[2], axis=2)
        return data


class FillMissingChannelsd(MapTransform):
    """Insert an all-zero volume for every channel this case does not have.

    Runs *after* normalisation so the zeros land at the normalised mean rather
    than being fed to a statistic that has no samples to compute from.
    """

    def __init__(self, keys=CHANNELS, reference_key="ct"):
        super().__init__(list(keys))
        self.channels = tuple(keys)
        self.reference_key = reference_key

    def __call__(self, data):
        data = dict(data)
        ref = data[self.reference_key]
        for key in self.channels:
            if key not in data or data[key] is None:
                # Clone the reference rather than building a bare array:
                # ConcatItemsd refuses to mix MetaTensor with ndarray, and the
                # loaded channels are MetaTensors.
                if hasattr(ref, "clone"):
                    zeros = ref.clone()
                    zeros[:] = 0
                else:
                    zeros = np.zeros_like(np.asarray(ref, dtype=np.float32))
                data[key] = zeros
        return data


class RandModalityDropoutd(RandomizableTransform):
    """Zero whole input channels at random, never all of them.

    Only channels this case actually has can be dropped — dropping an already
    empty channel would be a no-op that silently weakens the augmentation rate.
    At least one available channel always survives, so no sample is ever an
    all-zero input paired with a CT target.
    """

    def __init__(self, input_key="input", prob=0.5, per_channel_prob=0.25,
                 channels=CHANNELS):
        RandomizableTransform.__init__(self, prob)
        self.input_key = input_key
        self.per_channel_prob = float(per_channel_prob)
        self.channels = tuple(channels)

    def __call__(self, data):
        data = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return data
        available = [self.channels.index(c)
                     for c in data.get("available_channels", self.channels)
                     if c in self.channels]
        if len(available) <= 1:
            return data
        drop = [i for i in available if self.R.random() < self.per_channel_prob]
        if len(drop) >= len(available):          # keep at least one
            drop = drop[: len(available) - 1]
        if not drop:
            return data
        x = np.array(data[self.input_key], copy=True)
        for i in drop:
            x[i] = 0.0
        data[self.input_key] = x
        return data


class RandBlockMaskd(RandomizableTransform):
    """Cutout-style block masking of the input, shared across channels.

    Shared across channels on purpose: masking each channel independently would
    let the network inpaint a hole in one modality from the same location in
    another, which teaches co-registration rather than robustness.
    """

    def __init__(self, input_key="input", mask_ratio=0.25, block_size=16, prob=1.0):
        RandomizableTransform.__init__(self, prob)
        self.input_key = input_key
        self.mask_ratio = float(mask_ratio)
        self.block_size = int(block_size)
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie in [0, 1)")

    def __call__(self, data):
        data = dict(data)
        self.randomize(None)
        if not self._do_transform or self.mask_ratio == 0.0:
            return data
        x = np.array(data[self.input_key], copy=True)
        spatial = x.shape[1:]
        grid = tuple(max(1, int(np.ceil(s / self.block_size))) for s in spatial)
        coarse = self.R.random(grid) < self.mask_ratio
        blocks = np.kron(coarse, np.ones((self.block_size,) * 3, dtype=bool))
        blocks = blocks[: spatial[0], : spatial[1], : spatial[2]]
        x[:, blocks] = 0.0
        data[self.input_key] = x
        return data


def _load_pipeline(case_channels: tuple[str, ...]):
    """Stages up to and including concatenation, for the channels a case has."""
    present = tuple(c for c in CHANNELS if c in case_channels)
    keys = [*present, "ct", "body_mask", "loss_mask"]
    steps = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        BoneCalibrationd(),
    ]
    if "topogram" in present:
        steps.append(BroadcastTopogramd())
    # --- normalisation: identical to finetune_v1/transforms.py ---
    if "nacpet" in present:
        steps.append(NormalizeIntensityd(
            keys=["nacpet"], nonzero=False, channel_wise=True, subtrahend=[0],
        ))
    mag = [k for k in MAGNITUDE_KEYS if k in present]
    if mag:
        steps.append(NormalizeIntensityd(
            keys=mag, nonzero=True, channel_wise=True,
        ))
    steps += [
        ScaleIntensityRanged(keys=["ct"], a_min=CT_A_MIN, a_max=CT_A_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        FillMissingChannelsd(),
        ConcatItemsd(keys=list(CHANNELS), name="input"),
        DeleteItemsd(keys=list(CHANNELS)),
    ]
    return steps


class ChannelAwareTransform:
    """Dispatches to a pipeline built for whichever channels the case has.

    MONAI's `Compose` is fixed at construction time, but the corpus is not
    homogeneous — BIC-MAC brings four channels and QUADRA_HC one. Rather than
    loading files that do not exist, the load/normalise stages are built per
    channel set and cached; everything after concatenation is shared.
    """

    def __init__(self, patch_size, num_samples, mask_ratio, block_size,
                 dropout_prob, per_channel_prob, train=True):
        self._cache: dict = {}
        tail = [
            RandCropByPosNegLabeld(
                keys=["input", "ct", "body_mask", "loss_mask"],
                label_key="body_mask",
                spatial_size=patch_size,
                pos=1.0, neg=0.0,
                num_samples=num_samples,
                allow_smaller=True,
            ),
        ]
        if train:
            tail += [
                RandModalityDropoutd(prob=dropout_prob,
                                     per_channel_prob=per_channel_prob),
                RandBlockMaskd(mask_ratio=mask_ratio, block_size=block_size),
            ]
        tail.append(EnsureTyped(keys=["input", "ct", "body_mask", "loss_mask"]))
        self.tail = Compose(tail)

    def _head(self, channels: tuple[str, ...]):
        if channels not in self._cache:
            self._cache[channels] = Compose(_load_pipeline(channels))
        return self._cache[channels]

    def set_random_state(self, seed=None, state=None):
        self.tail.set_random_state(seed=seed, state=state)
        return self

    # Only these reach the DataLoader. The manifest carries a `channels` dict
    # whose values are None for QUADRA's absent modalities, and the default
    # collator raises on None; the rest is bookkeeping the trainer never reads.
    KEEP = ("input", "ct", "body_mask", "loss_mask")

    def __call__(self, case: dict):
        channels = tuple(case["available_channels"])
        data = dict(case)
        # only hand MONAI the keys that exist on disk
        for key in CHANNELS:
            path = case.get("channels", {}).get(key)
            if key in channels and path:
                data[key] = path
            else:
                data.pop(key, None)
        out = self.tail(self._head(channels)(data))
        samples = out if isinstance(out, list) else [out]
        return [{k: s[k] for k in self.KEEP if k in s} for s in samples]


def get_transforms(patch_size, num_samples=2, mask_ratio=0.25, block_size=16,
                   dropout_prob=0.5, per_channel_prob=0.25):
    return ChannelAwareTransform(patch_size, num_samples, mask_ratio,
                                 block_size, dropout_prob, per_channel_prob,
                                 train=True)


def get_validation_transforms(patch_size, num_samples=2):
    """No dropout, no masking: the held-out number should measure the model,
    not the augmentation draw."""
    return ChannelAwareTransform(patch_size, num_samples, 0.0, 16, 0.0, 0.0,
                                 train=False)
