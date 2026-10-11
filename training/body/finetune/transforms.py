import numpy as np
from monai.transforms import *


DEFAULT_INPUT_KEYS = (
    "nacpet",
    "mri_combined_in_phase",
    "mri_combined_out_phase",
    "topogram",
)

MRI_INPUT_KEYS = (
    "mri_combined_in_phase",
    "mri_combined_out_phase",
)


class BroadcastTopogramd(MapTransform):
    """Broadcast a topogram from (C, X, 1, Z) onto a 3D reference grid."""

    def __init__(self, key="topogram", reference_key="nacpet"):
        super().__init__([key])
        self.key = key
        self.reference_key = reference_key

    def __call__(self, data):
        data = dict(data)
        topogram = data[self.key]
        reference = data[self.reference_key]
        if topogram.ndim != 4 or reference.ndim != 4:
            raise ValueError(
                f"expected channel-first 3D inputs, got topogram={topogram.shape}, "
                f"{self.reference_key}={reference.shape}"
            )
        if (
            topogram.shape[1] != reference.shape[1]
            or topogram.shape[3] != reference.shape[3]
            or topogram.shape[2] != 1
        ):
            raise ValueError(
                f"topogram {topogram.shape} cannot be broadcast onto "
                f"{self.reference_key} grid {reference.shape}"
            )
        if hasattr(topogram, "repeat_interleave"):
            data[self.key] = topogram.repeat_interleave(reference.shape[2], dim=2)
        else:
            data[self.key] = np.repeat(topogram, reference.shape[2], axis=2)
        return data


def get_inference_transforms(input_keys=DEFAULT_INPUT_KEYS):
    input_keys = tuple(input_keys)
    transforms = [
        LoadImaged(keys=input_keys),
        EnsureChannelFirstd(keys=input_keys),
    ]
    if "topogram" in input_keys:
        if "nacpet" not in input_keys:
            raise ValueError("topogram input requires nacpet as its 3D reference")
        transforms.append(BroadcastTopogramd())
    # Match the reference multimodal run: PET is scaled around zero using the
    # whole volume, while magnitude MRI is z-normalized over nonzero voxels.
    if "nacpet" in input_keys:
        transforms.append(NormalizeIntensityd(
            keys=["nacpet"],
            nonzero=False,
            channel_wise=True,
            subtrahend=[0],
        ))
    magnitude_keys = [key for key in (*MRI_INPUT_KEYS, "topogram") if key in input_keys]
    if magnitude_keys:
        transforms.append(NormalizeIntensityd(
            keys=magnitude_keys,
            nonzero=True,
            channel_wise=True,
        ))
    transforms.extend([
        ConcatItemsd(keys=input_keys, name="input"),
        EnsureTyped(keys=["input"]),
    ])
    return Compose(transforms)


def get_transforms(patch_size, num_samples=2, input_keys=DEFAULT_INPUT_KEYS):

    input_keys = tuple(input_keys)

    transforms = Compose(
        [

            LoadImaged(keys=[*input_keys, "ct", "prediction_mask"]),

            EnsureChannelFirstd(keys=[*input_keys, "ct", "prediction_mask"]),

            *([BroadcastTopogramd()] if "topogram" in input_keys else []),

            *([NormalizeIntensityd(
                keys=["nacpet"], nonzero=False, channel_wise=True,
                subtrahend=[0],
            )] if "nacpet" in input_keys else []),

            *([NormalizeIntensityd(
                keys=[key for key in (*MRI_INPUT_KEYS, "topogram") if key in input_keys],
                nonzero=True,
                channel_wise=True,
            )] if any(key in input_keys for key in (*MRI_INPUT_KEYS, "topogram")) else []),

            ScaleIntensityRanged(
                keys=["ct"],
                a_min=-1000,
                a_max=2000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            ConcatItemsd(
                keys=input_keys,
                name="input"
            ),

            # RandCrop only crops the concatenated input. Keeping the original
            # full-resolution modalities here would copy/collate several GB of
            # unused data for every sampled patch.
            DeleteItemsd(keys=input_keys),

            # Match the reference run's uniform fixed-size patch sampling.
            RandSpatialCropSamplesd(
                keys=["input", "ct", "prediction_mask"],
                roi_size=patch_size,
                random_size=False,
                num_samples=num_samples,
            ),
            EnsureTyped(keys=["input", "ct", "prediction_mask"]),

        ]
    )

    return transforms
