"""mu-space blend of the head cVAE patch into the body-network prediction.

Verbatim port of notebook Cell 6. The seam sits at the *inferior edge of the
head patch*, not at the neck: the VAE's box is upper-head only (its bottom edge
sits above the face-mask neck), so there is no VAE data to blend with down
there. Above the seam the volume is VAE; below it, and everywhere outside the
patch, it is the body network. x/y patch edges are feathered because they fall in air.
"""

import numpy as np

TRANSITION_MM = 30.0   # z blend length from the patch bottom upward
FEATHER_XY_MM = 8.0    # soft x/y patch edges


def hu_to_mu(hu):
    h = np.asarray(hu, np.float32) + 1000.0
    return np.maximum(
        np.where(h < 1047.0, 9.6e-5 * h, 5.10e-5 * h + 4.71e-2), 0.0
    ).astype(np.float32)


def mu_to_hu(mu):
    mu = np.asarray(mu, np.float32)
    hi = 5.10e-5 * 1047.0 + 4.71e-2
    hu1000 = np.where(mu < hi, mu / 9.6e-5, (mu - 4.71e-2) / 5.10e-5)
    return (hu1000 - 1000.0).astype(np.float32)


def _edge_ramp(N, a, b, f):
    """1 inside [a, b), cosine-feathered over f voxels inside each edge, else 0."""
    w = np.zeros(N, np.float32)
    f = max(1.0, f)
    for i in range(int(a), int(b)):
        d = min(i - a, b - 1 - i)
        w[i] = 0.5 * (1 - np.cos(np.pi * (d + 0.5) / f)) if d < f else 1.0
    return w


def vae_weight(shape, ct_box, spacing):
    """VAE weight in [0, 1]: 1 well above the seam, 0 in the body."""
    (x0, x1), (y0, y1), (z0, z1) = ct_box
    xr = _edge_ramp(shape[0], x0, x1, FEATHER_XY_MM / spacing[0])
    yr = _edge_ramp(shape[1], y0, y1, FEATHER_XY_MM / spacing[1])
    tz = max(1.0, TRANSITION_MM / spacing[2])
    z = np.arange(shape[2], dtype=np.float32)
    zr = np.clip((z - z0) / tz, 0.0, 1.0)   # rises from the patch bottom
    zr[z >= z1] = 0.0                        # nothing above the patch top
    return xr[:, None, None] * yr[None, :, None] * zr[None, None, :]


def blend_head_into_body(body_hu, head_hu, ct_box, affine):
    """Paste the head patch into the body volume and blend in mu-space."""
    full_head = body_hu.copy()
    sl = tuple(slice(int(b[0]), int(b[1])) for b in ct_box)
    full_head[sl] = head_hu

    spacing = np.abs(np.diag(affine)[:3])
    w = vae_weight(body_hu.shape, ct_box, spacing)
    hybrid_mu = w * hu_to_mu(full_head) + (1.0 - w) * hu_to_mu(body_hu)
    hybrid_hu = mu_to_hu(hybrid_mu)

    per_slice = w.max(axis=(0, 1))
    stats = {
        "seam_z": int(ct_box[2][0]),
        "transition_slices": int((per_slice > 0).sum() - (per_slice >= 0.999).sum()),
        "spacing_mm": [float(s) for s in spacing],
    }
    return hybrid_hu, stats
