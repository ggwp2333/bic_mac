"""
Multi-GPU DDP training for the head cVAE - ALL-DATA fine-tune (finetune_head_v2).

Derived from ../finetune_head/train_vae_ddp.py. Same model, losses, augmentation
and submission machinery, byte-identical; three deliberate differences:

  1. n_val_subjects = 0 -> every cached subject with a CT trains. Nothing is held
     out, so `val_*` in the log is a FIT metric on subjects the model has seen,
     not a generalisation estimate. Checkpoint selection therefore switches to
     `select_metric` (default train_plain_mu).
  2. submission.every_epochs = 5 -> a leaderboard entry every 5 epochs once the
     trigger fires, instead of every 25.
  3. ckpt_every -> epoch_NNN/last_full are written every N epochs rather than
     every epoch (~20 s/epoch of datastore I/O at ~188 MB/s). best_model is still
     written the moment the selection metric improves.

Launched by the notebook via:
    torchrun --standalone --nproc_per_node=<N> train_vae_ddp.py --config <ddp_config.json>

The config / dataset / losses / model blocks below are the VERBATIM notebook cells, so there is a
single source of truth. Only the training loop is DDP-specific. Rank 0 writes checkpoints in the
exact format the notebook's inference cell loads ({"epoch","model","cfg","metrics"} + last_full.pth).
Requires the head cache to already exist (run the notebook's cache cell first).
"""
import matplotlib
matplotlib.use("Agg")                       # headless subprocess
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# ============ NOTEBOOK CELL 1 (config) ============
# Cell 1 - Imports + config
import os, sys, csv, json, math, time, shutil, subprocess
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.ndimage import affine_transform

ROOT      = os.environ.get("BIC_MAC_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))
DATA_DIR  = f"{ROOT}/bic-mac-data/train"
VAL_DIR   = f"{ROOT}/bic-mac-data/val"          # challenge val (no CT) - only cached if you ask
# --- pretrain_brain: the only structural change from rithihas_brain's script is
# --- that the cache and the subject list are inputs rather than constants, so the
# --- same model, losses, augmentation and loop can be pointed at a SynthRAD cache.
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "head_cache_bicmac")   # ../build_head_cache.py
RUN_DIR   = os.path.join(OUT_DIR, "runs", "head_finetune_v2")

# ---- crop geometry (same numbers as version_0.ipynb) -------------------------
# +x=left  +y=anterior/face  +z=superior(up)
POS_VOX     = (65, 70, 110)
NEG_VOX     = (65, 100, 10)
COND_MARGIN = (20, 20, 20)      # extra voxels per side for MRI / NAC-PET / topogram

CT_SIZE   = tuple(p + n for p, n in zip(POS_VOX, NEG_VOX))                  # (130, 170, 120)
COND_SIZE = tuple(s + 2 * m for s, m in zip(CT_SIZE, COND_MARGIN))          # (170, 210, 160)

# Network canvas: cond box padded up to a multiple of 16 (4 pooling levels).
def _up16(v): return int(math.ceil(v / 16) * 16)
PAD_SHAPE = tuple(_up16(s) for s in COND_SIZE)                              # (176, 224, 160)

CT_KEY    = "ct-label/ct.nii.gz"
COND_KEYS = ["features/nacpet.nii.gz",
             "features/mri_combined_in_phase.nii.gz",
             "features/mri_combined_out_phase.nii.gz"]      # topogram handled separately (X,1,Z)
TOPO_KEY  = "features/topogram.nii.gz"
PMASK_KEY = "ct-label/prediction_mask.nii.gz"
BMASK_KEY = "ct-label/body_seg.nii.gz"
REF_KEY   = "features/nacpet.nii.gz"
CHANNELS  = ["nacpet", "mri_in_phase", "mri_out_phase", "topogram"]

# ---- CT normalization (identical to baseline_v3 transforms.py) ---------------
HU_MIN, HU_MAX = -1000.0, 2000.0            # ScaleIntensityRanged -> [0, 1], clipped
def hu_to_norm(hu):   return np.clip((hu - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)
AIR_NORM = float(hu_to_norm(np.float32(-1000.0)))   # 0.0 -> canvas fill value

# ---- training config (mirrors baseline_v3/config.yaml) ----------------------
CFG = dict(
    epochs            = 100,      # all-data fine-tune ceiling
    learning_rate     = 2e-4,     # initialised, not cold: full-strength LR discards the prior
    weight_decay      = 1e-5,
    batch_size        = 2,        # DA: as many head volumes as VRAM allows (raise until OOM)
    grad_accum_steps  = 4,        # effective batch = batch_size * this
    num_workers       = 6,        # parallelize the on-the-fly rotation/shift augmentation

    ct_loss_weight    = 0.0,      # L1 on normalized CT (HU)
    umap_loss_weight  = 1.0,      # MAE on the Carney-derived umap (mu)  <- the challenge target

    tissue_loss = dict(           # identical to config.yaml
        enabled=True, blend=0.5,
        lung_air_max_hu=-500, bone_min_hu=300,
        lung_air_weight=1.5, soft_tissue_weight=1.0, bone_weight=2.0,
        interface_multiplier=1.5,
    ),

    # ---- VAE specific -------------------------------------------------------
    base_ch           = 64,       # same width as the baseline UNet
    post_base_ch      = 32,       # posterior encoder is narrower (memory)
    z_ch              = 16,        # latent channels at 1/16 resolution
    beta              = 1e-3,     # KL weight (after warm-up)
    kl_warmup_epochs  = 5,        # linear 0 -> beta (latent already organised by the pretrain)
    free_bits         = 0.05,     # per-latent-element KL floor (nats), 0 disables

    # ---- output smoothness (CT is far more uniform than the noisy MRI/PET) --
    smooth_loss_weight = 0.1,     # edge-aware TV on the predicted CT; 0 disables. try 0.05-0.3
    smooth_edge_scale  = 0.02,    # norm-CT grad scale: smaller = keep bone/air edges sharp; large -> plain TV

    # ---- physics-informed umap losses (Shi et al. 2023, Phys Med Biol 68:035014) --
    gdl_loss_weight   = 0.5,      # gradient-difference loss on the umap (crisp bone/air edges); 0 disables
    lip_loss_weight   = 0.2,      # line-integral projection loss on the umap (attenuation-sinogram consistency)
    lip_angles        = 4,        # rotated umap projections about the axial axis (+3 orthogonal always on)
    lip_downsample    = 2,        # project the umap at 1/N resolution (memory/speed)

    # ---- data ---------------------------------------------------------------
    n_val_subjects    = 0,        # 0 = ALL-DATA: nothing held out, every cached CT trains
    monitor_subjects  = 2,        # subjects still scored each epoch when n_val_subjects=0.
                                  # They are IN the training set, so the numbers are fit, not held-out.
    select_metric     = "train_plain_mu",   # best_model.pth selection. In all-data mode val_* is not
                                            # held out, so selecting on it would select on memorisation.
    ckpt_every        = 5,        # write epoch_NNN/last_full every N epochs; best_model always on improvement
    cache_in_ram      = True,
    aug_shift         = (10, 10, 10),  # DA: random translation +/- voxels in x, y, z
    aug_rot_deg       = (10, 10, 10),  # DA: random rotation +/- degrees about x, y, z (shared over all modalities)
    aug_flip_x        = False,         # left/right flip of the head
    samples_per_epoch = 420,           # DA: augmented samples drawn per epoch from the ~66 subjects
    seed              = 0,
)

os.makedirs(CACHE_DIR, exist_ok=True)
for sub in ("checkpoints", "logs", "plots"):
    os.makedirs(os.path.join(RUN_DIR, sub), exist_ok=True)

def list_subjects(cache_dir):
    """Subjects are whatever the cache holds. build_cache.py writes subjects.json
    after pruning cases whose head box falls mostly outside the volume."""
    manifest = os.path.join(cache_dir, "subjects.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            names = sorted(json.load(f)["subjects"])
    else:
        names = sorted(f[:-4] for f in os.listdir(cache_dir) if f.endswith(".npz"))
    # A record with no "ct" is an inference-only cache entry and cannot be
    # trained on. head_cache_v2 holds 79 records: the 75 BIC-MAC training
    # subjects plus sub-004/009/010/018, the challenge validation subjects,
    # which ship no CT and were cached only so the head could be predicted for
    # submission. The original script never met them because it listed subjects
    # from bic-mac-data/train; listing from the cache does.
    usable = []
    for n in names:
        p = os.path.join(cache_dir, f"{n}.npz")
        try:
            with np.load(p) as z:
                if "ct" in z.files:
                    usable.append(n)
        except Exception:
            continue
    dropped = len(names) - len(usable)
    if dropped:
        print(f"[cache] {dropped} record(s) without CT skipped (inference-only)")
    return usable

train_subs_all = list_subjects(CACHE_DIR) if os.path.isdir(CACHE_DIR) else []
print(f"{len(train_subs_all)} cached head volumes in {CACHE_DIR}")
print(f"CT box   {CT_SIZE}   (+{POS_VOX} / -{NEG_VOX})")
print(f"cond box {COND_SIZE} (CT box + {COND_MARGIN} per side, same center)")
print(f"network canvas (padded to /16) {PAD_SHAPE} = {np.prod(PAD_SHAPE)/1e6:.1f}M voxels")
print("run dir:", RUN_DIR)


# ================= DDP config overrides (from the launcher's JSON) =================
_ap = argparse.ArgumentParser(); _ap.add_argument("--config", default=None)
_args, _ = _ap.parse_known_args()
if _args.config and os.path.exists(_args.config):
    with open(_args.config) as _f:
        _ov = json.load(_f)
    if "run_dir" in _ov:                     # optional: redirect outputs (used for smoke tests)
        RUN_DIR = _ov.pop("run_dir")
        os.makedirs(RUN_DIR, exist_ok=True)
    if "cache_dir" in _ov:
        CACHE_DIR = _ov.pop("cache_dir")
    CFG.update(_ov)                          # notebook CFG wins; geometry constants stay as embedded
    print("[ddp] CFG overrides applied:", list(_ov.keys()), "| RUN_DIR", RUN_DIR)

# ============ NOTEBOOK CELL 3 (dataset) ============
# Cell 3 - Dataset: cond canvas + CT/mask canvases, all on the (padded) cond grid
def _place_offset(actual_lo, nominal_lo, pad_lo):
    """Where a crop starts inside the canvas: nominal box origin + pad, shifted by any clamping."""
    return np.asarray(actual_lo) - np.asarray(nominal_lo) + np.asarray(pad_lo)

PAD_LO = np.array([(P - C) // 2 for P, C in zip(PAD_SHAPE, COND_SIZE)])   # symmetric pad of the cond box

class HeadDataset(Dataset):
    def __init__(self, subjects, train=True, cfg=CFG, cache_in_ram=None):
        self.subjects = list(subjects)
        self.train = train
        self.cfg = cfg
        self.cache_in_ram = cfg["cache_in_ram"] if cache_in_ram is None else cache_in_ram
        self._ram = {}
        self.rng = np.random.default_rng(cfg["seed"] + (0 if train else 1))

    def __len__(self):
        if self.train:
            return int(self.cfg.get("samples_per_epoch", len(self.subjects)))
        return len(self.subjects)

    def _raw(self, name):
        if name in self._ram:
            return self._ram[name]
        d = np.load(os.path.join(CACHE_DIR, f"{name}.npz"))
        rec = {k: d[k] for k in d.files}
        if self.cache_in_ram:
            self._ram[name] = rec
        return rec

    def build(self, name, jitter=None, flip=False, rot=None):
        r = self._raw(name)
        cond_lo = _place_offset(r["cond_box"][:, 0], r["cond_box_nom"][:, 0], PAD_LO)
        ct_lo   = _place_offset(r["ct_box"][:, 0],   r["cond_box_nom"][:, 0], PAD_LO)
        if jitter is not None:
            cond_lo = cond_lo + jitter
            ct_lo   = ct_lo + jitter

        cond_np = r["cond"].astype(np.float32)                       # (3, x, y, z)
        topo_np = np.repeat(r["topo"].astype(np.float32), cond_np.shape[2], axis=1)[None]
        stack   = np.concatenate([cond_np, topo_np], 0)              # (4, x, y, z)
        sub, div = r["norm"][:, 0][:, None, None, None], r["norm"][:, 1][:, None, None, None]
        stack = (stack - sub) / div

        cond = np.zeros((4, *PAD_SHAPE), np.float32)
        _paste(cond, stack, cond_lo, channel=True)

        out = {"cond": cond, "subject": name}
        if "ct" in r:
            ct = np.full(PAD_SHAPE, AIR_NORM, np.float32)
            _paste(ct, hu_to_norm(r["ct"].astype(np.float32)), ct_lo)
            pm = np.zeros(PAD_SHAPE, np.float32); _paste(pm, r["pmask"].astype(np.float32), ct_lo)
            bm = np.zeros(PAD_SHAPE, np.float32); _paste(bm, r["bmask"].astype(np.float32), ct_lo)
            out.update(ct=ct[None], pmask=pm[None], bmask=bm[None])
        if flip:
            for k in ("cond", "ct", "pmask", "bmask"):
                if k in out:
                    out[k] = np.ascontiguousarray(out[k][:, ::-1])
        if rot is not None and any(rot):
            _rotate_item(out, rot)
        return out

    def __getitem__(self, i):
        if self.train:
            rng = np.random.default_rng()                    # fresh entropy -> unique per worker/epoch
            name = self.subjects[int(rng.integers(len(self.subjects)))]   # random subject from the pool
            s = self.cfg["aug_shift"]
            jitter = (np.array([int(rng.integers(-v, v + 1)) if v else 0 for v in s])
                      if any(s) else None)
            rd = self.cfg["aug_rot_deg"]
            rot = ([float(rng.uniform(-v, v)) if v else 0.0 for v in rd] if any(rd) else None)
            flip = bool(self.cfg["aug_flip_x"] and rng.random() < 0.5)
        else:
            name, jitter, rot, flip = self.subjects[i], None, None, False
        item = self.build(name, jitter, flip, rot)
        return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in item.items()}

def _paste(canvas, arr, lo, channel=False):
    """Paste `arr` into `canvas` at voxel offset `lo`, clipping whatever falls outside."""
    spatial = canvas.shape[1:] if channel else canvas.shape
    src, dst = [], []
    for a in range(3):
        s0 = int(lo[a]); s1 = s0 + arr.shape[a + 1 if channel else a]
        d0, d1 = max(s0, 0), min(s1, spatial[a])
        if d1 <= d0:
            return
        src.append(slice(d0 - s0, d1 - s0)); dst.append(slice(d0, d1))
    if channel:
        canvas[(slice(None), *dst)] = arr[(slice(None), *src)]
    else:
        canvas[tuple(dst)] = arr[tuple(src)]

def _rot_matrix(angles_deg):
    ax, ay, az = np.radians(angles_deg)
    cx, sx = np.cos(ax), np.sin(ax); cy, sy = np.cos(ay), np.sin(ay); cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _rotate_vol(vol, R, order, cval):
    c = (np.array(vol.shape) - 1) / 2.0
    return affine_transform(vol, R, offset=c - R @ c, order=order,
                            mode="constant", cval=cval, output=np.float32)

def _rotate_item(out, angles):
    """Rotate every modality by the SAME angles (keeps CT <-> cond alignment)."""
    R = _rot_matrix(angles)
    cond = out["cond"]
    out["cond"] = np.stack([_rotate_vol(cond[c], R, 1, 0.0) for c in range(cond.shape[0])], 0)
    if "ct" in out:
        out["ct"]    = _rotate_vol(out["ct"][0],    R, 1, AIR_NORM)[None]
        out["pmask"] = (_rotate_vol(out["pmask"][0], R, 0, 0.0) > 0.5).astype(np.float32)[None]
        out["bmask"] = (_rotate_vol(out["bmask"][0], R, 0, 0.0) > 0.5).astype(np.float32)[None]
    return out

# ---- split: sub-000 / sub-001 held out, exactly like baseline_v3 -------------
# Recomputed here rather than at import time: CACHE_DIR may have been redirected
# by the launcher config, which is parsed after the listing above.
train_subs_all = list_subjects(CACHE_DIR)

# Two id conventions, two splits. BIC-MAC subjects are "sub-NNN" and must keep
# v1_DA's exact held-out pair (sub-000 / sub-001) so this run stays comparable
# to the head already in the shipping stack. SynthRAD ids look like 1HNA001, so
# there the fallback holds out every k-th subject, spanning the centres instead
# of taking a contiguous block from one of them.
ALL_DATA = int(CFG["n_val_subjects"]) <= 0
# monitor_subjects: 0 switches the per-epoch evaluation off entirely. In
# all-data mode those subjects are *in* the training set, so `val body mu-MAE`
# measures fit, not generalisation, and reporting it invites reading it as
# validation. With it off the only validation is the Codabench submission.
NO_EVAL = ALL_DATA and int(CFG.get("monitor_subjects", 0)) <= 0
_n_val = int(CFG["monitor_subjects"]) if ALL_DATA else int(CFG["n_val_subjects"])
_n_val = max(_n_val, 1)
if all(s.startswith("sub-") and s.split("-")[-1].isdigit() for s in train_subs_all):
    val_subs = [s for s in train_subs_all if int(s.split("-")[-1]) < _n_val]
else:
    _step    = max(len(train_subs_all) // _n_val, 1)
    val_subs = [train_subs_all[i] for i in range(0, len(train_subs_all), _step)][:_n_val]
if ALL_DATA:
    # Nothing is held out. val_subs stays as a per-epoch probe so the log keeps
    # its val_* columns and the curves remain readable, but those subjects are
    # also trained on - the numbers are fit quality, not generalisation.
    train_subs = list(train_subs_all)
    if NO_EVAL:
        print(f"ALL-DATA: {len(train_subs)} train / 0 held out | no evaluation pass; "
              f"validation is the Codabench submission only")
    else:
      print(f"ALL-DATA: {len(train_subs)} train / 0 held out "
          f"| monitor (in-train, NOT held out) = {val_subs}")
    print(f"ALL-DATA: checkpoint selection uses {CFG['select_metric']!r}, not val_body_mu_mae")
else:
    train_subs = [s for s in train_subs_all if s not in val_subs]
    print(f"split: {len(train_subs)} train / {len(val_subs)} val -> val = {val_subs}")

train_ds = HeadDataset(train_subs, train=True)
val_ds   = HeadDataset(val_subs,   train=False)
_s = train_ds[0]
print({k: tuple(v.shape) for k, v in _s.items() if torch.is_tensor(v)})
print("cond per-channel mean/std:",
      [(round(float(_s['cond'][c].mean()), 3), round(float(_s['cond'][c].std()), 3)) for c in range(4)])
print("loss voxels (pmask):", int(_s["pmask"].sum()), " body voxels:", int(_s["bmask"].sum()))

# ============ NOTEBOOK CELL 5 (losses) ============
# Cell 5 - Carney HU->mu, tissue weighting, blended umap MAE (port of baseline_v3/losses.py) + KL
_A, _B, _BP   = 5.10e-5, 4.71e-2, 1047.0     # must match evaluation/metrics/ct_whole_body_mae.py
_LOW_SLOPE    = 9.6e-5
_HU1000_RANGE = 3000.0
_HU_OFFSET    = -1000.0

def norm_ct_to_mu(norm_ct):
    hu1000 = norm_ct.float() * _HU1000_RANGE
    mu = torch.where(hu1000 < _BP, _LOW_SLOPE * hu1000, _A * hu1000 + _B)
    return mu.clamp(min=0.0)

def norm_ct_to_hu(norm_ct):
    return norm_ct.float() * _HU1000_RANGE + _HU_OFFSET

def tissue_class_map(target_norm, lung_air_max_hu, bone_min_hu):
    hu = norm_ct_to_hu(target_norm)
    classes = torch.ones_like(hu, dtype=torch.long)
    classes[hu < lung_air_max_hu] = 0
    classes[hu >= bone_min_hu] = 2
    return classes

def tissue_interface_mask(classes, mask=None):
    cf = classes.float()
    if mask is None:
        mx = mn = cf; valid = None
    else:
        valid = mask.bool()
        mx = torch.where(valid, cf, -torch.ones_like(cf))
        mn = torch.where(valid, cf, 3 * torch.ones_like(cf))
    local_max = F.max_pool3d(F.pad(mx, (1,) * 6, mode="replicate"), 3, stride=1)
    local_min = -F.max_pool3d(F.pad(-mn, (1,) * 6, mode="replicate"), 3, stride=1)
    inter = local_max.ne(local_min)
    return inter if valid is None else inter & valid

def tissue_weight_map(target_norm, cfg, mask=None):
    classes = tissue_class_map(target_norm.detach(), cfg["lung_air_max_hu"], cfg["bone_min_hu"])
    w = torch.full_like(target_norm, cfg["soft_tissue_weight"], dtype=torch.float32)
    w.masked_fill_(classes == 0, cfg["lung_air_weight"])
    w.masked_fill_(classes == 2, cfg["bone_weight"])
    inter = tissue_interface_mask(classes, mask)
    return w * torch.where(inter, w.new_tensor(cfg["interface_multiplier"]), w.new_tensor(1.0))

def umap_mae_loss(pred_norm, target_norm, mask=None):
    ae = (norm_ct_to_mu(pred_norm) - norm_ct_to_mu(target_norm)).abs()
    if mask is None:
        return ae.mean()
    m = mask.to(ae.dtype)
    return (ae * m).sum() / m.sum().clamp(min=1.0)

def tissue_weighted_umap_mae_loss(pred_norm, target_norm, mask, cfg):
    ae = (norm_ct_to_mu(pred_norm) - norm_ct_to_mu(target_norm)).abs()
    w = tissue_weight_map(target_norm, cfg, mask) * mask.to(torch.float32)
    return (ae * w).sum() / w.sum().clamp(min=1.0)

def blended_umap_mae_loss(pred_norm, target_norm, mask, cfg):
    """(blended loss, plain mu-MAE, tissue-weighted mu-MAE)."""
    plain = umap_mae_loss(pred_norm, target_norm, mask)
    if not cfg["enabled"]:
        return plain, plain, plain
    weighted = tissue_weighted_umap_mae_loss(pred_norm, target_norm, mask, cfg)
    b = cfg["blend"]
    return (1.0 - b) * plain + b * weighted, plain, weighted

def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p, free_bits=0.0):
    """KL(q||p) per latent element, averaged (with an optional per-element floor)."""
    kl = 0.5 * (logvar_p - logvar_q
                + (logvar_q - logvar_p).exp()
                + (mu_q - mu_p).pow(2) * (-logvar_p).exp() - 1.0)
    if free_bits > 0:
        kl = kl.clamp(min=free_bits)
    return kl.mean()

def edge_aware_smoothness_loss(pred_norm, target_norm, mask=None, edge_scale=0.02):
    """Anisotropic TV on the prediction, DOWN-weighted where the target CT has an edge.
    Pushes flat regions (brain / soft tissue) to be as smooth and uniform as the real CT
    without blurring true bone/air boundaries. Large edge_scale -> plain (unweighted) TV."""
    p, t = pred_norm.float(), target_norm.float()
    num = p.new_zeros(()); den = p.new_zeros(())
    for ax in (2, 3, 4):                              # X, Y, Z of (B, 1, X, Y, Z)
        n = p.size(ax) - 1
        if n <= 0:
            continue
        gp = (p.narrow(ax, 1, n) - p.narrow(ax, 0, n)).abs()          # |grad(pred)|
        gt = (t.narrow(ax, 1, n) - t.narrow(ax, 0, n)).abs()          # |grad(target)|
        w  = torch.exp(-gt.detach() / edge_scale)                     # ~1 flat, ~0 at target edges
        if mask is not None:
            m = mask.float()
            w = w * m.narrow(ax, 1, n) * m.narrow(ax, 0, n)           # only where both voxels valid
        num = num + (gp * w).sum()
        den = den + w.sum()
    return num / den.clamp(min=1.0)

# quick check against the numpy metric's hu_to_mu
_hu = torch.tensor([-1000.0, -500.0, 0.0, 47.0, 300.0, 1000.0])
_np_mu = np.where(_hu.numpy() + 1000 < 1047, 9.6e-5 * (_hu.numpy() + 1000),
                  5.10e-5 * (_hu.numpy() + 1000) + 4.71e-2).clip(0, None)
assert np.allclose(norm_ct_to_mu((_hu - _HU_OFFSET) / _HU1000_RANGE).numpy(), _np_mu, atol=1e-7)
print("Carney HU->mu matches the challenge metric")


# --- Physics-informed umap losses (Shi et al. 2023, Phys Med Biol 68:035014) ---
# GDL = gradient-difference loss (crisp bone/air edges); LIP = line-integral
# projection loss (differentiable parallel-ray forward projection of the umap =
# the attenuation sinogram / ACF that PET reconstruction consumes). Operate on
# mu tensors [B,1,X,Y,Z]; in the training loop pass norm_ct_to_mu(pred/target).
def gradient_difference_loss(pred_mu, target_mu, mask=None):
    total = pred_mu.new_zeros(()); denom = pred_mu.new_zeros(())
    for ax in (2, 3, 4):
        n = pred_mu.size(ax) - 1
        if n <= 0: continue
        gp = (pred_mu.narrow(ax, 1, n) - pred_mu.narrow(ax, 0, n)).abs()
        gt = (target_mu.narrow(ax, 1, n) - target_mu.narrow(ax, 0, n)).abs()
        d = (gp - gt).abs()
        if mask is not None:
            m = mask.to(d.dtype); w = m.narrow(ax, 1, n) * m.narrow(ax, 0, n)
            total = total + (d * w).sum(); denom = denom + w.sum()
        else:
            total = total + d.sum(); denom = denom + d.numel()
    return total / denom.clamp(min=1.0)

def line_integral_projection_loss(pred_mu, target_mu, mask=None, n_angles=4, downsample=2):
    import math
    p = pred_mu.float(); t = target_mu.float()
    if mask is not None:
        m = mask.to(p.dtype); p = p * m; t = t * m
    if downsample and downsample > 1:
        p = F.avg_pool3d(p, downsample); t = F.avg_pool3d(t, downsample)
    loss = p.new_zeros(()); count = 0
    for ax in (2, 3, 4):                          # 3 orthogonal parallel-beam projections
        loss = loss + (p.mean(ax) - t.mean(ax)).abs().mean(); count += 1
    if n_angles and n_angles > 0:                 # rotated views about the axial (X) axis
        b = p.shape[0]
        for a in range(n_angles):
            th = math.pi * a / n_angles; cos, sin = math.cos(th), math.sin(th)
            theta = p.new_tensor([[1., 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0]]).unsqueeze(0).expand(b, 3, 4)
            grid = F.affine_grid(theta, p.shape, align_corners=False)
            pr = F.grid_sample(p, grid, align_corners=False, padding_mode="zeros")
            tr = F.grid_sample(t, grid, align_corners=False, padding_mode="zeros")
            loss = loss + (pr.mean(4) - tr.mean(4)).abs().mean(); count += 1
    return loss / max(count, 1)


# ============ NOTEBOOK CELL 6 (model) ============
# Cell 6 - Model
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.relu  = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.skip  = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        idt = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.skip is not None:
            idt = self.skip(idt)
        return self.relu(out + idt)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch)
        self.pool  = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.block(x)
        return x, self.pool(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up    = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        return self.block(torch.cat([x, skip], dim=1))

class CondUNetVAE(nn.Module):
    def __init__(self, in_channels=4, base=32, post_base=16, z_ch=8, dropout=0.2):
        super().__init__()
        self.z_ch = z_ch
        # conditioning encoder (baseline UNet encoder)
        self.enc1 = EncoderBlock(in_channels, base)
        self.enc2 = EncoderBlock(base, base * 2)
        self.enc3 = EncoderBlock(base * 2, base * 4)
        self.enc4 = EncoderBlock(base * 4, base * 8)
        self.bottleneck = nn.Sequential(ResidualBlock(base * 8, base * 16), nn.Dropout3d(dropout))
        # prior p(z | cond)
        self.prior = nn.Sequential(ResidualBlock(base * 16, base * 4),
                                   nn.Conv3d(base * 4, 2 * z_ch, 1))
        # posterior q(z | cond, CT)
        pb = post_base
        self.post = nn.Sequential(
            ResidualBlock(in_channels + 1, pb),      nn.MaxPool3d(2),
            ResidualBlock(pb, pb * 2),               nn.MaxPool3d(2),
            ResidualBlock(pb * 2, pb * 4),           nn.MaxPool3d(2),
            ResidualBlock(pb * 4, pb * 8),           nn.MaxPool3d(2),
            ResidualBlock(pb * 8, pb * 8),
        )
        self.post_head = nn.Conv3d(pb * 8, 2 * z_ch, 1)
        # decoder p(CT | z, cond)
        self.z_merge = ResidualBlock(base * 16 + z_ch, base * 16)
        self.dec4 = DecoderBlock(base * 16, base * 8)
        self.dec3 = DecoderBlock(base * 8, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 2)
        self.dec1 = DecoderBlock(base * 2, base)
        self.out_conv = nn.Conv3d(base, 1, 1)

    def encode_cond(self, cond):
        s1, p1 = self.enc1(cond)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        return self.bottleneck(p4), (s1, s2, s3, s4)

    def decode(self, b, skips, z):
        s1, s2, s3, s4 = skips
        x = self.z_merge(torch.cat([b, z], dim=1))
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.out_conv(x)

    @staticmethod
    def _split(h):
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar.clamp(-10.0, 10.0)

    def forward(self, cond, ct=None, sample=False):
        """ct given -> posterior z (training). ct None -> prior (mu_p, or a sample if sample=True)."""
        b, skips = self.encode_cond(cond)
        mu_p, logvar_p = self._split(self.prior(b))
        if ct is not None:
            mu_q, logvar_q = self._split(self.post_head(self.post(torch.cat([cond, ct], dim=1))))
            z = mu_q + torch.randn_like(mu_q) * (0.5 * logvar_q).exp() if self.training else mu_q
        else:
            mu_q, logvar_q = mu_p, logvar_p
            z = mu_p + torch.randn_like(mu_p) * (0.5 * logvar_p).exp() if sample else mu_p
        return self.decode(b, skips, z), (mu_q, logvar_q, mu_p, logvar_p)

def build_model(cfg=CFG, in_channels=4):
    return CondUNetVAE(in_channels=in_channels, base=cfg["base_ch"],
                       post_base=cfg["post_base_ch"], z_ch=cfg["z_ch"])

_m = build_model()
print(f"{sum(p.numel() for p in _m.parameters())/1e6:.1f}M parameters "
      f"| latent {CFG['z_ch']}x{tuple(d//16 for d in PAD_SHAPE)}")
del _m


# ================================= DDP training =================================
TISSUE   = CFG["tissue_loss"]
CT_W     = CFG["ct_loss_weight"]
UMAP_W   = CFG["umap_loss_weight"]
SMOOTH_W = CFG["smooth_loss_weight"]
SEDGE    = CFG["smooth_edge_scale"]
GDL_W    = float(CFG.get("gdl_loss_weight", 0.0))     # Shi et al. gradient-difference loss on mu
LIP_W    = float(CFG.get("lip_loss_weight", 0.0))     # Shi et al. line-integral projection loss on mu
LIP_ANG  = int(CFG.get("lip_angles", 4))
LIP_DS   = int(CFG.get("lip_downsample", 2))
ACCUM    = max(1, CFG["grad_accum_steps"])
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
LOG_CSV  = os.path.join(RUN_DIR, "logs", "train_log.csv")
FIELDS   = ["epoch", "lr", "beta",
            "train_total", "train_recon", "train_plain_mu", "train_weighted_mu", "train_kl",
            "train_smooth", "train_gdl", "train_lip",
            "val_plain_mu", "val_weighted_mu", "val_body_mu_mae", "val_ct_mae_hu", "secs"]


def recon_terms(pred, y, mask):
    mu_loss, plain, weighted = blended_umap_mae_loss(pred, y, mask, TISSUE)
    loss = UMAP_W * mu_loss
    if CT_W:
        valid = mask.to(pred.dtype)
        loss = loss + CT_W * (((pred - y).abs() * valid).sum() / valid.sum().clamp(min=1))
    return loss, mu_loss, plain, weighted


def kl_beta(epoch):
    w = CFG["kl_warmup_epochs"]
    return CFG["beta"] * (min(1.0, (epoch + 1) / w) if w > 0 else 1.0)


@torch.no_grad()
def evaluate(module, device, val_loader):
    module.eval()
    acc = dict(val_plain_mu=0.0, val_weighted_mu=0.0, val_body_mu_mae=0.0, val_ct_mae_hu=0.0)
    n = 0
    for batch in val_loader:
        cond  = batch["cond"].to(device, non_blocking=True)
        y     = batch["ct"].to(device, non_blocking=True)
        mask  = batch["pmask"].to(device, non_blocking=True).bool()
        bmask = batch["bmask"].to(device, non_blocking=True).bool()
        y = y.masked_fill(~mask, 0.0)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred, _ = module(cond, ct=None)
        pred = pred.float()
        _, _, plain, weighted = recon_terms(pred, y, mask)
        acc["val_plain_mu"]    += float(plain)
        acc["val_weighted_mu"] += float(weighted)
        bm = bmask.float()
        ae_mu = (norm_ct_to_mu(pred) - norm_ct_to_mu(y)).abs()
        acc["val_body_mu_mae"] += float((ae_mu * bm).sum() / bm.sum().clamp(min=1))
        ae_hu = (norm_ct_to_hu(pred) - norm_ct_to_hu(y)).abs()
        acc["val_ct_mae_hu"]   += float((ae_hu * bm).sum() / bm.sum().clamp(min=1))
        n += 1
    return {k: v / max(n, 1) for k, v in acc.items()}


def run_ddp():
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_main    = rank == 0
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(CFG["seed"] + rank); np.random.seed(CFG["seed"] + rank)

    def log(*a):
        if is_main: print(*a, flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "plots"), exist_ok=True)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                              num_workers=CFG["num_workers"], pin_memory=True,
                              persistent_workers=(CFG["num_workers"] > 0))
    val_loader = (DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
                  if is_main and not NO_EVAL else None)
    log(f"[ddp] world_size={world_size} | {len(train_ds)} samples/epoch "
        f"| per-rank batch {CFG['batch_size']} x accum {ACCUM} "
        f"-> global batch {world_size * CFG['batch_size'] * ACCUM}")

    model = build_model().to(device)
    ddp   = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=CFG["learning_rate"],
                                  weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    scaler    = torch.amp.GradScaler("cuda")

    # ---- resume (all ranks load the same weights/opt state) ----
    start_epoch, history = 0, []
    _full = os.path.join(CKPT_DIR, "last_full.pth")
    if os.path.exists(_full):
        ck = torch.load(_full, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, history = ck["epoch"] + 1, ck.get("history", [])
        log(f"[ddp] resumed from epoch {ck['epoch']} -> starting {start_epoch}")
    elif CFG.get("init_checkpoint"):
        # Fine-tuning FROM a pretrained head: weights only. The epoch counter,
        # optimizer and LR schedule all start fresh, so the fine-tune sees the
        # same schedule a from-scratch run would -- this is an initialisation,
        # not a resumption. `last_full.pth` above takes precedence so an
        # interrupted fine-tune still resumes correctly.
        _ip = CFG["init_checkpoint"]
        if not os.path.isabs(_ip):
            _ip = os.path.join(OUT_DIR, _ip)
        _ck = torch.load(_ip, map_location="cpu", weights_only=False)
        model.load_state_dict(_ck.get("model", _ck))
        log(f"[ddp] initialised from {_ip} (pretrained epoch {_ck.get('epoch')})")
    if is_main and not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
    SEL = CFG.get("select_metric") or "val_body_mu_mae"
    if SEL not in FIELDS:
        raise KeyError(f"select_metric {SEL!r} is not a logged column: {FIELDS}")
    best = min((h[SEL] for h in history), default=float("inf"))
    if is_main:
        log(f"[ddp] best_model.pth selected on {SEL}"
            + ("  (all-data: val_* is in-train fit, not held out)" if ALL_DATA else ""))

    # ---- loss-triggered leaderboard submissions -----------------------------
    # Same idea as finetune_v1: wait for the loss to hold below a threshold, then
    # submit on a fixed cadence. The cadence here is far wider, because this arm
    # scores a PET metric and every submission needs a full reconstruction
    # (~45 min for the four subjects) rather than a CT upload.
    SUB = dict(CFG.get("submission", {}))
    sub_on      = bool(SUB.get("enabled", False))
    sub_metric  = SUB.get("metric", "train_plain_mu")
    sub_thresh  = float(SUB.get("threshold", 0.006))
    sub_consec  = int(SUB.get("consecutive_epochs", 5))
    sub_every   = int(SUB.get("every_epochs", 5))
    sub_max     = int(SUB.get("max_submissions", 4))
    # The worker is this package's own sibling copy, so v2 keeps its own lock
    # and its own runs/submissions tree and cannot collide with finetune_head's.
    # Override with submission.script to point at a different chain.
    sub_script  = SUB.get("script") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "submit_head_arm.py")
    # Two trigger modes. `start_epoch` is the plain one -- submit from that epoch
    # on a fixed cadence, regardless of the loss -- and it takes precedence when
    # set. Without it the loss trigger applies: wait for `metric` to hold below
    # `threshold` for `consecutive_epochs`, then submit on the same cadence.
    sub_start   = SUB.get("start_epoch")
    sub_start   = int(sub_start) if sub_start is not None else None
    sub_ct_only = bool(SUB.get("ct_only", False))
    sub_streak, sub_fire_at, sub_spent = 0, None, 0
    if sub_start is not None:
        sub_fire_at = sub_start
        sub_spent = sum(1 for h in history
                        if int(h["epoch"]) >= sub_start
                        and (int(h["epoch"]) - sub_start) % sub_every == 0)
    else:
        for h in history:                   # replay so a resume does not re-spend
            v = h.get(sub_metric)
            if v is None:
                continue
            sub_streak = sub_streak + 1 if float(v) < sub_thresh else 0
            if sub_fire_at is None and sub_streak >= sub_consec:
                sub_fire_at = int(h["epoch"])
            if sub_fire_at is not None and (int(h["epoch"]) - sub_fire_at) % sub_every == 0:
                sub_spent += 1
    if is_main and sub_on:
        what = "CT only" if sub_ct_only else "CT + PET reconstruction"
        if sub_start is not None:
            log(f"[submit] trigger: from epoch {sub_start}, every {sub_every}, "
                f"max {sub_max} | {what}"
                + (f" | {sub_spent} already spent" if sub_spent else ""))
        else:
            log(f"[submit] trigger: {sub_metric} < {sub_thresh} for {sub_consec} epochs, "
                f"then every {sub_every} for {sub_max} submissions | {what}"
                + (f" | already fired at {sub_fire_at}, {sub_spent} spent" if sub_fire_at else ""))

    log(f"[ddp] training to {CFG['epochs']} epochs")
    for epoch in range(start_epoch, CFG["epochs"]):
        t0 = time.time(); beta = kl_beta(epoch)
        sampler.set_epoch(epoch); ddp.train()
        sums = torch.zeros(8, device=device)      # total, recon, plain, weighted, kl, smooth, gdl, lip
        nb_t = torch.zeros(1, device=device)
        optimizer.zero_grad(set_to_none=True)
        steps = len(train_loader)
        it = tqdm(train_loader, leave=False, desc=f"ep{epoch}") if is_main else train_loader
        for i, batch in enumerate(it):
            cond = batch["cond"].to(device, non_blocking=True)
            y    = batch["ct"].to(device, non_blocking=True)
            mask = batch["pmask"].to(device, non_blocking=True).bool()
            y = y.masked_fill(~mask, 0.0)
            with torch.amp.autocast("cuda"):
                pred, (mu_q, logvar_q, mu_p, logvar_p) = ddp(cond, ct=y)
                pred = pred.float()
                loss_rec, mu_loss, plain, weighted = recon_terms(pred, y, mask)
                kl = gaussian_kl(mu_q.float(), logvar_q.float(), mu_p.float(), logvar_p.float(),
                                 CFG["free_bits"])
                smooth = (edge_aware_smoothness_loss(pred, y, mask, SEDGE)
                          if SMOOTH_W > 0 else pred.new_zeros(()))
                gdl = (gradient_difference_loss(norm_ct_to_mu(pred), norm_ct_to_mu(y), mask)
                       if GDL_W > 0 else pred.new_zeros(()))
                lip = (line_integral_projection_loss(norm_ct_to_mu(pred), norm_ct_to_mu(y), mask,
                                                     n_angles=LIP_ANG, downsample=LIP_DS)
                       if LIP_W > 0 else pred.new_zeros(()))
                loss = loss_rec + beta * kl + SMOOTH_W * smooth + GDL_W * gdl + LIP_W * lip
            scaler.scale(loss / ACCUM).backward()
            if (i + 1) % ACCUM == 0 or (i + 1) == steps:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            sums += torch.stack([loss.detach(), loss_rec.detach(), plain.detach(),
                                 weighted.detach(), kl.detach(),
                                 torch.as_tensor(float(smooth), device=device),
                                 torch.as_tensor(float(gdl), device=device),
                                 torch.as_tensor(float(lip), device=device)])
            nb_t += 1
            if is_main:
                it.set_postfix(loss=f"{float(loss):.5f}", kl=f"{float(kl):.4f}")
        scheduler.step()

        dist.all_reduce(sums, op=dist.ReduceOp.SUM); dist.all_reduce(nb_t, op=dist.ReduceOp.SUM)
        m = (sums / nb_t.clamp(min=1)).tolist()
        means = dict(total=m[0], recon=m[1], plain=m[2], weighted=m[3], kl=m[4], smooth=m[5],
                     gdl=m[6], lip=m[7])

        if is_main:
            val = ({k: float("nan") for k in
                    ("val_plain_mu", "val_weighted_mu", "val_body_mu_mae", "val_ct_mae_hu")}
                   if NO_EVAL else evaluate(model, device, val_loader))
            row = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], beta=beta,
                       train_total=means["total"], train_recon=means["recon"],
                       train_plain_mu=means["plain"], train_weighted_mu=means["weighted"],
                       train_kl=means["kl"], train_smooth=means["smooth"],
                       train_gdl=means["gdl"], train_lip=means["lip"], **val,
                       secs=round(time.time() - t0, 1))
            history.append(row)
            with open(LOG_CSV, "a", newline="") as f:
                csv.DictWriter(f, FIELDS).writerow({k: row[k] for k in FIELDS})
            # ---- checkpoint I/O ------------------------------------------
            # epoch_NNN (0.85 GB) + last_full (2.55 GB) every epoch costs ~20 s on
            # this datastore (~188 MB/s) with the other ranks idle at the barrier.
            # ckpt_every=N writes them every N epochs plus the last; a crash-resume
            # then loses up to N epochs. Submission epochs must keep their file, so
            # ckpt_every should divide submission.every_epochs (5 and 5 by default).
            _every = max(1, int(CFG.get("ckpt_every", 1)))
            _keep  = (epoch % _every == 0) or (epoch == CFG["epochs"] - 1)
            ep_path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth")
            _light  = {"epoch": epoch, "model": model.state_dict(), "cfg": CFG, "metrics": row}
            if _keep:
                torch.save(_light, ep_path)
                torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                            "cfg": CFG, "history": history}, _full)
            tag = ""
            if row[SEL] < best:
                best = row[SEL]
                _bp = os.path.join(CKPT_DIR, "best_model.pth")
                if _keep:
                    shutil.copyfile(ep_path, _bp)
                else:
                    torch.save(_light, _bp)
                tag = "  <- best"

            # ---- fire a submission? decided from this epoch's logged metrics ----
            if sub_on:
                _v = row.get(sub_metric)
                if sub_start is None and _v is not None:
                    sub_streak = sub_streak + 1 if float(_v) < sub_thresh else 0
                    if sub_fire_at is None and sub_streak >= sub_consec:
                        sub_fire_at = epoch
                        log(f"[submit] {sub_metric} held below {sub_thresh} for "
                            f"{sub_consec} epochs; submitting at {epoch} then every "
                            f"{sub_every} ({sub_max} total)")
                if True:
                    due = (sub_fire_at is not None
                           and epoch >= sub_fire_at
                           and (epoch - sub_fire_at) % sub_every == 0
                           and sub_spent < sub_max)
                    if due:
                        sub_spent += 1
                        if not _keep:          # ckpt_every skipped it; the worker needs the file
                            torch.save(_light, ep_path)
                            log(f"[submit] wrote {os.path.basename(ep_path)} for the submission")
                        # Detached: head inference + merge + reconstruction take
                        # ~an hour, and training must not block on them. The
                        # worker holds a lock, so an epoch whose turn comes round
                        # while the previous submission is still running is
                        # skipped rather than queued.
                        cmd = [sys.executable, sub_script,
                               "--checkpoint", ep_path, "--epoch", str(epoch)]
                        if sub_ct_only:
                            cmd += ["--ct-only"]
                        if SUB.get("gpu_ids"):
                            cmd += ["--gpu-ids", str(SUB["gpu_ids"])]
                        if SUB.get("dry_run"):
                            cmd += ["--dry-run"]
                        _slog = os.path.join(RUN_DIR, "logs", f"submit_e{epoch:04d}.log")
                        with open(_slog, "w") as _lf:
                            subprocess.Popen(cmd, stdout=_lf, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                        log(f"[submit] launched {sub_spent}/{sub_max} for epoch "
                            f"{epoch} -> {_slog}")
            with open(os.path.join(RUN_DIR, "logs", "history.json"), "w") as f:
                json.dump(history, f, indent=1)
            _valtxt = ("" if NO_EVAL else
                       f"val body mu-MAE {val['val_body_mu_mae']:.5f} "
                       f"({val['val_ct_mae_hu']:.1f} HU) | ")
            print(f"ep {epoch:3d} | total {means['total']:.5f} recon {means['recon']:.5f} "
                  f"plain {means['plain']:.5f} kl {means['kl']:.4f} smooth {means['smooth']:.5f} | "
                  f"{_valtxt}{row['secs']:.0f}s{tag}", flush=True)
        dist.barrier()

    if is_main:
        print(f"done. best {SEL}: {best}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    run_ddp()
