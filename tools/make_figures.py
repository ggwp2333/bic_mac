"""Regenerate the README figures in assets/.

Charts are drawn from the CSV/JSON files in results/ and need nothing else.
The pseudo-NAC panel also needs the BIC-MAC data (set BIC_MAC_DATA_ROOT or put
it under data/) and is much faster with a GPU.

    python tools/make_figures.py              # everything
    python tools/make_figures.py --charts     # results/ only, no data needed

Every chart is written twice, <name>_light.png and <name>_dark.png, and the
README picks one with a <picture> element so it follows the GitHub theme.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ASSETS = REPO / "assets"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834",
                  dim="#c3c2b7"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926",
                 dim="#55544f"),
}


def style(t):
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.grid": True, "axes.axisbelow": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 18, "legend.frameon": False,
    })


def subtitle(ax, text, t):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=t["ink2"], fontsize=9.5,
            va="bottom")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------- charts
def leaderboard(t, out):
    rows = read_csv(RESULTS / "leaderboard_ablation.csv")
    base = rows[0]
    final = rows[-1]
    metrics = [("ct_mu_map_mae", "CT μ-MAE"), ("pet_whole_body_suv_mae", "SUV MAE"),
               ("pet_organ_bias", "Organ bias"), ("pet_brain_outlier_score", "Brain outlier")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw=dict(width_ratios=[1, 1.45]))

    # (a) final vs baseline, all four ranked metrics
    pct = [100 * (float(final[k]) / float(base[k]) - 1) for k, _ in metrics]
    y = np.arange(len(metrics))[::-1]
    a.barh(y, pct, height=0.55, color=t["s1"])
    for yi, p in zip(y, pct):
        label = f"{p:.1f}%".replace("-", "−")
        if p < -20:   # long bar: label inside its end
            a.text(p + 0.8, yi, label, va="center", ha="left", color=t["surface"], fontsize=10,
                   fontweight="bold")
        else:
            a.text(p - 0.8, yi, label, va="center", ha="right", color=t["ink"], fontsize=10)
    a.set_yticks(y, [m for _, m in metrics])
    a.tick_params(axis="y", colors=t["ink2"], length=0)
    a.set_xlim(-40, 0)
    a.axvline(0, color=t["axis"], lw=1)
    a.grid(axis="y", visible=False)
    a.set_xlabel("change vs. challenge baseline (lower is better)")
    a.set_title("Final entry improves all four ranked metrics")
    subtitle(a, "Codabench validation phase, 4 subjects, full CT + PET reconstruction", t)

    # (b) brain outlier: whole-body models alone vs. successive head layers
    seq = [("baseline", base),
           ("body w64\nno head", rows[1]),
           ("pretrained\nbody,\nno head", rows[5]),
           ("+ cVAE\nhead", rows[6]),
           ("+ SynthRAD\npretrain", rows[7]),
           ("+ all-data\nfine-tune\n(final)", rows[8])]
    vals = [float(r["pet_brain_outlier_score"]) for _, r in seq]
    x = np.arange(len(seq))
    colors = [t["dim"]] * (len(seq) - 1) + [t["s1"]]
    b.bar(x, vals, width=0.6, color=colors)
    b.axhline(vals[0], color=t["muted"], lw=1)
    b.text(len(seq) - 0.55, vals[0] * 1.03, "baseline", color=t["muted"], fontsize=9, ha="right")
    for xi, v in zip(x, vals):
        b.text(xi, v + 0.0006, f"{v:.4f}", ha="center", va="bottom", fontsize=9,
               color=t["ink"] if xi == x[-1] else t["ink2"])
    b.set_xticks(x, [s for s, _ in seq], fontsize=9.5)
    b.tick_params(axis="x", colors=t["ink2"], length=0)
    b.set_ylim(0, 0.042)
    b.grid(axis="x", visible=False)
    b.set_ylabel("PET brain outlier score")
    b.set_title("A dedicated head model fixes the brain")
    subtitle(b, "Whole-body networks double the brain error; the head layer buys it back", t)

    fig.tight_layout(w_pad=3)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def training(t, out):
    body = read_csv(RESULTS / "logs" / "body_finetune.csv")
    ckpt = read_csv(RESULTS / "body_finetune_checkpoints.csv")
    head = read_csv(RESULTS / "logs" / "head_finetune.csv")
    head_ck = read_csv(RESULTS / "head_finetune_checkpoints.csv")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ep = [int(r["epoch"]) for r in body]
    mu = [float(r["train_plain_mu"]) for r in body]
    a.plot(ep, mu, color=t["s1"], lw=2)
    a.axhline(0.006, color=t["muted"], lw=1)
    a.set_ylim(0.0045, 0.0105)
    a.set_xlabel("epoch")
    a.set_ylabel("training μ-MAE (cm⁻¹)")
    a.set_title("Body fine-tune")
    subtitle(a, "all 75 subjects; gray line = checkpoint-scoring trigger (0.006)", t)

    e = np.array([int(r["epoch"]) for r in ckpt])
    s = np.array([float(r["ct_mu_map_mae"]) for r in ckpt])
    best = int(np.argmin(s))
    b.plot(e, s, color=t["s1"], lw=2, marker="o", ms=4.5)
    b.plot(e[best], s[best], "o", ms=10, mfc=t["s1"], mec=t["surface"], mew=2)
    b.annotate(f"epoch {e[best]}: {s[best]:.7f}\n(released body)", (e[best], s[best]),
               xytext=(12, -6), textcoords="offset points", color=t["ink"], fontsize=9,
               va="top")
    b.set_ylim(0.00565, 0.00598)
    b.set_xlabel("epoch")
    b.set_ylabel("leaderboard CT μ-MAE")
    b.set_title("Body checkpoints on the leaderboard")
    subtitle(b, f"{len(e)} scored checkpoints, every 5 epochs after the trigger", t)

    he = [int(r["epoch"]) for r in head]
    hm = [float(r["train_plain_mu"]) for r in head]
    c.plot(he, hm, color=t["s1"], lw=2)
    sel = {int(r["head_epoch"]): float(r["ct_mu_map_mae"]) for r in head_ck}
    best_e = min(sel, key=sel.get)
    for k in sel:
        y = hm[he.index(k)]
        chosen = k == best_e
        c.plot(k, y, "o", ms=9 if chosen else 6, mfc=t["s1"] if chosen else t["surface"],
               mec=t["s1"] if not chosen else t["surface"], mew=2)
        c.annotate(str(k), (k, y), xytext=(0, -14), textcoords="offset points",
                   ha="center", color=t["ink"] if chosen else t["ink2"], fontsize=8.5)
    c.text(0.97, 0.95, "merged CT μ-MAE by head epoch\n" + "\n".join(
        f"e{k}  {v:.7f}" + ("  (released)" if k == best_e else "") for k, v in sel.items()),
        transform=c.transAxes, ha="right", va="top", fontsize=8.5, color=t["ink2"],
        family="DejaVu Sans Mono")
    c.set_ylim(0.0035, 0.0105)
    c.set_xlabel("epoch")
    c.set_ylabel("training μ-MAE, head crop (cm⁻¹)")
    c.set_title("Head cVAE fine-tune")
    subtitle(c, "init: SynthRAD-pretrained head, epoch 80", t)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(out, dpi=170)
    plt.close(fig)


# ------------------------------------------------------------ pseudo-NAC panel
def pseudo_nac(t, out, data_root: Path, subject="sub-000"):
    import nibabel as nib
    import torch
    from scipy import ndimage

    sys.path.insert(0, str(REPO / "training" / "body" / "stage1_ct_pretrain"))
    from synthesize_nac import attenuation_weighting, resample_like

    d = data_root / "bic-mac-data" / "train" / subject
    ct_img = nib.load(d / "ct-label" / "ct.nii.gz")
    ct = ct_img.get_fdata(dtype=np.float32)
    body = np.asanyarray(nib.load(d / "ct-label" / "body_seg.nii.gz").dataobj) > 0
    nac = nib.load(d / "features" / "nacpet.nii.gz").get_fdata(dtype=np.float32)
    ac_img = nib.load(d / "pet-label" / "pet.nii.gz")
    ac = resample_like(ac_img.get_fdata(dtype=np.float32), ac_img, ct_img)
    W = attenuation_weighting(ct, device="cuda" if torch.cuda.is_available() else "cpu")
    syn = ac * W

    def norm(v):
        return v / max(float(v[body].mean()), 1e-9)

    # coronal slice through the body's centre of mass, superior at the top
    yc = int(round(ndimage.center_of_mass(body)[1]))
    def cor(v):
        return np.rot90(v[:, yc, :])

    panels = [("AC-PET (what public data ships)", norm(ac)),
              ("synthesised NAC  =  AC · W", norm(syn)),
              ("real NAC-PET (challenge input)", norm(nac))]

    v = json.load(open(RESULTS / "pseudo_nac_validation.json"))
    bins = ["0-10", "20-30", "40-60", "80-120", "120-400"]
    labels = ["0–10", "20–30", "40–60", "80–120", ">120\n(4 subj.)"]
    # a depth bin is only recorded for subjects with >1000 voxels in it
    def bin_mean(k, key):
        return np.mean([s["depth_profile"][k][key] for s in v["subjects"] if k in s["depth_profile"]])
    true_r = [bin_mean(k, "true_ac_over_nac") for k in bins]
    syn_r = [bin_mean(k, "synth_ac_over_nac") for k in bins]

    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.45], wspace=0.08)
    for i, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cor(img), cmap="inferno", vmin=0, vmax=3.0, aspect=2.0 / 1.523,
                  interpolation="lanczos")
        ax.set_title(title, fontsize=10.5, fontweight="normal", loc="center", pad=6,
                     color=t["ink"])
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(len(bins))
    ax.plot(x, true_r, color=t["s1"], lw=2, marker="o", ms=7, label="real AC / NAC")
    ax.plot(x, syn_r, color=t["s2"], lw=2, marker="s", ms=7, label="AC / synthesised NAC")
    ax.axhline(1, color=t["axis"], lw=1)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.3, len(bins) - 0.2)
    ax.set_xlabel("depth below body surface (mm)")
    ax.set_ylabel("AC / NAC, mean ratio")
    ax.legend(loc="upper left", labelcolor=t["ink2"])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.set_title("The attenuation depth trend is recovered", pad=18)
    subtitle(ax, f"r vs real NAC: {v['r_ac_vs_true_mean']:.2f} (raw AC) → "
                 f"{v['r_synth_vs_true_mean']:.2f}; MAE −{v['mae_reduction_pct']:.1f}% (8 subjects)", t)
    fig.text(0.01, 0.03, f"BIC-MAC {subject}, coronal; each image scaled to in-body mean 1, shared window",
             color=t["muted"], fontsize=8.5)
    fig.subplots_adjust(left=0.01, right=0.95, top=0.86, bottom=0.19)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--charts", action="store_true", help="skip the data-dependent panel")
    ap.add_argument("--data-root", type=Path,
                    default=Path(os.environ.get("BIC_MAC_DATA_ROOT", REPO / "data")))
    a = ap.parse_args()
    ASSETS.mkdir(exist_ok=True)
    for name, t in THEMES.items():
        style(t)
        leaderboard(t, ASSETS / f"leaderboard_{name}.png")
        training(t, ASSETS / f"training_{name}.png")
        if not a.charts:
            pseudo_nac(t, ASSETS / f"pseudo_nac_{name}.png", a.data_root)
        print(f"wrote {name} figures -> {ASSETS}")


if __name__ == "__main__":
    main()
