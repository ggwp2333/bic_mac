"""
Build the stage-2 corpus: the same 171 scans as stage 1, but described in the
*model's own input space* rather than as bare CTs.

Stage 1 (`pretrain_v1`) was self-supervised — corrupted CT in, CT out — which is
why its stem never transferred: the layer that sees the challenge's inputs was
trained from scratch every time. Stage 2 is **supervised**: the four real input
channels in, CT out, over an enlarged corpus. QUADRA_HC can join because its
attenuation-corrected PET has been turned into NAC-PET
(`pretrain_v1/synthesize_nac.py`, r = 0.979 / MAE −49.4% against real NAC on the
8 subjects carrying both).

    BIC-MAC   75 scans   [nacpet, mri_in, mri_out, topogram]  all real
    QUADRA_HC 96 scans   [pseudo-NAC, --, --, --]             PET only

The missing QUADRA channels are not a special case to be worked around — they
are what modality dropout makes ordinary (see transforms.py).

The split is inherited from `pretrain_v1/corpus.json` verbatim so the two stages
hold out the same subjects. QUADRA's 96 scans are 48 subjects imaged twice;
splitting a test/retest pair would leak.

Usage:
    python prepare_corpus.py                 # writes corpus.json
    python prepare_corpus.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC = BASE_DIR.parent
SHARED = Path(os.environ.get("BIC_MAC_DATA_ROOT", BASE_DIR.parents[2] / "data"))  # holds bic-mac-data/, quadra-hc-data/, synthrad2025-data/

STAGE1 = SRC / "stage1_ct_pretrain"
STAGE1_CORPUS = STAGE1 / "corpus.json"
PSEUDO_NAC_DIR = STAGE1 / "outputs" / "pseudo_nac"
BICMAC_TRAIN = SHARED / "bic-mac-data" / "train"

# The channel order is the model's, and must match finetune_v1/config.yaml's
# `input_keys` exactly — this is the whole point of stage 2.
CHANNELS = ("nacpet", "mri_in_phase", "mri_out_phase", "topogram")


def bicmac_channels(sub: str) -> dict:
    f = BICMAC_TRAIN / sub / "features"
    return {
        "nacpet": str(f / "nacpet.nii.gz"),
        "mri_in_phase": str(f / "mri_combined_in_phase.nii.gz"),
        "mri_out_phase": str(f / "mri_combined_out_phase.nii.gz"),
        "topogram": str(f / "topogram.nii.gz"),
    }


def quadra_channels(case_id: str) -> dict:
    stem = Path(case_id).name                     # QUADRA_HC_001_Retest
    nac = PSEUDO_NAC_DIR / f"{stem}_pseudonac.nii.gz"
    return {
        "nacpet": str(nac) if nac.is_file() else None,
        "mri_in_phase": None,
        "mri_out_phase": None,
        "topogram": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=BASE_DIR / "corpus.json")
    ap.add_argument("--stage1-corpus", type=Path, default=STAGE1_CORPUS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.stage1_corpus.is_file():
        raise SystemExit(
            f"stage-1 corpus not found at {a.stage1_corpus} — "
            "run pretrain_v1/prepare_corpus.py first"
        )
    with open(a.stage1_corpus) as f:
        stage1 = json.load(f)

    cases, dropped = [], []
    for c in stage1["cases"]:
        entry = {
            "case_id": c["case_id"],
            "dataset": c["dataset"],
            "subject": c["subject"],
            "split": c.get("split", "train"),
            "ct": c["ct"],
            "body_mask": c["body_mask"],
            "bone_gain": c.get("bone_gain", 1.0),
        }
        if c["dataset"] == "BIC-MAC":
            sub = c["case_id"].split("/")[-1]
            entry["channels"] = bicmac_channels(sub)
            pm = BICMAC_TRAIN / sub / "ct-label" / "prediction_mask.nii.gz"
            # The downstream run scores inside prediction_mask; matching it here
            # keeps the pretraining objective pointed at the same region.
            entry["loss_mask"] = str(pm) if pm.is_file() else c["body_mask"]
        else:
            entry["channels"] = quadra_channels(c["case_id"])
            entry["loss_mask"] = c["body_mask"]

        present = [k for k, v in entry["channels"].items() if v]
        if not present:
            dropped.append(entry["case_id"])
            continue
        entry["available_channels"] = present
        cases.append(entry)

    missing_files = [
        (c["case_id"], k, v)
        for c in cases for k, v in c["channels"].items()
        if v and not Path(v).is_file()
    ]
    if missing_files:
        raise SystemExit(
            "channel files missing:\n  "
            + "\n  ".join(f"{cid}: {k} -> {v}" for cid, k, v in missing_files[:10])
        )

    counts: dict = {}
    for c in cases:
        key = (c["dataset"], c["split"], len(c["available_channels"]))
        counts[key] = counts.get(key, 0) + 1
    print("corpus:")
    for k in sorted(counts):
        print(f"  {k[0]:12s} {k[1]:5s} {k[2]} channel(s)  {counts[k]:4d} scans")
    print(f"  {'TOTAL':12s}            {len(cases):4d} scans")
    if dropped:
        print(f"dropped, no usable channel ({len(dropped)}): {dropped[:5]}"
              + (" ..." if len(dropped) > 5 else ""))
        print("  -> if these are QUADRA, pretrain_v1/synthesize_nac.py --quadra "
              "has not finished")

    if a.dry_run:
        return

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage1_corpus": str(a.stage1_corpus),
        "channel_order": list(CHANNELS),
        "pseudo_nac": {
            "dir": str(PSEUDO_NAC_DIR),
            "method": "AC x BP(exp(-Radon(mu)))/BP(1); "
                      "see pretrain_v1/synthesize_nac.py",
            "validation": "r 0.979, MAE -49.4% vs real NAC on 8 BIC-MAC subjects",
        },
        "n_cases": len(cases),
        "cases": sorted(cases, key=lambda c: c["case_id"]),
    }
    with open(a.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
