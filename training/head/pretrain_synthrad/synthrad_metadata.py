"""
Read the SynthRAD2025 per-region acquisition sheets and reduce each case to the
few fields that decide whether its MR is usable for pretraining a Dixon-input
model.

Each region ships `overviews/1_<REGION>_train_parameters.xlsx` with a `CT` and an
`MR` sheet, one row per patient ID. The field that matters most is the MR series
description: a large part of SynthRAD Task 1 is T1 mDIXON / MRCAT in-phase --
the same contrast family as the BIC-MAC whole-body Dixon -- and the rest is
0.35 T MR-Linac imaging, which is not.
"""

from __future__ import annotations

import glob
import os


def sequence_family(series_desc: str, manufacturer: str, field_t) -> str:
    s = str(series_desc).upper()
    m = str(manufacturer).upper()
    if "VIEWRAY" in m:
        return "MR-Linac 0.35T"
    if "DIXON" in s or "MRCAT" in s:
        return "T1 Dixon"
    try:
        if float(field_t) < 1.0:
            return "MR-Linac 0.35T"
    except (TypeError, ValueError):
        pass
    # Some centres ship rows with no SeriesDescription at all; say so rather
    # than pooling them with a named family.
    return "unspecified"


def load_metadata(root: str) -> dict:
    """pid -> metadata dict, for every region sheet under `root`."""
    import pandas as pd

    out: dict = {}
    pattern = os.path.join(root, "Task1", "*", "overviews", "*_train_parameters.xlsx")
    for p in sorted(glob.glob(pattern)):
        xl = pd.ExcelFile(p)
        mr = xl.parse("MR") if "MR" in xl.sheet_names else None
        ct = xl.parse("CT") if "CT" in xl.sheet_names else None
        if mr is None:
            continue
        ct_by_id = {}
        if ct is not None and "ID" in ct.columns:
            for _, r in ct.iterrows():
                ct_by_id[str(r["ID"])] = r
        for _, r in mr.iterrows():
            pid = str(r.get("ID", "")).strip()
            if not pid or pid == "nan":
                continue
            sd = r.get("SeriesDescription", "")
            man = r.get("Manufacturer", "")
            fs = r.get("MagneticFieldStrength", "")
            rec = {
                "series_description": str(sd),
                "mr_manufacturer": str(man),
                "mr_model": str(r.get("ManufacturerModelName", "")),
                "field_strength_T": _num(fs),
                "sequence_family": sequence_family(sd, man, fs),
                "contrast_agent": "gd" in str(sd).lower(),
                "echo_time_ms": _num(r.get("EchoTime")),
                "repetition_time_ms": _num(r.get("RepetitionTime")),
                "patient_sex": str(r.get("PatientSex", "")),
                "patient_age": str(r.get("PatientAge", "")),
            }
            c = ct_by_id.get(pid)
            if c is not None:
                rec["ct_kvp"] = _num(c.get("KVP"))
                rec["ct_manufacturer"] = str(c.get("Manufacturer", ""))
                rec["ct_slice_thickness_mm"] = _num(c.get("SliceThickness"))
            out[pid] = rec
    return out


def _num(v):
    try:
        f = float(v)
        return None if f != f else round(f, 4)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    import sys
    from collections import Counter

    md = load_metadata(sys.argv[1])
    print(len(md), "cases with metadata")
    print(Counter(v["sequence_family"] for v in md.values()))
    print(Counter(v["field_strength_T"] for v in md.values()).most_common(6))
