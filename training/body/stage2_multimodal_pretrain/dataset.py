"""Corpus access for stage-2 pretraining. Reads the manifest, nothing else."""

from __future__ import annotations

import json
from pathlib import Path


def load_corpus(manifest_path) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


def get_dataset(manifest_path, split: str | None = None,
                datasets: list[str] | None = None) -> list[dict]:
    corpus = load_corpus(manifest_path)
    cases = corpus["cases"]
    if split is not None:
        cases = [c for c in cases if c.get("split") == split]
    if datasets is not None:
        cases = [c for c in cases if c["dataset"] in datasets]
    for c in cases:
        for key in ("ct", "body_mask", "loss_mask"):
            if not Path(c[key]).is_file():
                raise FileNotFoundError(
                    f"{c['case_id']}: missing {key} at {c[key]} — "
                    "rerun prepare_corpus.py"
                )
    return cases


def corpus_summary(cases: list[dict]) -> str:
    by: dict = {}
    subjects: dict = {}
    for c in cases:
        by[c["dataset"]] = by.get(c["dataset"], 0) + 1
        subjects.setdefault(c["dataset"], set()).add(c["subject"])
    parts = [
        f"{n} {by[n]} scans / {len(subjects[n])} subjects"
        for n in sorted(by)
    ]
    full = sum(1 for c in cases if len(c["available_channels"]) == 4)
    return (f"{len(cases)} scans — " + ", ".join(parts)
            + f" — {full} with all four channels, {len(cases) - full} PET-only")
