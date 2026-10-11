"""
Corpus access for CT-only pretraining.

The pretraining task has no inputs to assemble -- there is one image per case and
it is both the source and the target -- so this module only reads the manifest
that `prepare_corpus.py` wrote and hands MONAI a list of dicts.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_corpus(manifest_path) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


def get_dataset(manifest_path, split: str | None = None,
                datasets: list[str] | None = None) -> list[dict]:
    """Return case dicts, optionally filtered by split and dataset of origin.

    Each entry carries `ct`, `body_mask`, `bone_gain`, `dataset`, `case_id`.
    `bone_gain` travels with the case because it is a property of the scanner
    protocol, not of the run -- see prepare_corpus.py.
    """
    corpus = load_corpus(manifest_path)
    cases = corpus["cases"]
    if split is not None:
        cases = [c for c in cases if c.get("split") == split]
    if datasets is not None:
        cases = [c for c in cases if c["dataset"] in datasets]
    for c in cases:
        for key in ("ct", "body_mask"):
            if not Path(c[key]).is_file():
                raise FileNotFoundError(
                    f"{c['case_id']}: missing {key} at {c[key]} — "
                    "rerun prepare_corpus.py"
                )
    return cases


def corpus_summary(cases: list[dict]) -> str:
    counts: dict[str, int] = {}
    subjects: dict[str, set] = {}
    for c in cases:
        counts[c["dataset"]] = counts.get(c["dataset"], 0) + 1
        subjects.setdefault(c["dataset"], set()).add(c["subject"])
    parts = [
        f"{name} {counts[name]} scans / {len(subjects[name])} subjects"
        for name in sorted(counts)
    ]
    return f"{len(cases)} scans total — " + ", ".join(parts)
