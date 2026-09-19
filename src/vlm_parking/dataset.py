"""Which collected signs are in the dataset.

The accepted cohort is every candidate that was accepted in triage *and* whose OCR text reads as a parking
sign (ocr_status == "parking"). triage.csv is left untouched as the record of human decisions; this rule is
applied on top of it. OCR-unknown signs accepted in triage round 1 are therefore excluded (decision 2026-09-19:
only ~7% of them proved fully transcribable, and one selection rule is easier to describe than two).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLLECT = Path("data/collect")


def all_candidates(collect: Path = COLLECT) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in sorted(collect.glob("*/candidates.csv"))], ignore_index=True)


def triage_decisions(collect: Path = COLLECT) -> pd.DataFrame:
    return pd.read_csv(collect / "triage.csv")


def accepted_candidates(collect: Path = COLLECT) -> pd.DataFrame:
    cands = all_candidates(collect)
    triage = triage_decisions(collect)
    accepted = set(triage.loc[triage.decision == "accept", "candidate_id"])
    keep = cands.candidate_id.isin(accepted) & cands.ocr_status.eq("parking")
    return cands[keep].reset_index(drop=True)
