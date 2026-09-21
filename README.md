# Can VLMs Read an LA Parking Sign?

A small benchmark testing whether open-weights vision-language models can decide if parking is legal from a photo of a real Los Angeles parking sign, and whether their failures come from *reading* the sign or *reasoning* about it.

See [Reference/parking_sign_vlm_mvp_plan.md](Reference/parking_sign_vlm_mvp_plan.md) for the full plan and [notebooks/00_concepts.ipynb](notebooks/00_concepts.ipynb) for background.

## Setup

```bash
uv sync
cp .env.example .env   # then add your MAPILLARY_TOKEN
```

## Phase 0 — day-1 go/no-go

```bash
uv run python scripts/phase0_probe.py
```

Then review the sample in `scripts/phase0_survey.py` / `scripts/phase0_probe.py`.

Results: *pending*

## Decisions and deviations from the plan

| Decision | Why |
|---|---|
| Pre-screen on Mapillary's sign detections (size, shape, position) and drop freeway images before downloading anything | A random sample of street images almost never contains a usable sign |
| Triage round 1 showed all crops except those OCR clearly read as non-parking signs | Avoid biasing selection toward easy-to-read signs (OCR missed 58% of usable Westwood signs) |
| From triage round 2 on, only crops whose OCR text reads as parking are triaged, and the accepted cohort is triage-accepted **and** OCR-confirmed (`src/vlm_parking/dataset.py`); the 136 OCR-unknown round-1 accepts are excluded, though `triage.csv` keeps every human decision | OCR-unknown crops were ~20% of accepts but only ~7% of them proved fully transcribable (2/28 in the gold sample). Selection is "OCR-confirmed, then human-verified", which may make condition A somewhat easier than for LA signs in general; the writeup must say so. The gold sample keeps 72 of its 100 signs |
| Every sign in the cohort is labeled by hand, blind; no model-assisted pre-fill or labeler calibration | Manual labeling takes ~23 s per labeled sign and ~3 s per reject, so assistance saves little and every label stays human |
| A stay must be legal for its whole duration; queries never fall on holidays; signs with unreadable fine print are rejected | Agreed evaluator semantics (ANNOTATION_GUIDE.md §5) |

