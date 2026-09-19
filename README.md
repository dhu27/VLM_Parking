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

Then review the sample in [notebooks/01_phase0_go_no_go.ipynb](notebooks/01_phase0_go_no_go.ipynb).

Results: *pending*
