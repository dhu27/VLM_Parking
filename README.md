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

## Decisions and deviations from the plan

| Decision | Why |
|---|---|
| Pre-screen on Mapillary's sign detections (size, shape, position) and drop freeway images before downloading anything | A random sample of street images almost never contains a usable sign |
| Triage round 1 showed all crops except those OCR clearly read as non-parking signs | Avoid biasing selection toward easy-to-read signs (OCR missed 58% of usable Westwood signs) |
| From triage round 2 on, only crops whose OCR text reads as parking are triaged; signs accepted in round 1 are all kept, including OCR-unknown ones | OCR-unknown crops were ~20% of accepts but only ~7% of them proved fully transcribable (2/28 in the gold sample). Selection is therefore "OCR-confirmed or round-1 accepted, then human-verified"; condition A may be somewhat easier than for signs in general |
| A stay must be legal for its whole duration; queries never fall on holidays; signs with unreadable fine print are rejected | Agreed evaluator semantics (ANNOTATION_GUIDE.md §5) |

