# MVP Plan — Can VLMs Read an LA Parking Sign?

**Owner:** Daniel
**Scope:** 500 LA signs, 2–3 open-weights VLMs, 2 conditions
**Budget:** ~10–12 hrs/week, 3–4 weeks; GPU rental ~$20–40 total
**Goal:** a sendable result (repo + figure + 3-page writeup) for the UCLA Mobility Lab

**[VERIFY]** marks assumptions to confirm rather than trust.

---

## 1. MVP definition

### The question

> Given a photo of a real LA parking sign and a query ("Tuesday 11:30am, no permit, 45 minutes — legal?"), can open-weights VLMs give the right verdict, and when they fail, is it because they couldn't *read* the sign or couldn't *reason* about it?

### In scope

- 500 LA signs from Mapillary, structured transcriptions, a rule evaluator
- ~7,500 (sign, query) examples
- 3 models: Qwen3-VL-8B, InternVL3.5-8B, MiniCPM-V-4.5
- 2 conditions: **A** (image + query) and **B** (ground-truth text + query)
- Metrics: verdict accuracy stratified by panel count, rule attribution, bootstrap CIs

### Explicitly deferred

Fine-tuning, condition C (transcribe-then-reason), OCR baseline, prompt primer, calibration analysis, 30B+ models, cross-city comparison. All are week 5+ extensions.

### Success criteria

The MVP succeeds if it produces:
1. A defensible labeled dataset with a measured agreement rate
2. An accuracy-vs-complexity figure with the A/B decomposition
3. A repo that reproduces both with one command

**It does not need the models to fail interestingly.** If they score 95%, that's a finding; push complexity in the extension.

---

## 2. Phase 0 — Day 1 go/no-go (3 hrs)

**Do this before writing any other code.**

1. **Get a Mapillary API token.** Register a free developer account; create an app; store the token in `.env`.
2. **Query one Westwood bounding box.** Roughly `lon -118.455 to -118.435, lat 34.055 to 34.075` **[VERIFY]** against a map. Use the images endpoint on `graph.mapillary.com` with a `bbox` parameter, requesting `id, thumb_2048_url, captured_at, compass_angle, is_pano, geometry` **[VERIFY exact field names in current API docs]**.
3. **Pull 100 images. Look at them.** Count how many contain a legible parking sign.
4. **Check the sign-detection layer.** Query map features in the same bbox and print the distinct `object_value` strings. See whether anything parking-related appears and whether the detections are useful for cropping.
5. **Try transcribing 5 signs by hand.** Time yourself.

### Gate

| Observation | Action |
|---|---|
| ≥ 10 legible parking signs per 100 images | **Proceed.** 500 signs is achievable. |
| 3–10 per 100 | Proceed, but widen to more neighborhoods and plan ~3× the query volume. |
| < 3 per 100 | **Stop and reconsider.** Try Koreatown/Downtown first; if still sparse, revisit the Kaggle SF option and reframe. |

Write the day-1 numbers into the README. They justify every later design choice.

---

## 3. Phase 1 — Collection (6–8 hrs, mostly unattended)

### Target areas (LA city, permit-heavy and dense)

| Area | Why |
|---|---|
| Westwood / Westwood Village | Your home turf; permit districts; you can sanity-check anything |
| Koreatown | Dense, heavy permit and street-cleaning signage |
| Downtown (Historic Core, Arts District) | Metered and loading-zone variety |
| Hollywood / East Hollywood | Permit districts, mixed commercial |
| Venice / Mar Vista | Overnight and oversized-vehicle restrictions |

Aim for roughly 100–150 signs per area so no single neighborhood's conventions dominate.

### Funnel

| Stage | Method | Output |
|---|---|---|
| 1. Sample points | Load OSM drivable street geometry with OSMnx; place query points every ~100 m | ~2,000 points |
| 2. Query images | Mapillary images endpoint, small bbox per point, recent captures preferred | ~10,000–15,000 image records |
| 3. Cheap filter | Drop low-resolution, very old captures; keep a mix of `is_pano` true/false | ~6,000 |
| 4. Download | Fetch `thumb_2048_url`; store with metadata | ~6,000 images |
| 5. OCR screen | PaddleOCR; keep images containing parking vocabulary | ~1,200 |
| 6. Crop | Crop around the OCR text region (or the Mapillary detection box, if usable) with padding | ~1,200 crops |
| 7. Dedup | Perceptual hash (`imagehash`) + geographic proximity; collapse near-identical signs on the same block | ~700–900 |
| 8. Queue for labeling | Rank by OCR confidence and text volume so the best candidates surface first | 500 labeled |

**OCR keyword list (v1):** `NO PARKING, NO STOPPING, NO STANDING, TOW AWAY, PERMIT, DISTRICT, STREET CLEANING, STREET SWEEPING, HOUR PARKING, MINUTE, METER, PASSENGER LOADING, LOADING ZONE, EXCEPT, VIOLATORS, AM, PM, MON, TUE, WED, THU, FRI, SAT, SUN`

### Notes

- **Run collection as a background job** across a few days; it's API-rate-bound, not compute-bound. Start it on day 2 and let it run while you build the evaluator.
- **Respect rate limits**; add retry with backoff; checkpoint progress so a crash doesn't restart the crawl.
- **Record per image:** `image_id`, `creator_username`, `captured_at`, `lat`, `lon`, `compass_angle`, `is_pano`, source URL, and CC BY-SA attribution. This goes in `data/attribution.csv`.
- **Store crops, not full images**, for the labeling queue — faster to review.

---

## 4. Phase 2 — Schema, guide, evaluator (5–7 hrs)

Do this **before** labeling. Every hour here saves three during labeling.

### 4.1 Schema (Pydantic)

```python
class Panel(BaseModel):
    order: int                      # top to bottom, 0-indexed
    rule: Literal["no_parking","no_stopping","no_standing","permit_only",
                  "time_limited","metered","passenger_loading","tow_away",
                  "unrestricted"]
    days: list[Day]                 # explicit list, never a range string
    start: str | None               # "HH:MM", 24h
    end: str | None                 # may wrap midnight
    limit_min: int | None           # for time_limited
    district: str | None            # for permit_only
    except_holidays: bool = False
    tow_away: bool = False

class Sign(BaseModel):
    sign_id: str
    source: Literal["mapillary"]
    panels: list[Panel]
    arrow: Literal["left","right","both","none"] | None
    n_panels: int
    image_quality: ImageQuality       # angle, glare, occlusion, legible
    ambiguous: bool = False
    provenance: Literal["gold_manual","assisted_accepted","assisted_corrected"]
    notes: str = ""
```

**Keep it this small.** Every extra field costs seconds × 500.

### 4.2 Annotation guide (`ANNOTATION_GUIDE.md`)

Decide these once, write them down, never deviate:

| Case | Rule |
|---|---|
| Overnight window ("6 PM to 8 AM") | One panel, `start > end` means wraparound. Don't split. |
| No days listed | Record as all seven days. Note the assumption in the guide. |
| "EXCEPT SUNDAY" | Enumerate the included days explicitly. |
| "2 HOUR PARKING" with no hours | `limit_min: 120`, `start`/`end` null (applies all day). |
| Holidays | `except_holidays: true`. Don't enumerate dates. |
| Tow-away warning attached to a rule | `tow_away: true` on that panel, not a separate panel. |
| One panel illegible | **Reject the whole sign.** Log the reason. |
| Panels genuinely conflict | `ambiguous: true`. Still transcribe both. |
| Arrow present | Record it; do not use it to modify the rules. |
| Multiple signs on one pole, clearly one stack | One `Sign` with multiple panels. |
| Two unrelated signs in one crop | Re-crop, or reject. |

This document doubles as the pre-fill prompt for the labeler model.

### 4.3 The evaluator

```python
def evaluate(sign: Sign, query: Query) -> Verdict:
    """Returns (legal|illegal|ambiguous, governing_panel_order, reason)."""
```

**Priority logic:** the most restrictive applicable panel governs. `no_stopping` > `no_parking` > `permit_only` (without permit) > `time_limited` (if duration exceeds limit) > legal.

**Tests are mandatory** (`tests/test_evaluator.py`), covering at minimum:
- Each rule type in isolation
- Overnight wraparound (query at 23:00 and 02:00 against an 18:00–08:00 window)
- Boundary minutes (window 10:00–12:00, queries at 09:59, 10:00, 11:59, 12:00, 12:01)
- Day-of-week edges (Sunday query against "MON-SAT")
- Permit present vs. absent against the same sign
- Duration exceeding vs. fitting a time limit
- Multi-panel priority conflicts
- Ambiguous signs returning `ambiguous`

**Then hand-check 30 real cases** against the code. If any disagree, fix before proceeding. A silent bug here invalidates every number downstream.

---

## 5. Phase 3 — Labeling 500 signs (12–16 hrs)

### 5.1 Review UI (3–4 hrs to build)

Streamlit app, one sign per screen:
- **Left:** the crop, zoomable, plus a link to the full image
- **Right:** editable spec fields, pre-filled
- **Keyboard:** `a` accept, `r` reject (with reason dropdown), `f` flag ambiguous, `n` next
- Writes to SQLite with `provenance` and a timestamp per sign
- Shows a running counter and per-sign timing

Timing per sign is free data — include it in the writeup.

### 5.2 Gold set: 100 signs, blind (5–6 hrs)

- Sample **stratified by panel count** so complex signs are represented, not just the easy majority.
- Label with the pre-fill **disabled**. You must not see model output.
- Go slowly; this is the reference standard.

### 5.3 Calibrate the labeler (2 hrs)

1. Run a frontier API model (never one you're evaluating) over the same 100 gold signs, with the annotation guide as the system prompt and guided JSON output.
2. Compare **field by field**: rule type, days, start, end, limit_min, district, n_panels.
3. Read every disagreement. Where the guide was ambiguous, fix the guide and re-run.

**Gate: ≥95% agreement on every field.** If a field stays below, transcribe that field manually throughout and note it.

**Cost:** 100 images through a frontier model is a couple of dollars.

### 5.4 Assisted labeling: 400 signs (5–7 hrs)

- Pre-fill each, verify each, accept or correct.
- Work in 45-minute sessions; accuracy drops after that.
- **Re-label 50 signs a week later, blind**, to measure your own self-consistency. Report it.

### 5.5 Final dataset checks

- Panel-count distribution: you want a usable tail of 3+ panel signs, not 95% single-panel. If the tail is thin, target additional collection at commercial blocks.
- Rule-type coverage: every rule in the vocabulary should appear at least ~15 times.
- Rejection log: count and reasons, reported in the writeup.

---

## 6. Phase 4 — Query generation (1–2 hrs)

**15 queries per sign → 7,500 examples.** Per sign:

| Type | Count | Definition |
|---|---|---|
| Clearly legal | 5 | ≥30 min from any rule boundary |
| Clearly illegal | 5 | ≥30 min inside a restriction |
| Boundary | 3 | Within 5 min of a window edge |
| Permit-dependent | 2 | Same time, with and without permit |

Each query carries `time` (day + HH:MM), `has_permit` (district or none), and `duration_min`.

Seed the generator; save the query set as a versioned Parquet file. Balance the overall legal/illegal split so accuracy isn't inflated by a lopsided prior — report the base rate either way.

---

## 7. Phase 5 — Inference (6–8 hrs, ~$20–40)

### 7.1 Don't rent an A100

For 8B models, a **48GB L40S or a 24GB RTX 4090** is plenty and costs a fraction of an A100. Save the A100 for the fine-tuning stretch.

| GPU | Typical rate **[VERIFY]** | Fits |
|---|---|---|
| RTX 4090 (24GB) | ~$0.35–0.70/hr | 8B bf16, modest batch |
| L40S (48GB) | ~$0.80–1.10/hr | 8B comfortably, large batch |
| A100 80GB | ~$1.30–2.00/hr | Overkill for the MVP |

**Recommendation: one L40S on RunPod community cloud.** Expect 6–10 GPU-hours total → **$8–20**, with slack to $40.

### 7.2 Pod setup

1. Template with PyTorch + CUDA preinstalled.
2. **Persistent volume (~100GB)** for model weights, so you download once. Delete it when done — it bills while stopped.
3. `pip install vllm`, then pull weights with `huggingface-cli download`.
4. Upload `data/` (crops + labels + queries) via `runpodctl` or S3.
5. Verify GPU and model load with a 5-example smoke test before anything else.

### 7.3 Inference harness

- **Serve with vLLM**, batched, `temperature=0`, fixed seed.
- **Guided JSON decoding** against a Pydantic response schema. Never parse free text.
- **Cache** keyed on `(model_id, image_sha256, prompt_sha256, temperature)` in SQLite. Resume is then free.
- **Write results incrementally** (JSONL append), never only at the end.
- **Resize crops** to a sensible max dimension and log tokens-per-image; high-resolution tiling can multiply cost several-fold.
- At the end of the run script: `runpodctl stop pod` so a finished job doesn't bill overnight.

### 7.4 Run matrix

| Model | Condition A (image) | Condition B (text) |
|---|---|---|
| Qwen3-VL-8B | 7,500 | 7,500 |
| InternVL3.5-8B | 7,500 | 7,500 |
| MiniCPM-V-4.5 | 7,500 | 7,500 |

**Order of operations:** get the full pipeline working on Colab free tier with 20 examples first. Rent only once it runs clean end to end.

**Prompt (both conditions), fixed across models:** task statement, the query, and an output schema instruction. No LA-convention primer — that's the extension's variable.

---

## 8. Phase 6 — Analysis (5–6 hrs)

### Metrics

| Metric | Detail |
|---|---|
| Verdict accuracy | Overall, and by panel count (1, 2, 3, 4+) |
| Rule attribution | Did it name the correct governing panel? Track right-verdict-wrong-reason separately |
| Query-type breakdown | Clearly legal / clearly illegal / boundary / permit |
| Image-quality breakdown | Angle, glare, `is_pano` |
| **A−B gap** | The perception cost, per model |
| Parse failure rate | Should be ~0 with guided decoding; report it |

### Statistics

- **Bootstrap 95% CIs resampling over signs**, not queries. The 15 queries from one sign are correlated; treating them as independent gives falsely tight intervals.
- Report n per stratum; suppress cells with n < 20.
- Include the base rate (share of legal answers) so accuracy is interpretable.

### The main figure

X-axis: panel count. Y-axis: accuracy. Two lines per model (condition A solid, condition B dashed), with error bars. **The vertical gap between solid and dashed is the headline result.**

### Error review

Read 50 condition-A failures by hand and tag them: misread digit, AM/PM flip, dropped panel, wrong day range, boundary error, ignored exception, hallucinated rule. A counts table of these is often the most cited part of a benchmark paper.

---

## 9. Phase 7 — Writeup and outreach (5–6 hrs)

**Report (3 pages):** question; why it matters (VLMs as the reasoning layer for driving; existing parking-sign work is detection + OCR only); data and labeling protocol with the agreement rate; method (conditions A/B); the main figure; the error taxonomy; limitations (500 signs, one city, single-shot).

**Repo:** README with the day-1 numbers, `ANNOTATION_GUIDE.md`, one-command reproduction, dataset under CC BY-SA with attribution, tests passing.

**Email (~150 words), to Zhiyu Huang or another driving-side author, with Prof. Ma CC'd:** one line on who you are; one line naming the specific work (AOI 1, foundation models for driving, regulation-grounded reasoning); two sentences on what you built and the single most interesting number; the repo link; one specific question about their work; a request for 20 minutes.

Send it the moment you have the figure. Don't wait for polish.

---

## 10. Tech stack

| Layer | Choice |
|---|---|
| Python | 3.11, managed with `uv` |
| Image acquisition | Mapillary Graph API (`requests`) + OSMnx for sampling points |
| Image handling | Pillow, OpenCV, `imagehash` for dedup |
| OCR | PaddleOCR (fallback EasyOCR) |
| Schema / validation | Pydantic v2 |
| Labeling UI | Streamlit |
| Storage | SQLite (labels, cache, results); Parquet for analysis |
| Pre-fill labeler | A frontier VLM API (not an evaluated model) |
| Inference | vLLM with guided JSON decoding |
| Models | Qwen3-VL-8B, InternVL3.5-8B, MiniCPM-V-4.5 |
| GPU | RunPod community L40S (or 4090) |
| Analysis | pandas, numpy, scipy |
| Plots | matplotlib |
| Testing | pytest |
| Repo | git + GitHub; images gitignored, released separately |

---

## 11. Schedule (10–12 hrs/week)

### Week 1 — data and ground truth (12 hrs)

| Day | Hours | Task | Gate |
|---|---|---|---|
| 1 | 3 | Phase 0 go/no-go | **Signs-per-100 ≥ 3** |
| 2 | 3 | Collection pipeline; start the crawl | Running unattended |
| 3 | 3 | Schema + annotation guide | Guide covers all edge cases |
| 4 | 3 | Evaluator + tests + 30 hand-checks | All tests pass |

### Week 2 — labeling (12 hrs)

| Day | Hours | Task | Gate |
|---|---|---|---|
| 5 | 3 | Review UI | Usable, keyboard-driven |
| 6 | 4 | Gold set, 100 signs, blind | Gold set complete |
| 7 | 2 | Labeler calibration | **≥95% per-field agreement** |
| 8 | 3 | Assisted labeling, ~200 signs | 300 total |

### Week 3 — labeling and inference (12 hrs)

| Day | Hours | Task | Gate |
|---|---|---|---|
| 9 | 3 | Assisted labeling, ~200 signs | **500 total** |
| 10 | 2 | Query generation + dataset checks | 7,500 examples |
| 11 | 3 | Inference harness on Colab free tier | 20-example run parses cleanly |
| 12 | 4 | Rent GPU; run the full 3×2 matrix | Raw results in hand |

### Week 4 — analysis and send (10 hrs)

| Day | Hours | Task |
|---|---|---|
| 13 | 3 | Metrics, bootstrap, main figure |
| 14 | 3 | Error taxonomy on 50 failures |
| 15 | 2 | Repo cleanup, README, reproduction check |
| 16 | 2 | Writeup + **send the email** |

**Aggressive version:** weeks 1–3 compressed to 2.5 weeks if collection yields well and labeling goes faster than 40 s/sign.

---

## 12. Gates and risks

| Gate | When | Fail action |
|---|---|---|
| Signs per 100 images ≥ 3 | Day 1 | Widen neighborhoods; if still sparse, reconsider the data source |
| Evaluator tests pass + 30 hand-checks agree | Day 4 | Stop; fix before labeling |
| Per-field agreement ≥ 95% | Day 7 | Fix the guide/prompt, or label that field manually |
| Parse failure rate < 1% | Day 11 | Tighten guided decoding before spending GPU money |
| 500 signs labeled | Day 9 | Ship with what you have; 350 is still a study |

| Risk | Mitigation |
|---|---|
| Panel-count distribution too flat (mostly 1-panel) | Target commercial blocks in collection; oversample complex signs into the gold set |
| Models score > 95% | Report it, and push complexity in the extension rather than reframing the MVP |
| Crop quality varies systematically by neighborhood | Record image quality flags; check that findings survive stratification |
| GPU bill creeping | Cache everything; stop the pod in the run script; delete the volume when done |
| Labeling fatigue | 45-minute sessions; re-label 50 later to catch drift |

---

## 13. First three actions

1. Register for a Mapillary API token.
2. Pull 100 images from a Westwood bounding box and count the legible parking signs.
3. Hand-transcribe 5 of them with a timer running.

Everything else depends on what those three tell you.
