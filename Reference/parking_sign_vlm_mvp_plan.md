# MVP Plan — Can VLMs Read an LA Parking Sign?
**Goal:** a sendable result (repo + figure + 3-page writeup) for the UCLA Mobility Lab

**[VERIFY]** marks assumptions to confirm rather than trust.

*Updated 2026-09-21: phases 0–4 are done and this plan now records what was actually built; phases 5–7 are the remaining work. Numbers below are measured unless marked as estimates.*

---

## Status

| Phase | State | Result |
|---|---|---|
| 0 — go/no-go | ✅ done | Mapillary viable, but only with pre-screening; random street images almost never show a usable sign |
| 1 — collection | ✅ done | 104,230 images screened → 5,218 sign crops → 2,072 OCR-confirmed candidates |
| 2 — schema, guide, evaluator | ✅ done | `schema.py`, `ANNOTATION_GUIDE.md`, `evaluator.py`, 37 evaluator tests |
| 3 — labeling | ✅ done | 692 triaged accepts → **272 labeled signs**, 420 rejected (289 for unreadable fine print) |
| 4 — query generation | ✅ done | **4,021 queries**, 44.1% legal |
| 5 — inference | ⏳ next | Harness written and dry-run tested; needs a GPU |
| 6 — analysis | ⏳ | Not started |
| 7 — writeup and outreach | ⏳ | Not started |

---

## 1. MVP definition

### The question

> Given a photo of a real LA parking sign and a query ("Tuesday 11:30am, no permit, 45 minutes — legal?"), can open-weights VLMs give the right verdict, and when they fail, is it because they couldn't *read* the sign or couldn't *reason* about it?

### In scope (as built)

- **272 LA signs** from Mapillary (target was 500), structured transcriptions, a rule evaluator
- **4,021 (sign, query) examples** (target was ~7,500)
- 3 models: Qwen3-VL-8B, InternVL3.5-8B, MiniCPM-V-4.5 **[VERIFY model ids and vLLM support]**
- 2 conditions: **A** (sign crop + query) and **B** (ground-truth transcription + query)
- Metrics: verdict accuracy stratified by panel count, rule attribution, bootstrap CIs over signs

### Why 272 rather than 500

Street-level imagery rarely resolves the fine print on LA signs. Of 692 human-accepted candidates, 420 were rejected at labeling, 289 of them because text that changes the rule (column headers such as "MON–SAT / SUNDAY", "EXCEPT SUNDAY" lines) could not be read even in the original photo. This is a finding in its own right, not only a shortfall: it bounds what any model could do from this imagery.

### Explicitly deferred

Fine-tuning, condition C (transcribe-then-reason), OCR baseline, prompt primer, calibration analysis, 30B+ models, cross-city comparison, own-phone photos to extend the multi-panel tail, a context-crop variant of condition A.

### Success criteria

1. A defensible labeled dataset with its selection rules documented ✅
2. An accuracy-vs-complexity figure with the A/B decomposition ⏳
3. A repo that reproduces both with one command ⏳

**It does not need the models to fail interestingly.** If they score 95%, that's a finding.

---

## 2. Phase 0 — go/no-go (done)

Built `scripts/phase0_survey.py` and `scripts/phase0_probe.py`, reviewed in ``scripts/phase0_probe.py``.

**What it showed**
- Mapillary has dense LA coverage (~175k images across the five target areas) but it is mostly 2016–2019 dashcam footage; ~14% of images sit within 30 m of a freeway.
- **A random sample is the wrong sample.** In 100 random Westwood images, half the sign detections were under 22 px tall — far too small to read.
- Mapillary's *map feature* layer is useless for finding parking signs (about 27 parking-classified signs in all of Westwood), but its *per-image* detections are good for locating sign plates.

**Consequence:** a pre-screen was added before any human review — minimum resolution, off-freeway, and at least one large, upright, uncut, non-overhead sign detection.

---

## 3. Phase 1 — Collection (done)

`src/vlm_parking/{mapillary,prescreen,collect}.py`, driven by `scripts/collect.py` (five resumable stages).

| Stage | Method | Result |
|---|---|---|
| Search | Graph API bbox search, tiled at 0.005°, tiles split on result cap or server errors | ~175k image records in 5 areas |
| Pre-screen | ≥1920 px long side; >30 m from a freeway (OSM via Overpass, cached); sign detection ≥60 px on the 2048 scale, portrait, uncut, not overhead | 104,230 screened → 19,321 passing |
| Crop | Merge detections that stack vertically on one pole into one crop; cut from the 2048-px image | 22,239 stacks |
| OCR | RapidOCR (PP-OCR models) on each crop, **as a reject-only filter** | 3,619 of 22,239 stacks dropped as clearly non-parking |
| Dedup | Cluster views of one physical sign by camera position (≤25 m), perceptual hash and OCR-text overlap; keep the sharpest | 5,218 unique candidates |
| Full-res | Re-cut the chosen view from the original photo | 2,072 OCR-confirmed crops |

**Design decisions that differ from the original plan**
- **OSMnx point sampling was replaced** by bbox search plus a freeway filter: simpler, and it needs no street network.
- **OCR never gates a sign *in*.** In the gold sample, OCR recognised only 38 of 90 usable Westwood signs; gating on it would drop hard-to-read signs and shrink the very gap the study measures. From triage round 2 the *display* was limited to OCR-confirmed crops for throughput, and the accepted cohort is defined as triage-accepted **and** OCR-confirmed (`src/vlm_parking/dataset.py`). The writeup must state this, because it makes condition A somewhat easier than for LA signs in general.
- **Crops are tight on the sign** (≈12% padding), not whole street views: the ground truth is per sign, and a downscaled full frame would make the sign unreadable. Scope limitation to report.

---

## 4. Phase 2 — Schema, guide, evaluator (done)

- **`src/vlm_parking/schema.py`** — `Panel`, `Sign`, `ImageQuality`, `Query`, `Verdict`, with validation (a time limit needs minutes, a permit-only panel needs a district, times are HH:MM, panel orders are 0..n-1).
- **`ANNOTATION_GUIDE.md`** — reject reasons, field rules, five worked LA examples, and the evaluator's semantics in plain language.
- **`src/vlm_parking/evaluator.py`** — `evaluate(sign, query) -> Verdict` on a weekly minute timeline.
- **`tests/test_evaluator.py`** — 37 tests: every rule type, overnight wraparound, boundary minutes, day-of-week edges, permits, durations, multi-panel priority, ambiguity, schema validation, and the guide's examples.

**Semantics agreed while building it**
1. A query is a *stay*: the whole interval must be legal, so a stay that runs into a restriction is illegal.
2. Windows are half-open: in force at the start minute, over at the end minute.
3. Time limits count only time spent inside the window; meters are assumed paid.
4. `district` on a non-permit panel means permit holders are exempt from that panel; exemption plates never cover street sweeping.
5. Queries never fall on holidays, so `except_holidays` never changes an answer.
6. Most restrictive panel governs: no stopping > no standing > no parking > loading > permit-only > time limit.
7. `ambiguous` covers unclear wording as well as conflicting panels; ambiguous signs generate no queries.

---

## 5. Phase 3 — Labeling (done)

Built `scripts/triage.py` (fast accept/reject of crops) and `scripts/label.py` (transcription, blind, with an inline evaluator check). Both are local browser tools rather than Streamlit, for keyboard control and a dynamic panel list.

| Step | Result |
|---|---|
| Triage | 5,218 candidates reviewed → 828 accepted (OCR-confirmed cohort: 692) |
| Labeling | 692 → **272 labeled**, 420 rejected |
| Effort | median 9 s per labeled sign, 2 s per reject; ~2.4 h total |

**Reject reasons:** illegible fine print 289, illegible panel 75, not a parking sign 30, multiple signs 15, unsupported condition 6, cut off 5.

**Deviations from the original plan**
- **No model-assisted pre-fill and no labeler-calibration gate.** Manual labeling turned out to take ~9 s per sign, so a frontier-model pre-fill would have saved little and would have put model output into the ground truth. Every label is `gold_manual`.
- **The gold set is the first 100 signs** (stratified by detected panel count, seeded draw); 72 remain after the cohort rule. Everything after it was labeled the same way.
- **Self-consistency re-label** (50 signs, blind, `--round 2`) is still to do, and now replaces inter-annotator agreement as the dataset's quality number. Draw it *excluding* the signs on `data/labels/audit_worklist.csv`: re-labeling a sign just corrected would measure agreement with the correction, not consistency.
- **`tow_away` is unannotated in v1** — `false` on all 272 signs, including the 57+ that print "TOW-AWAY". Not backfilled: the evaluator never reads the field, so nothing measured depends on it, and its uniform absence can't bias the A−B gap. Documented in ANNOTATION_GUIDE.md §3 and to be stated in the writeup.
- **OCR label audit** (`scripts/audit_labels.py`, local, CPU). Compares the words RapidOCR reads off each crop against the rules in the label. Found 16 signs (231 queries, 5.7%) where the sign says NO STOPPING but the label carries only `no_parking`; several also drop the no-stopping panels entirely. Verdicts are unaffected (both rules are in `PROHIBITIONS`) but `SEVERITY` differs, so `governing_panel` can change on the 15 queries where two panels are in play. Corrections go through `scripts/label.py --only`. Checks for day exceptions, time limits and times were tried and dropped as pure OCR noise; the script's docstring records why, so they don't get re-added.

**Dataset as built**

| | Signs | Note |
|---|---|---|
| 1 panel | 174 | |
| 2 panels | 61 | |
| 3+ panels | 37 | main figure bins as 1 / 2 / 3+ |
| Total | **272** | 420 panels; rules: no_parking 233, time_limited 114, no_stopping 67, permit_only 6 |

Areas: Hollywood 108, Koreatown 67, Venice/Mar Vista 52, Downtown 30, Westwood 15. Capture years 2013–2026, mostly 2016–2017 and 2020–2021. Sign height in the original photo: median 242 px (IQR 173–366).

**Coverage gap:** no metered, no-standing, loading or unrestricted panels survived, so the study covers the four common LA rule types only.

---

## 6. Phase 4 — Query generation (done)

`src/vlm_parking/queries.py` + `scripts/make_queries.py`. Types are defined by how stable the verdict is, which works for every rule type:

| Type | Definition | Count |
|---|---|---|
| clear_legal | verdict unchanged under ±30 min of arrival/duration | 1,040 |
| clear_illegal | same, illegal | 1,545 |
| boundary | verdict flips within ±5 min | 591 |
| permit | same stay with/without the sign's district permit changes the answer | 26 (13 signs) |
| permit_irrelevant | same stay with/without a permit the sign doesn't mention; the answer must not change | 820 |

**4,021 queries, 44.1% legal** (majority-class baseline 55.9%). Only 13 signs post a permit district, so the permit-distractor type was added to keep 15 queries per sign and to probe models inventing exemptions.

---

## 7. Phase 5 — Inference (next, ~$5–8)

### Harness (built)

`src/vlm_parking/prompts.py` + `scripts/run_inference.py`:
- **One prompt** for both conditions; only the sign representation differs (image vs. rendered transcription). No LA-convention primer.
- **Answer schema** `reason` → `governing_panel` → `verdict`, enforced by guided JSON decoding.
- **temperature 0**, fixed seed, `max_tokens 200`.
- **Fixed image policy:** longest side ≤1024 px, JPEG q95, no upscaling; the image goes first in the prompt so vLLM's prefix cache is reused across a sign's ~15 queries.
- **Caching:** every response keyed on (model, condition, image/sign hash, prompt, sampling params) in SQLite; results stream to JSONL. Re-runs resume for free.
- **`--dry-run`** builds every prompt with no GPU.

### Hardware

**A100 40 GB (SXM4)**, not the A10 24 GB: roughly 2–2.5× faster, and InternVL's tiling (~3k visual tokens per image) leaves little KV cache on 24 GB. Estimated 3–4 hours for all six runs, ~$5–8 total. A 100 GB persistent volume holds the three models' weights; delete it when done.

### Run order

1. Smoke test: `--limit 20`, one model, condition B, then condition A. Check the answers parse and look sane.
2. Condition B for all three models (fast, text only).
3. Condition A for all three models.
4. `runpodctl stop pod` at the end of the run script.

### Gates

- Parse failure rate < 1% (guided decoding should make it ~0).
- Same `--max-side` for every model; if memory forces a change, change it everywhere and re-run.
- Record vLLM version, model revision, GPU type with the results.

---

## 8. Phase 6 — Analysis (to build)

- **Metrics:** verdict accuracy overall and by panel count (1 / 2 / 3+), query type, area, sign size, capture year; rule attribution (right verdict, wrong governing panel); parse failure rate.
- **A−B gap** per model: the perception cost, the headline number.
- **Permit distractors:** does mentioning an irrelevant permit change the verdict? (It must not.)
- **Statistics:** bootstrap 95% CIs **resampling over signs**, not queries; report n per stratum; suppress cells with n < 20; report the base rate (44.1% legal).
- **Main figure:** x = panel count, y = accuracy, two lines per model (A solid, B dashed), error bars. The vertical gap is the result.
- **Error review:** 50 condition-A failures tagged by hand (misread digit, AM/PM flip, dropped panel, wrong day range, boundary error, ignored exception, hallucinated rule, invented permit exemption).

## 9. Phase 7 — Writeup and outreach

Unchanged from the original plan, with these additions to the limitations section: 272 signs rather than 500; selection is OCR-confirmed then human-verified; crops are sign-tight rather than full street views; four rule types only; and the 42% fine-print rejection rate as a finding about street-level imagery.

---

## 10. Tech stack (as built)

| Layer | Choice |
|---|---|
| Python | 3.11, managed with `uv` |
| Image acquisition | Mapillary Graph API (`requests`); OpenStreetMap Overpass for freeway geometry |
| Image handling | Pillow, `imagehash` for dedup |
| OCR | RapidOCR (PP-OCR ONNX models) — reject-only filter |
| Schema / validation | Pydantic v2 |
| Triage / labeling UI | Local `http.server` + vanilla JS (not Streamlit) |
| Storage | JSONL + CSV for the pipeline, SQLite for labels and the response cache, Parquet for the query set |
| Inference | vLLM with guided JSON decoding |
| Models | Qwen3-VL-8B, InternVL3.5-8B, MiniCPM-V-4.5 **[VERIFY]** |
| GPU | RunPod A100 40 GB SXM4 |
| Analysis | pandas, numpy, matplotlib |
| Testing | pytest (62 tests) |
| Repo | git + GitHub; images gitignored, labels and query set tracked |

---

## 11. Remaining schedule

| Day | Hours | Task | Gate |
|---|---|---|---|
| 1 | 1 | Verify model ids; RunPod pod setup; smoke test 20 examples | Answers parse; verdicts look sane |
| 2 | 3 | Full 3×2 matrix; stop the pod | Parse failure < 1%; raw results in hand |
| 3 | 3 | Metrics, bootstrap, main figure | Figure reproduces from one command |
| 4 | 3 | Error taxonomy on 50 failures; self-consistency re-label of 50 signs | Both numbers in the README |
| 5 | 3 | Writeup + send the email | Sent |

---

## 12. Gates and risks (updated)

| Gate | When | Fail action |
|---|---|---|
| Parse failure rate < 1% | Smoke test | Tighten guided decoding before the full run |
| Condition B accuracy ≫ chance | After B runs | If B is near the base rate, the task is broken, not the perception |
| A−B gap interpretable | After A runs | Check the error taxonomy before claiming a perception cost |
| Each panel-count bin n ≥ 20 signs | Analysis | Merge bins (1 / 2 / 3+ already satisfies this) |

| Risk | Status / mitigation |
|---|---|
| Too few signs | Realised: 272. Mitigations if needed: own phone photos (also fixes the thin 3+ tail), more neighbourhoods |
| Selection bias from OCR filtering | Realised and documented; report condition A accuracy split by OCR status |
| Models score > 95% | Report it; push complexity in the extension |
| GPU bill creeping | Cache everything; stop the pod in the run script; delete the volume |
| Ground truth wrong | 37 evaluator tests; inline query checks during labeling; self-consistency re-label still to do |
