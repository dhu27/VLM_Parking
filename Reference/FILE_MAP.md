# File Map

What every file and folder in this repo is for. Folders full of one kind of file (e.g. sign images) are described once, not file by file.

**Git:** 🟢 = tracked in git · ⚪ = ignored (local only, regenerable). Under `data/`, only `data/collect/triage.csv`, `data/collect/<area>/candidates.csv`, `data/labels/` (labels database + gold set) and `data/queries/` (the query set) are committed; images are never committed.

---

## Top level

| Path | What it is |
|---|---|
| 🟢 `README.md` | Project overview, setup, and how to run Phase 0. Day-1 numbers go here. |
| 🟢 `ANNOTATION_GUIDE.md` | Rules for transcribing a sign into the schema, reject reasons, worked examples, and the evaluator's legality rules. Every sign is labeled by hand against it; there is no model pre-fill. |
| 🟢 `pyproject.toml` / `uv.lock` | Python 3.11 project and pinned dependencies (`uv sync` installs them). |
| 🟢 `.gitignore` | Keeps secrets, images, data dumps and logs out of git. |
| 🟢 `.env.example` | Template for API keys. Copy to `.env`. |
| ⚪ `.env` | Your real keys (`MAPILLARY_TOKEN`, later labeler / Hugging Face / RunPod keys). Never commit. |
| ⚪ `.venv/` | The project's virtual environment, created by `uv sync`. |

## `Reference/`

| Path | What it is |
|---|---|
| 🟢 `parking_sign_vlm_mvp_plan.md` | The full MVP plan: phases, schedule, gates, risks. |
| 🟢 `FILE_MAP.md` | This file. |

---

## `src/vlm_parking/` — the library code

| File | What it does |
|---|---|
| `mapillary.py` | Mapillary Graph API client. Searches images and map features in a bounding box (split into small tiles; tiles that hit the result cap or keep erroring are split further), retries with backoff, fetches per-image detections, downloads thumbnails, and decodes Mapillary's base64 detection outlines into pixel boxes. |
| `prescreen.py` | Cheap filters that decide which images are worth a look, before downloading anything: minimum resolution, not on or beside a freeway (OpenStreetMap freeway lines via Overpass, cached), and at least one large, upright, uncut, non-overhead sign detection. Also defines the five target `AREAS` and the sampling / spatial-dedup helpers. |
| `collect.py` | Helpers for Phase 1 collection: merging stacked panels on one pole into a single crop, the OCR rule (OCR may only *reject* crops it clearly reads as non-parking signs; it never gates signs in), and clustering views of the same physical sign across drives. |
| `queries.py` | Query generation for one sign: query types defined by how stable the verdict is (±30 min for "clear", flips within ±5 min for "boundary"), permit pairs, and distractor permit pairs for signs with no district. |
| `dataset.py` | Defines the **accepted cohort**: triage-accepted *and* OCR-confirmed parking text. Everything downstream (labeling, query generation) uses this, not `triage.csv` directly. |
| `schema.py` | Pydantic models for a transcribed sign (`Sign`, `Panel`, `ImageQuality`), a query (`Query`: day, time, duration, permit), and an answer (`Verdict`). Validates transcriptions. |
| `evaluator.py` | The ground-truth evaluator: `evaluate(sign, query) -> Verdict`. Decides legal / illegal / ambiguous and which panel governs, using the semantics in `ANNOTATION_GUIDE.md` §5. |
| `__init__.py` | Empty package marker. |

## `scripts/` — things you run

| Script | What it does | Run with |
|---|---|---|
| `phase0_survey.py` | Runs the pre-screen on a sample of images in each target area, without downloading images, and prints a per-area yield table. Writes `data/phase0/survey/`. | `uv run python scripts/phase0_survey.py` |
| `phase0_probe.py` | Builds a ~100-image review sample for the Phase 0 go/no-go notebook: pre-screened (`--prescreen`) or random baseline. Downloads the images and their detections. | `uv run python scripts/phase0_probe.py --area koreatown --prescreen` |
| `collect.py` | **The Phase 1 collection pipeline.** Five resumable stages per area: `screen` (detections for up to `--n-screen` images; keep ones with a big sign) → `crop` (download 2048-px thumbnails, merge stacks, crop, OCR) → `dedup` (drop clear non-parking signs; collapse repeat views of one sign; keep your triaged crops as representatives) → `final` (full-resolution crops from the originals + `candidates.csv`; by default only OCR-confirmed or already-triaged signs, `--all-ocr` for everything) → `sheets` (contact sheets). | `uv run python scripts/collect.py --areas koreatown` |
| `label.py` | **Phase 3 labeling tool** (browser). Every sign in the accepted cohort is labeled by hand, blind: transcribe it panel by panel or reject it with a reason code, and check any query against the form with the evaluator. Queue: the stratified gold sample first, then the rest, most promising first (OCR-confirmed, more panels, larger). `--round 2` hides round-1 labels for the self-consistency re-label. | `uv run python scripts/label.py` → http://localhost:8766 |
| `make_queries.py` | **Phase 4 query generation.** 15 queries per labeled sign (clearly legal / clearly illegal / boundary / permit / permit-distractor), each with its evaluator verdict; writes `data/queries/*.parquet`. | `uv run python scripts/make_queries.py` |
| `triage.py` | Local browser tool for accept / maybe / reject triage of the collected crops (OCR-confirmed only; `--all` for everything). Saves decisions to `data/collect/triage.csv` after every page. | `uv run python scripts/triage.py` → http://localhost:8765 |

## `tests/`

| File | What it covers |
|---|---|
| `test_prescreen.py` | Bbox tiling, detection-geometry decoding orientation, sign-size/shape/position filters, freeway buffer, spatial dedup. |
| `test_collect.py` | Duplicate-box removal, stack merging, the OCR parking/other/unknown rule, duplicate-sign clustering. |
| `test_queries.py` | Query counts and types, verdicts matching the evaluator, clear queries stable under ±30 min, boundary queries flipping within ±5 min, permit pairs differing, distractor permit pairs not differing, determinism. |
| `test_evaluator.py` | Every evaluator case from the plan (each rule type, overnight windows, boundary minutes, day-of-week edges, permits, durations, multi-panel priority, ambiguity), schema validation, and a check that the guide's JSON examples validate. |

Run all with `uv run pytest`.

## `notebooks/`

| File | What it is |
|---|---|
| `00_concepts.ipynb` | Background reading: how VLMs work and extend LLMs, the perception-vs-reasoning design (conditions A and B), LA signs as a task, and inference mechanics (memory, vLLM, decoding, guided JSON, prompting). |
| `01_phase0_go_no_go.ipynb` | Phase 0 review: step through a sample with detections outlined, record legible parking signs, calibrate the size threshold, time 5 hand transcriptions, save the day-1 numbers. |
| `figures/` | SVG diagrams used by `00_concepts.ipynb` (VLM architecture, inference pipeline). |

---

## `data/` — local only, except the files marked 🟢

### `data/phase0/` — Phase 0 exploration

| Path | Contains |
|---|---|
| `cache/` | JSON: every Mapillary image record per area (`<area>_images.json`) and cached freeway lines (`freeways_*.json`). Reused by later runs so searches aren't repeated. |
| `survey/` | JSON per area: pre-screen funnel counts and the images that passed. `summary.csv` compares areas. |
| `<area>_prescreen/` | One Phase 0 review sample: `sample.csv` (image metadata + attribution), `detections.json`, `candidates.json` (which detections passed), `crops_sheet.jpg` (contact sheet of passing crops). |
| `<area>_prescreen/images/` | **Full street-level photos** (2048-px thumbnails), one JPEG per sampled image, named `<mapillary_image_id>.jpg`. |

### `data/collect/` — Phase 1 collection

| Path | Contains |
|---|---|
| 🟢 `triage.csv` | **Your triage decisions**: `candidate_id`, `accept` / `maybe` / `reject`, timestamp. The most important file here. |
| `run*.log` | Logs from collection runs. |
| `<area>/screened.jsonl` | One line per screened image: metadata, whether it passed the pre-screen, and its sign boxes. |
| `<area>/stacks.jsonl` | One line per merged sign stack: box, detected panel count, OCR text and confidence, perceptual hash. |
| `<area>/dedup.csv` | One row per unique physical sign (best view chosen, triaged crops kept), with how many views were collapsed into it. |
| 🟢 `<area>/candidates.csv` | **The candidate list for triage**: one row per unique sign with crop path, Mapillary source link, capture date, location, photographer (for CC BY-SA attribution), detected panels, and OCR status. |
| `<area>/thumb_crops/` | **Low-res parking-sign crops** (from 2048-px thumbnails), every stack including duplicates, named `<image_id>_<k>.jpg`. Working copies used for OCR and duplicate detection; safe to delete once the dataset is final. |
| `crops/<area>/` | **Full-resolution parking-sign crops**, one JPEG per unique candidate sign, named `<image_id>_<k>.jpg`. These are what you triage and what the models will see. |
| `sheets/<area>/` | **Contact sheets**: `sheet_NN.jpg` grids of candidate crops, with `sheet_NN.csv` mapping each tile number to its `candidate_id`. |

### `data/queries/` — Phase 4 query set

| Path | Contains |
|---|---|
| 🟢 `signs_v1.parquet` | One row per labeled sign: crop path, attribution (photographer, source URL, licence), location, capture date, panel count, and the sign as schema JSON. |
| 🟢 `queries_v1.parquet` | One row per query: sign, type, day, time, duration, permit, and the evaluator's verdict, governing panel and reason. |

### `data/labels/` — Phase 3 labels

| Path | Contains |
|---|---|
| 🟢 `labels.sqlite` | **All sign labels.** Table `labels`: one row per (sign, round) with status (labeled / rejected), reject reason, the sign as schema JSON, provenance, seconds spent, timestamp. Table `history`: every save, append-only. |
| 🟢 `gold_set.csv` | The first 100 signs labeled: a seeded, stratified draw from round-1 accepts (40 one-panel, 35 two-panel, 25 three-plus); 72 remain in the cohort. Every later label is also manual. |

Areas: `westwood`, `koreatown`, `downtown`, `hollywood`, `venice_mar_vista` (boxes defined in `prescreen.py`).
