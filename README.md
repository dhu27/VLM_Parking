# Can VLMs Read an LA Parking Sign?

A small benchmark testing whether open-weights vision-language models can decide if parking is legal from a photo of a real Los Angeles parking sign, and whether their failures come from *reading* the sign or *reasoning* about it.

Each model answers every question twice:

- **Condition A:** the model sees a crop of the sign.
- **Condition B:** the model gets a hand-made transcription of the sign as text.

The prompt, answer format and question are the same in both, so **A − B** is what it costs a model to work from the real sign.

## Key results so far

234 hand-labeled signs from Mapillary street imagery in five LA neighborhoods, and 3,451 questions of the form "Tuesday 11:30, park for 45 minutes, legal?". 42.6% of the questions are legal, so always answering "illegal" scores 57.4%. Every interval below is a 95% bootstrap over **signs**, not questions, because a sign's ~15 questions aren't independent.

| model | B (text) | A (image) | A − B | lost on image¹ |
|---|---|---|---|---|
| Qwen3-VL-8B-Thinking² | **98.7%** [98.1, 99.2] | **92.6%** [91.1, 94.1] | −6.1 pts [−7.6, −4.6] | **6.7%** [5.3, 8.1] |
| Qwen3-VL-8B | 91.3% [90.0, 92.5] | 82.4% [80.4, 84.3] | −8.9 pts [−10.8, −7.1] | 13.6% [11.7, 15.6] |
| InternVL3.5-8B | 78.9% [76.9, 81.0] | 77.9% [75.4, 80.5] | −0.9 pts [−3.4, +1.8] | 18.0% [15.6, 20.6] |
| MiniCPM-V-4.5 | 84.4% [82.6, 86.2] | 76.3% [74.0, 78.6] | −8.1 pts [−10.4, −5.8] | 18.1% [15.6, 20.7] |

¹ Of the questions a model got right from the text, the share it gets wrong from the image.
² Sampled at temperature 1.0, one seed so far. The other three are greedy (temperature 0), one run each.

![Accuracy by number of panels on the sign, condition A vs B, per model](reports/phase6/accuracy_by_panels.png)

1. **Every model clears the baseline** comfortably, in both conditions.
2. **Working from the image costs Qwen and MiniCPM 8–9 points.** InternVL's near-zero gap is misleading. It loses 18% of the questions it got right from the text, as many as MiniCPM does, but gains back about as many that it got wrong from the text. Part of that comes from leaning toward "illegal" only when shown an image.
3. **Complex signs are where it breaks.** For every model, the loss on the image roughly doubles to triples from single-panel signs to signs with three or more panels.
4. **Most losses aren't misreadings.** In 50 losses read by hand, about 1 in 8 was a genuine misread, such as a wrong day or a dropped panel. The rest quote the sign correctly, then misunderstand it or reason badly. The most common misunderstanding, *default-deny*, treats a sign as a list of the only hours parking is allowed. "2 HOUR PARKING 8AM–8PM" then becomes "no parking at 10 PM".
5. **Thinking fixes reasoning, not interpretation.** Qwen3-VL-8B-Thinking nearly aces the text condition and halves the loss on the image. What remains is mostly default-deny: from the image it calls a legal stay illegal 195 times, from the text 8 times. The transcription names each rule, so it does the interpreting for the model. For this model, the A − B gap is mostly interpretation, not eyesight.
6. **More of the image didn't help.** InternVL gets over 2,000 more image tokens per sign than the other two models and loses at least as much.

The full walkthrough, with every table and caveat, is [notebooks/03_results.ipynb](notebooks/03_results.ipynb). [reports/phase6/summary.md](reports/phase6/summary.md) has the tables for the three original models.

### Limits

- **Condition A is easier than LA signs in general.** The dataset keeps only signs a person could fully read and whose text OCR could confirm. 295 candidates were rejected because rule-changing fine print was unreadable even in the original photo. Treat the loss on the image as a floor.
- **Label quality has no single number yet.** The labels were corrected over several passes, but no second annotator has checked them.
- **These are three models, not a controlled comparison,** and the thinking model is a separate checkpoint from Qwen3-VL-8B, not the same model with thinking turned on.
- **The data is thin in places.** Only 43 signs have three or more panels, and only 15 are from Westwood.

## Status

| Phase | State |
|---|---|
| 0–1: find and collect signs from Mapillary | ✅ 104,230 images screened → 5,218 unique sign crops |
| 2: schema, annotation guide, rule evaluator | ✅ |
| 3: hand labeling | ✅ 234 signs used |
| 4: question generation | ✅ 3,451 questions |
| 5: inference, 3 models × 2 conditions | ✅ one A100, vLLM 0.29.0 |
| 5b: a thinking model | ⏳ seed 0 done; seeds 1–2 to run |
| 6: analysis | ✅ `scripts/analyze.py`, notebook 03 |
| 7: writeup | ⏳ |

**Next:**
- Rerun condition B with a transcription worded like the sign ("2 HOUR PARKING 8AM TO 8PM") instead of one that names each rule. This separates interpretation from reading directly.
- Run seeds 1–2 of the thinking model. First, make the prompt say what `governing_panel` should be for a legal stay: that question caused all 5 of its unfinished answers.
- Have a second annotator label about 30 signs.

## Setup

```bash
uv sync
cp .env.example .env   # then add your MAPILLARY_TOKEN
```

Inference needs a CUDA GPU. On the GPU machine, also run `pip install -r requirements-gpu.txt` to get the pinned vLLM version.

## Running it

| Step | Command |
|---|---|
| Collect sign crops | `uv run python scripts/collect.py` |
| Triage crops (browser) | `uv run python scripts/triage.py` |
| Label signs (browser) | `uv run python scripts/label.py` |
| Build the question set | `uv run python scripts/make_queries.py` |
| Run one model in one condition (GPU) | `uv run python scripts/run_inference.py --model qwen3-vl-8b --condition A` |
| Run the thinking model (GPU) | `uv run python scripts/run_inference_thinking.py --condition A --seed 0` |
| Every table and figure | `uv run python scripts/analyze.py` → `reports/phase6/` |

Every script documents its options in its docstring. The labels, question set and run outputs are committed, so `scripts/analyze.py` and notebook 03 run without a GPU or a Mapillary token. The sign images are not committed.

## Where things are

- [Reference/parking_sign_vlm_mvp_plan.md](Reference/parking_sign_vlm_mvp_plan.md): the plan, and a record of what was built in each phase.
- [Reference/FILE_MAP.md](Reference/FILE_MAP.md): what every file is for.
- [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md): how signs are transcribed, and the legality rules the evaluator applies.
- The notebooks: [00](notebooks/00_concepts.ipynb) background, [01](notebooks/01_data_collection_and_annotation.ipynb) data collection and annotation, [02](notebooks/02_inference_pipeline.ipynb) the inference pipeline, [03](notebooks/03_results.ipynb) results.

## Decisions and deviations from the plan

| Decision | Why |
|---|---|
| Pre-screen on Mapillary's sign detections (size, shape, position) and drop freeway images before downloading anything | A random sample of street images almost never contains a usable sign |
| OCR only ever rejects a crop, never admits one, in triage round 1 | Avoid biasing selection toward easy-to-read signs (OCR missed 58% of usable Westwood signs) |
| From triage round 2 on, only crops whose OCR text reads as parking are triaged. The dataset is signs that were triage-accepted **and** OCR-confirmed (`src/vlm_parking/dataset.py`). `triage.csv` keeps every human decision | Only about 7% of OCR-unknown accepts proved fully transcribable (2 of 28 in the gold sample). This selection may make condition A easier than for LA signs in general, and the writeup says so |
| Every sign is labeled by hand, blind, with no model-assisted pre-fill | Manual labeling of the whole cohort took about 2.4 hours, so assistance would save little, and every label stays human |
| A stay must be legal for its whole duration; questions never fall on holidays; signs with unreadable fine print are rejected | Agreed evaluator semantics (ANNOTATION_GUIDE.md §5) |
| Signs that restrict only oversize vehicles are rejected as `unsupported_condition` (38 signs) | The schema can't express a rule that applies to some vehicles and not others. Found by the inference smoke test, where the labels had applied them to all cars |
| Condition B transcriptions name each rule ("Time-limited parking; limit 120 minutes") | Removes reading from the task. The results show it also removes interpretation, which is why a sign-worded transcription is the next experiment |
| Parse failures score as wrong, not dropped | Dropping them would remove each model's hardest questions from its own denominator |
| The thinking model runs through a separate script at the model card's sampling settings | Keeps `run_inference.py`, which produced the original six runs, unchanged. Sampling means several seeds are needed |
