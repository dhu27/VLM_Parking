"""Phase 6: every table and figure from the six inference runs, in one command (plan §8).

    uv run python scripts/analyze.py                  # writes reports/phase6/
    uv run python scripts/analyze.py --n-boot 200     # faster, noisier intervals while iterating

Reads data/runs/<model>_<A|B>.jsonl and the query set. Every interval is a 95% percentile bootstrap over signs
(src/vlm_parking/analysis.py). Deterministic: the same inputs and seed reproduce the same numbers. The same
functions drive notebooks/03_results.ipynb, which walks through these results in prose.

Writes:
  summary.md                        every table, readable (the figures' table view)
  *.csv                             the same tables, machine-readable
  accuracy_by_panels.svg            the main figure
  perception_loss_by_panels.svg     its companion: what the naive gap hides
  error_review_sample.csv           50 condition-A failures to tag by hand
(each .svg gets an untracked .png beside it for quick viewing)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from vlm_parking import analysis as an  # noqa: E402
from vlm_parking import figures as fg  # noqa: E402

RUNS = Path("data/runs")
QUERIES = Path("data/queries")
OUT = Path("reports/phase6")


ci = fg.ci


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, rule] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def save(fig, path: Path) -> None:
    fig.savefig(path, facecolor=fg.SURFACE)
    fig.savefig(path.with_suffix(".png"), dpi=200, facecolor=fg.SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=an.N_BOOT)
    ap.add_argument("--seed", type=int, default=an.SEED)
    args = ap.parse_args()
    kw = {"n_boot": args.n_boot, "seed": args.seed}

    queries = pd.read_parquet(QUERIES / "queries_v1.parquet")
    signs = pd.read_parquet(QUERIES / "signs_v1.parquet")
    d = an.load_scored(RUNS, queries, signs)
    OUT.mkdir(parents=True, exist_ok=True)
    legal = queries.verdict.eq("legal").mean()
    majority = max(legal, 1 - legal)
    print(f"{len(d):,} answers | {queries.sign_id.nunique()} signs | {len(queries):,} queries | "
          f"{legal:.1%} legal, majority class {majority:.1%}")

    overall = an.accuracy_table(d, None, **kw)
    gaps = an.paired_gap(d, **kw)
    decomp = an.perception_decomposition(d, **kw)
    decomp_panels = an.perception_decomposition(d, "panels", **kw)
    strata = {by: an.accuracy_table(d, by, **kw) for by in ["panels", "type", "area", "sign_size", "capture_year"]}
    attribution = an.rule_attribution(d, **kw)
    permit = an.permit_distractor(d, **kw)
    tokens = an.prompt_tokens(d)
    failures = an.parse_failures(d)

    for name, t in [("overall", overall), ("gap", gaps), ("perception_decomposition", decomp),
                    ("perception_decomposition_by_panels", decomp_panels), ("rule_attribution", attribution),
                    ("permit_distractor", permit), ("prompt_tokens", tokens), ("parse_failures", failures)] + \
                   [(f"by_{k}", v) for k, v in strata.items()]:
        t.to_csv(OUT / f"{name}.csv", index=False)

    save(fg.accuracy_by_panels(strata["panels"], gaps, majority), OUT / "accuracy_by_panels.svg")
    save(fg.perception_loss_by_panels(decomp_panels), OUT / "perception_loss_by_panels.svg")
    sample = an.error_review_sample(d, 50, args.seed)
    sample.to_csv(OUT / "error_review_sample.csv", index=False)

    # --- summary.md: the readable twin of everything above ---------------------------------------------
    acc = overall.set_index(["model", "condition"])
    gp = gaps.set_index("model")
    head = pd.DataFrame([{
        "model": fg.NAMES[m], "B (text)": ci(*acc.loc[(m, "B"), ["est", "lo", "hi"]]),
        "A (image)": ci(*acc.loc[(m, "A"), ["est", "lo", "hi"]]), "A − B": fg.pts(*gp.loc[m, ["gap", "lo", "hi"]]),
    } for m in an.MODELS])
    dec = decomp.set_index("model")
    decomp_md = pd.DataFrame([{
        "model": fg.NAMES[m], "right on both": dec.loc[m, "both_right"], "lost on image": dec.loc[m, "lost_on_image"],
        "rescued by image": dec.loc[m, "rescued_by_image"], "wrong on both": dec.loc[m, "both_wrong"],
        "P(wrong on image | right on text)": ci(*dec.loc[m, ["p_loss", "p_loss_lo", "p_loss_hi"]]),
        "P(right on image | wrong on text)": ci(*dec.loc[m, ["p_rescue", "p_rescue_lo", "p_rescue_hi"]]),
    } for m in an.MODELS])
    att = attribution.set_index(["model", "condition"])
    att_md = pd.DataFrame([{"model": fg.NAMES[m], **{c: ci(*att.loc[(m, c), ["est", "lo", "hi"]]) for c in "BA"},
                            "queries (B)": att.loc[(m, "B"), "n_queries"]} for m in an.MODELS])
    per = permit.set_index(["model", "condition"])
    permit_md = pd.DataFrame([{
        "model": fg.NAMES[m], "condition": c, "pairs": per.loc[(m, c), "n_pairs"],
        "verdict flipped": ci(*per.loc[(m, c), ["flip", "flip_lo", "flip_hi"]]),
        "invented exemption": ci(*per.loc[(m, c), ["invented_exemption", "invented_lo", "invented_hi"]]),
    } for m in an.MODELS for c in an.CONDITIONS])
    tok = tokens.set_index("model")
    tok_md = pd.DataFrame([{"model": fg.NAMES[m], "B (text)": f"{tok.loc[m, 'B']:.0f}",
                            "A (image)": f"{tok.loc[m, 'A']:.0f}",
                            "extra image tokens vs smallest": f"{tok.loc[m, 'extra_image_tokens_vs_min']:.0f}"}
                           for m in an.MODELS])
    fail = failures.set_index(["model", "condition"])
    fail_md = pd.DataFrame([{"model": fg.NAMES[m], **{c: f"{fail.loc[(m, c), 'failures']} / {fail.loc[(m, c), 'n']}"
                                                       for c in "BA"}} for m in an.MODELS])

    titles = {"panels": "panels on the sign", "type": "query type", "area": "area",
              "sign_size": "sign size (native pixel height: small < 180, medium < 315, large)",
              "capture_year": "capture year"}
    body = [
        "# Phase 6 results", "",
        f"Generated by `scripts/analyze.py` ({args.n_boot:,} bootstrap replicates, seed {args.seed}). Every interval "
        "is a 95% percentile bootstrap **over signs**, not queries: a sign's ~15 queries share one image and one "
        f"transcription. Parse failures score as incorrect. {queries.sign_id.nunique()} signs, "
        f"{len(queries):,} queries per run; {legal:.1%} of queries are legal, so always answering "
        f"\"illegal\" scores **{majority:.1%}**. The walkthrough is `notebooks/03_results.ipynb`.", "",
        "## Headline: the perception cost", "",
        "Condition B hands the model the ground-truth transcription; condition A shows it the sign. "
        "A − B is how much accuracy each model loses when it has to read the sign itself.", "", md(head), "",
        "## What the gap is made of", "",
        "The naive A − B gap nets two things against each other: queries a model **loses on the image** (right "
        "given the text, wrong from the picture) and queries the image **rescues** (wrong given the text, right "
        "from the picture). A model that is often wrong on text has more queries available to rescue, which can "
        "shrink its gap without it reading any better. P(wrong on image | right on text) isolates the perception "
        "loss: the model demonstrably could apply the rules, so the failure is in reading the sign.", "",
        md(decomp_md), "",
    ]
    for by, t in strata.items():
        body += [f"## Accuracy by {titles[by]}", "", f"Strata with fewer than {an.MIN_SIGNS} signs are blanked (—).", "",
                 md(fg.stratum_table(t, by)), ""]
    body += [
        "## Rule attribution", "",
        "Of the gold-illegal queries the model also calls illegal, the share where it names the same governing "
        "panel as the evaluator. **Strict match**: the evaluator names the most restrictive violated panel, so "
        "citing another panel that is also violated counts as a mismatch. **Condition A is not comparable**: "
        "there the model counts physical plates in the image, while the annotation guide splits one plate into "
        "several panels (\"NO STOPPING 7–9AM, 4–7PM\" is two), so indices can disagree without any misreading. "
        "Read condition B only.", "", md(att_md), "",
        "## Permit distractor", "",
        "The same stay asked with and without a permit for a district the sign never mentions. The correct "
        "answer cannot change, so any flip is caused by the permit being mentioned. *Invented exemption*: of "
        "the pairs the model correctly calls illegal without the permit, the share it flips to legal once the "
        "irrelevant permit appears. Pairs with a parse failure are excluded.", "", md(permit_md), "",
        "## Visual tokens", "",
        "Mean prompt tokens. All three models sit on Qwen3-family language backbones and receive identical "
        "condition-A text, so the difference between models in condition A is image tokens: how much visual "
        "information each is given from the same crop.", "", md(tok_md), "",
        "## Parse failures", "", md(fail_md), "",
        "## Hand-tagging sample", "",
        f"`error_review_sample.csv`: {len(sample)} condition-A failures where the **same model was right on the "
        "text**, so the error is in reading the sign. Tag each with one of: "
        + ", ".join(f"`{t}`" for t in an.ERROR_TAGS) + ". The two before `other` came from the smoke test "
        "and are not in the plan's list.", "",
    ]
    (OUT / "summary.md").write_text("\n".join(body))
    print(f"\n{md(head)}\n\n{md(decomp_md)}\n\nwrote {OUT}/")


if __name__ == "__main__":
    main()
