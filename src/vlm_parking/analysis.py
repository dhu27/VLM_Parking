"""Phase 6: score the inference runs and summarize them with confidence intervals over signs.

Every interval here resamples *signs*, not queries (plan §8). A sign contributes ~15 queries that share one
image and one transcription, so its answers are not independent draws; resampling queries would treat 3,451
answers as 3,451 independent observations and report intervals several times too narrow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["qwen3-vl-8b", "internvl3_5-8b", "minicpm-v-4_5"]
CONDITIONS = ["A", "B"]  # A: the sign crop. B: the ground-truth transcription as text.
MIN_SIGNS = 20  # plan §8: report n per stratum, blank the estimate below this many signs
N_BOOT = 2000
SEED = 0


# --- scoring ----------------------------------------------------------------------------------------


def load_scored(runs: Path, queries: pd.DataFrame, signs: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, condition, query), joined to the query's gold answer and its sign's attributes.

    A parse failure scores as incorrect, never dropped: dropping would quietly remove a model's hardest cases
    from its own denominator.
    """
    rows = []
    for model in MODELS:
        for cond in CONDITIONS:
            for line in (runs / f"{model}_{cond}.jsonl").open():
                r = json.loads(line)
                a = r["answer"] or {}
                rows.append({"model": model, "condition": cond, "query_id": r["query_id"],
                             "pred": a.get("verdict"), "pred_panel": a.get("governing_panel"),
                             "pred_reason": a.get("reason"), "parse_ok": r["parse_ok"],
                             "prompt_tokens": r["prompt_tokens"]})
    d = pd.DataFrame(rows)

    # Results must cover exactly the current query set. A mismatch means the runs predate a regeneration of
    # the queries (as happened when the oversize-vehicle signs were rejected), and scoring them would be wrong.
    want = set(queries.query_id)
    for (model, cond), g in d.groupby(["model", "condition"]):
        got = set(g.query_id)
        if got != want or len(g) != len(want):
            raise ValueError(f"{model} {cond}: {len(got - want)} answers for unknown queries, "
                             f"{len(want - got)} queries unanswered, {len(g) - len(got)} duplicates")

    d = d.merge(queries, on="query_id", how="left", validate="many_to_one")
    d = d.merge(signs[["sign_id", "height_native", "year", "crop_path"]], on="sign_id", how="left",
                validate="many_to_one")
    d["correct"] = d.pred.eq(d.verdict)  # a parse failure has pred None, so it never equals the gold verdict
    d["panels"] = pd.Categorical(d.n_panels.clip(upper=3).map({1: "1", 2: "2", 3: "3+"}), ["1", "2", "3+"], ordered=True)
    d["sign_size"] = pd.Categorical(
        pd.cut(d.height_native, [0, 180, 315, np.inf], right=False, labels=["small", "medium", "large"]),
        ["small", "medium", "large"], ordered=True)
    d["capture_year"] = pd.Categorical(
        pd.cut(d.year, [0, 2017, 2020, np.inf], right=False, labels=["≤2016", "2017–19", "2020+"]),
        ["≤2016", "2017–19", "2020+"], ordered=True)
    return d


# --- the sign bootstrap -------------------------------------------------------------------------------


def _sign_sums(x: pd.DataFrame, value: str, index: pd.Index | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-sign (numerator, denominator) for the share of rows where `value` is true."""
    g = x.groupby("sign_id")[value].agg(["sum", "count"])
    if index is not None:
        g = g.reindex(index, fill_value=0)
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)


def _keep_order(t: pd.DataFrame, d: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """Restore a stratum's category order, which building rows from dicts drops (else "2020+" sorts before
    "≤2016" and "large" before "small")."""
    if by and isinstance(d[by].dtype, pd.CategoricalDtype):
        t[by] = pd.Categorical(t[by], categories=d[by].cat.categories, ordered=True)
        t = t.sort_values(["model", by] + (["condition"] if "condition" in t else [])).reset_index(drop=True)
    return t


def _resample(n_signs: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_signs, size=(n_boot, n_signs))


def ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Pooled share sum(num)/sum(den) with a 95% percentile interval from resampling signs with replacement."""
    if den.sum() == 0:
        return np.nan, np.nan, np.nan
    idx = _resample(len(num), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return num.sum() / den.sum(), lo, hi


def share(x: pd.DataFrame, value: str = "correct", n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    num, den = _sign_sums(x, value)
    point, lo, hi = ratio_ci(num, den, n_boot, seed)
    return {"n_signs": len(num), "n_queries": int(den.sum()), "est": point, "lo": lo, "hi": hi}


def accuracy_table(d: pd.DataFrame, by: str | None = None, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Accuracy per model, condition and (optionally) stratum, with a sign-bootstrap CI.

    n_signs counts the signs with at least one query in the stratum. Strata below MIN_SIGNS are still listed,
    so the n is reported, but their estimate is blanked (plan §8).
    """
    keys = ["model", "condition"] + ([by] if by else [])
    out = []
    for key, g in d.groupby(keys, observed=True):
        out.append(dict(zip(keys, key)) | share(g, "correct", n_boot, seed))
    t = pd.DataFrame(out)
    t["suppressed"] = t.n_signs < MIN_SIGNS
    t.loc[t.suppressed, ["est", "lo", "hi"]] = np.nan
    return _keep_order(t, d, by)


def paired_gap(d: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """A minus B accuracy per model: the perception cost, the headline number (plan §8).

    Paired: each bootstrap replicate draws one set of signs and scores both conditions on it, so the interval
    reflects that A and B are the same signs and the same queries, not two independent samples.
    """
    out = []
    for model, g in d.groupby("model"):
        a, b = g[g.condition == "A"], g[g.condition == "B"]
        signs = pd.Index(sorted(set(a.sign_id)))
        na, da = _sign_sums(a, "correct", signs)
        nb, db = _sign_sums(b, "correct", signs)
        idx = _resample(len(signs), n_boot, seed)
        boot = na[idx].sum(1) / da[idx].sum(1) - nb[idx].sum(1) / db[idx].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({"model": model, "acc_A": na.sum() / da.sum(), "acc_B": nb.sum() / db.sum(),
                    "gap": na.sum() / da.sum() - nb.sum() / db.sum(), "lo": lo, "hi": hi, "n_signs": len(signs)})
    return pd.DataFrame(out)


# --- what the gap is made of --------------------------------------------------------------------------


def perception_decomposition(d: pd.DataFrame, by: str | None = None, n_boot: int = N_BOOT,
                             seed: int = SEED) -> pd.DataFrame:
    """Split each model's answers by whether it was right on the text (B) and on the image (A).

    P(A wrong | B right) isolates perception: the model could apply the rules when handed them, so failing
    from the image means it misread the sign. The naive A−B gap nets those losses against rescues
    (B wrong, A right), so a model that is often wrong on text can show a small gap without reading well.
    Strata with fewer than MIN_SIGNS signs among the text-correct queries have their rates blanked.
    """
    index = ["model", "sign_id", "query_id"] + ([by] if by else [])
    # pivot, not pivot_table: it refuses duplicates instead of aggregating them, and a missing cell would
    # otherwise become True under astype(bool)
    w = d.pivot(index=index, columns="condition", values="correct").reset_index()
    if w[["A", "B"]].isna().any().any():
        raise ValueError("a query is missing an answer in one condition")
    w["A"], w["B"] = w["A"].astype(bool), w["B"].astype(bool)
    w["loss"] = w.B & ~w.A  # right on text, wrong on the image
    w["rescue"] = ~w.B & w.A  # wrong on text, right on the image
    keys = ["model"] + ([by] if by else [])
    out = []
    for key, g in w.groupby(keys, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        loss = share(g[g.B], "loss", n_boot, seed)
        rescue = share(g[~g.B], "rescue", n_boot, seed)
        row = dict(zip(keys, key)) | {
            "both_right": int((g.A & g.B).sum()), "lost_on_image": int(g.loss.sum()),
            "rescued_by_image": int(g.rescue.sum()), "both_wrong": int((~g.A & ~g.B).sum()),
            "n_signs": loss["n_signs"], "p_loss": loss["est"], "p_loss_lo": loss["lo"], "p_loss_hi": loss["hi"],
            "p_rescue": rescue["est"], "p_rescue_lo": rescue["lo"], "p_rescue_hi": rescue["hi"]}
        if loss["n_signs"] < MIN_SIGNS:
            row |= {k: np.nan for k in ("p_loss", "p_loss_lo", "p_loss_hi", "p_rescue", "p_rescue_lo", "p_rescue_hi")}
        out.append(row)
    return _keep_order(pd.DataFrame(out), d, by)


def answer_bias(d: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """How often each model answers "illegal", against how often it should.

    A model that leans toward "illegal" gains on illegal queries and loses on legal ones. Most queries are
    illegal, so a lean that appears only in one condition can buy "rescues" in that condition with no better
    reading of the sign. A parse failure counts as not saying "illegal".
    """
    x = d.assign(says_illegal=d.pred.eq("illegal"))
    out = []
    for (model, cond), g in x.groupby(["model", "condition"]):
        out.append({"model": model, "condition": cond} | share(g, "says_illegal", n_boot, seed))
    t = pd.DataFrame(out)
    t["gold"] = d.drop_duplicates("query_id").verdict.eq("illegal").mean()
    return t


def rule_attribution(d: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Among gold-illegal queries the model also calls illegal, how often it names the evaluator's panel.

    Strict match. The evaluator names the most restrictive violated panel (evaluator.SEVERITY), so a model that
    cites a different panel that is *also* violated counts as a mismatch here. Only well-defined in condition B:
    the transcription numbers panels exactly as the label does, whereas in condition A the model counts physical
    plates, and the guide splits one plate into several panels ("NO STOPPING 7-9AM, 4-7PM" is two panels).
    """
    x = d[(d.verdict == "illegal") & (d.pred == "illegal")].copy()
    x["panel_match"] = pd.to_numeric(x.pred_panel, errors="coerce").eq(x.governing_panel.astype(float))
    out = []
    for (model, cond), g in x.groupby(["model", "condition"]):
        out.append({"model": model, "condition": cond} | share(g, "panel_match", n_boot, seed))
    return pd.DataFrame(out)


def permit_distractor(d: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """The same stay asked with and without a permit for a district the sign never mentions (plan §8).

    The gold verdict is identical by construction, so any change in the model's verdict is caused by the permit
    being mentioned. `invented_exemption`: of the pairs the model correctly calls illegal without the permit,
    how many it flips to legal once the irrelevant permit appears. Pairs with a parse failure are left out, since
    "the verdict changed" means nothing when one side has no verdict.
    """
    p = d[(d.type == "permit_irrelevant") & d.parse_ok]
    key = ["model", "condition", "sign_id", "day", "time", "duration_min"]
    pairs = p[p.permit_district.isna()].merge(p[p.permit_district.notna()], on=key, suffixes=("_none", "_permit"))
    pairs["flip"] = pairs.pred_none != pairs.pred_permit
    pairs["invented"] = pairs.pred_permit == "legal"
    out = []
    for (model, cond), g in pairs.groupby(["model", "condition"]):
        flip = share(g, "flip", n_boot, seed)
        base = g[(g.verdict_none == "illegal") & (g.pred_none == "illegal")]
        inv = share(base, "invented", n_boot, seed)
        out.append({"model": model, "condition": cond, "n_pairs": len(g), "n_signs": flip["n_signs"],
                    "flip": flip["est"], "flip_lo": flip["lo"], "flip_hi": flip["hi"],
                    "n_illegal_pairs": len(base), "invented_exemption": inv["est"],
                    "invented_lo": inv["lo"], "invented_hi": inv["hi"]})
    return pd.DataFrame(out)


def prompt_tokens(d: pd.DataFrame) -> pd.DataFrame:
    """Mean prompt tokens per model and condition.

    All three models are built on Qwen3-family language backbones and receive identical condition-A text, so in
    condition A the difference between two models' prompt lengths is image tokens: how much visual information
    each model is given from the same crop.
    """
    t = d.groupby(["model", "condition"]).prompt_tokens.mean().unstack()
    t["extra_image_tokens_vs_min"] = t["A"] - t["A"].min()
    return t.reset_index()


def parse_failures(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby(["model", "condition"]).parse_ok.agg(n="size", failures=lambda s: int((~s).sum())).reset_index()


# --- the hand-tagging sample -------------------------------------------------------------------------

ERROR_TAGS = ["misread digit", "AM/PM flip", "dropped panel", "wrong day range", "boundary error",
              "ignored exception", "hallucinated rule", "invented permit exemption",
              # seen in the smoke test, not in the plan's list
              "minute-carry arithmetic", "arrival-only reasoning", "other"]


def error_review_sample(d: pd.DataFrame, n: int = 50, seed: int = SEED) -> pd.DataFrame:
    """Condition-A failures where the same model was right on the text, for hand-tagging (plan §8).

    Restricted to "lost on the image" because those are the perception errors: the model could apply the rules,
    so the failure is in reading the sign. Split evenly across models, seeded.
    """
    a = d[d.condition == "A"].set_index(["model", "query_id"])
    b = d[d.condition == "B"].set_index(["model", "query_id"])
    lost = a[~a.correct & b.correct.reindex(a.index)].reset_index()
    per = -(-n // len(MODELS))
    picks = [g.sample(min(per, len(g)), random_state=seed) for _, g in lost.groupby("model")]
    s = pd.concat(picks).head(n)
    s = s.merge(b.reset_index()[["model", "query_id", "pred_reason"]].rename(columns={"pred_reason": "reason_on_text"}),
                on=["model", "query_id"])
    s["question"] = (s.day + " " + s.time + " for " + s.duration_min.astype(str) + " min"
                     + s.permit_district.map(lambda x: f", permit {x}" if isinstance(x, str) else ""))
    s = s.rename(columns={"verdict": "gold", "reason": "gold_reason", "pred": "model_verdict",
                          "pred_reason": "reason_on_image"})
    s["tag"], s["notes"] = "", ""
    s["crop"] = "data/collect/" + s.crop_path
    cols = ["model", "sign_id", "query_id", "panels", "question", "gold", "model_verdict", "gold_reason",
            "reason_on_image", "reason_on_text", "crop", "tag", "notes"]
    return s[cols].sort_values(["model", "sign_id"]).reset_index(drop=True)
