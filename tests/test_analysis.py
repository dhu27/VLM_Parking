"""Analysis: intervals resample signs, parse failures count against the model, and the decompositions add up."""

import numpy as np
import pandas as pd
import pytest

from vlm_parking.analysis import (MIN_SIGNS, accuracy_table, paired_gap, perception_decomposition,
                                  permit_distractor, ratio_ci, share)


def rows(sign_id: str, n: int, correct: bool, **kw) -> list[dict]:
    return [{"sign_id": sign_id, "query_id": f"{sign_id}-{i}", "correct": correct} | kw for i in range(n)]


def test_interval_resamples_signs_not_queries():
    # Two signs, 100 queries each: one always right, one always wrong. Resampling queries would give a tight
    # interval around 0.5; resampling the two signs can draw both from either, so the interval spans 0 to 1.
    d = pd.DataFrame(rows("right", 100, True) + rows("wrong", 100, False))
    s = share(d)
    assert s["est"] == pytest.approx(0.5)
    assert s["lo"] == pytest.approx(0.0) and s["hi"] == pytest.approx(1.0)
    assert s["n_signs"] == 2 and s["n_queries"] == 200


def test_interval_is_reproducible():
    num, den = np.array([3.0, 5, 1, 4]), np.array([5.0, 5, 5, 5])
    assert ratio_ci(num, den, seed=7) == ratio_ci(num, den, seed=7)


def test_parse_failure_counts_as_wrong():
    d = pd.DataFrame({"pred": ["legal", None], "verdict": ["legal", "legal"]})
    assert d.pred.eq(d.verdict).tolist() == [True, False]  # the rule load_scored applies


def test_small_strata_report_n_but_blank_the_estimate():
    big = [r for i in range(MIN_SIGNS) for r in rows(f"b{i}", 3, True, stratum="big")]
    small = [r for i in range(MIN_SIGNS - 1) for r in rows(f"s{i}", 3, True, stratum="small")]
    d = pd.DataFrame(big + small).assign(model="m", condition="A")
    t = accuracy_table(d, "stratum").set_index("stratum")
    assert t.loc["big", "est"] == 1.0 and not t.loc["big", "suppressed"]
    assert np.isnan(t.loc["small", "est"]) and t.loc["small", "suppressed"]
    assert t.loc["small", "n_signs"] == MIN_SIGNS - 1


def test_strata_keep_their_order():
    # built from dict rows, the stratum column loses its categorical order and sorts as plain strings,
    # putting "2020+" before "≤2016"
    labels = ["≤2016", "2017–19", "2020+"]
    d = pd.DataFrame([r | {"model": "m", "condition": "A", "yr": lab}
                      for lab in labels for i in range(MIN_SIGNS) for r in rows(f"{lab}{i}", 1, True)])
    d["yr"] = pd.Categorical(d.yr, labels, ordered=True)
    assert accuracy_table(d, "yr").yr.astype(str).tolist() == labels


def conditions(a: dict[str, bool], b: dict[str, bool]) -> pd.DataFrame:
    """One query per sign, answered in both conditions with the given correctness."""
    return pd.DataFrame([{"model": "m", "condition": c, "sign_id": s, "query_id": s, "correct": ok}
                         for c, ans in (("A", a), ("B", b)) for s, ok in ans.items()])


def test_identical_conditions_have_zero_gap():
    same = {f"s{i}": i % 3 != 0 for i in range(30)}
    g = paired_gap(conditions(same, same)).iloc[0]
    assert g.gap == 0 and g.lo == 0 and g.hi == 0  # paired, so no replicate can differ


def test_decomposition_separates_losses_from_rescues():
    # four outcomes, ten signs each (enough to clear MIN_SIGNS): right on both, lost on the image,
    # rescued by the image, wrong on both
    pattern = {"both": (True, True), "lost": (True, False), "rescued": (False, True), "neither": (False, False)}
    b = {f"{k}{i}": v[0] for k, v in pattern.items() for i in range(10)}
    a = {f"{k}{i}": v[1] for k, v in pattern.items() for i in range(10)}
    r = perception_decomposition(conditions(a, b)).iloc[0]
    assert (r.both_right, r.lost_on_image, r.rescued_by_image, r.both_wrong) == (10, 10, 10, 10)
    assert r.p_loss == 0.5  # half the queries it got right on text
    assert r.p_rescue == 0.5
    # the naive gap nets the losses against the rescues and reports no perception cost at all
    assert paired_gap(conditions(a, b)).iloc[0].gap == 0


def test_decomposition_blanks_small_strata():
    b = {f"s{i}": True for i in range(MIN_SIGNS - 1)}
    r = perception_decomposition(conditions(b, b)).iloc[0]
    assert np.isnan(r.p_loss) and r.n_signs == MIN_SIGNS - 1


def test_permit_distractor_detects_an_invented_exemption():
    base = {"model": "m", "condition": "A", "type": "permit_irrelevant", "parse_ok": True, "verdict": "illegal",
            "day": "MON", "time": "09:00", "duration_min": 60}
    d = pd.DataFrame([
        base | {"sign_id": "s1", "permit_district": None, "pred": "illegal"},
        base | {"sign_id": "s1", "permit_district": "7", "pred": "legal"},  # the permit changed the answer
        base | {"sign_id": "s2", "permit_district": None, "pred": "illegal"},
        base | {"sign_id": "s2", "permit_district": "7", "pred": "illegal"},
    ])
    r = permit_distractor(d).iloc[0]
    assert r.n_pairs == 2 and r.flip == 0.5 and r.invented_exemption == 0.5
