"""Query generation: every query must satisfy the definition of its type, and be reproducible."""

import collections

import pytest

from vlm_parking.evaluator import evaluate
from vlm_parking.queries import TARGETS, generate_for_sign, is_boundary, is_clear
from vlm_parking.schema import DAYS, Panel, Query, Sign

ALL = list(DAYS)
WEEKDAYS = ALL[:5]


def sign(*panels: dict, sign_id="t", **kw) -> Sign:
    return Sign(sign_id=sign_id, panels=[Panel(order=i, **p) for i, p in enumerate(panels)], **kw)


TYPICAL = sign(
    {"rule": "no_parking", "days": ["THU"], "start": "08:00", "end": "10:00"},
    {"rule": "time_limited", "days": ALL[:6], "start": "08:00", "end": "18:00", "limit_min": 120},
)
PERMIT_SIGN = sign(
    {"rule": "no_parking", "days": ALL, "start": "18:00", "end": "08:00", "district": "31"},
    {"rule": "time_limited", "days": ALL, "start": "08:00", "end": "18:00", "limit_min": 120, "district": "31"},
    sign_id="permit",
)
ALWAYS_BANNED = sign({"rule": "no_parking", "days": ALL}, sign_id="banned")


def as_query(row: dict) -> Query:
    return Query(day=row["day"], time=row["time"], duration_min=row["duration_min"], permit_district=row["permit_district"])


def test_counts_and_types():
    rows = generate_for_sign(TYPICAL)
    assert len(rows) == sum(TARGETS.values()) == 15
    counts = collections.Counter(r["type"] for r in rows)
    assert counts["clear_legal"] == 5 and counts["clear_illegal"] == 5 and counts["boundary"] == 3
    assert counts["permit_irrelevant"] == 2  # no district posted on this sign
    assert len({r["query_id"] for r in rows}) == 15


def test_verdicts_match_the_evaluator():
    for row in generate_for_sign(TYPICAL):
        v = evaluate(TYPICAL, as_query(row))
        assert row["verdict"] == v.verdict and row["governing_panel"] == v.governing_panel


@pytest.mark.parametrize("kind,expected", [("clear_legal", "legal"), ("clear_illegal", "illegal")])
def test_clear_queries_are_stable(kind, expected):
    rows = [r for r in generate_for_sign(TYPICAL) if r["type"] == kind]
    assert rows
    for row in rows:
        assert row["verdict"] == expected
        assert is_clear(TYPICAL, as_query(row))  # survives ±30 min


def test_boundary_queries_flip_within_five_minutes():
    rows = [r for r in generate_for_sign(TYPICAL) if r["type"] == "boundary"]
    assert rows
    for row in rows:
        assert is_boundary(TYPICAL, as_query(row))


def test_permit_pairs_differ_only_by_the_permit():
    rows = [r for r in generate_for_sign(PERMIT_SIGN) if r["type"] == "permit"]
    assert len(rows) == 2
    a, b = (as_query(r) for r in rows)
    assert (a.day, a.time, a.duration_min) == (b.day, b.time, b.duration_min)
    assert {a.permit_district, b.permit_district} == {None, "31"}
    assert evaluate(PERMIT_SIGN, a).verdict != evaluate(PERMIT_SIGN, b).verdict


def test_permit_irrelevant_pairs_must_not_change_the_answer():
    rows = [r for r in generate_for_sign(TYPICAL) if r["type"] == "permit_irrelevant"]
    assert len(rows) == 2
    a, b = (as_query(r) for r in rows)
    assert (a.day, a.time, a.duration_min) == (b.day, b.time, b.duration_min)
    assert (a.permit_district is None) != (b.permit_district is None)
    assert evaluate(TYPICAL, a).verdict == evaluate(TYPICAL, b).verdict  # the sign grants no exemption


def test_sign_with_no_legal_time_stays_balanced_within_reason():
    rows = generate_for_sign(ALWAYS_BANNED)
    counts = collections.Counter(r["type"] for r in rows)
    assert all(r["verdict"] == "illegal" for r in rows)  # nothing else is possible for this sign
    assert counts["clear_illegal"] <= 8  # capped, rather than 15 of the same thing
    assert len(rows) <= 15


def test_deterministic_given_seed():
    assert generate_for_sign(TYPICAL, seed=1) == generate_for_sign(TYPICAL, seed=1)
    assert generate_for_sign(TYPICAL, seed=1) != generate_for_sign(TYPICAL, seed=2)
