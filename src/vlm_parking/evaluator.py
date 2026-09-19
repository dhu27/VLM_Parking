"""Ground-truth evaluator: is a given stay legal under a transcribed sign?

Semantics (see ANNOTATION_GUIDE.md, "Evaluator semantics"):
- A query is a *stay*: the interval [arrival, arrival + duration). The whole stay must be legal, so a
  stay that runs into a restriction is illegal.
- Windows are half-open [start, end): a rule for 10:00-12:00 is in force at 10:00 and over at 12:00.
- Prohibitions (no stopping / standing / parking, loading zones) are violated by any overlap.
- permit_only is violated by any overlap unless the driver holds that district's permit.
- time_limited / metered are violated when the time spent inside the window exceeds limit_min.
  Meters are assumed paid; a metered panel without a limit never makes a stay illegal.
- A panel's `district` (on any rule except permit_only) exempts drivers holding that permit.
- Queries never fall on holidays, so except_holidays has no effect. Arrows are ignored.
- If several panels are violated, the most restrictive governs; ties go to the higher (lower-order) panel.
"""

from __future__ import annotations

from vlm_parking.schema import DAYS, Panel, Query, Sign, Verdict, to_minutes

DAY_MIN = 24 * 60
WEEK_MIN = 7 * DAY_MIN

# Lower = more restrictive; decides which violated panel governs.
SEVERITY = {
    "no_stopping": 0,
    "no_standing": 1,
    "no_parking": 2,
    "passenger_loading": 3,
    "commercial_loading": 3,
    "permit_only": 4,
    "time_limited": 5,
    "metered": 5,
}
PROHIBITIONS = {"no_stopping", "no_standing", "no_parking", "passenger_loading", "commercial_loading"}

Interval = tuple[int, int]


def _wrap(start: int, end: int) -> list[Interval]:
    """An interval on the weekly minute axis, split if it runs past Sunday midnight."""
    length = end - start
    start %= WEEK_MIN
    end = start + length
    if end <= WEEK_MIN:
        return [(start, end)]
    return [(start, WEEK_MIN), (0, end - WEEK_MIN)]


def stay_intervals(q: Query) -> list[Interval]:
    start = DAYS.index(q.day) * DAY_MIN + to_minutes(q.time)
    return _wrap(start, start + q.duration_min)


def panel_intervals(p: Panel) -> list[Interval]:
    out: list[Interval] = []
    for day in p.days:
        base = DAYS.index(day) * DAY_MIN
        if p.start is None:
            out += _wrap(base, base + DAY_MIN)
            continue
        s, e = to_minutes(p.start), to_minutes(p.end)
        if e <= s:  # overnight: runs into the next day
            e += DAY_MIN
        out += _wrap(base + s, base + e)
    return out


def overlap_minutes(a: list[Interval], b: list[Interval]) -> int:
    return sum(max(0, min(a1, b1) - max(a0, b0)) for a0, a1 in a for b0, b1 in b)


def _describe(p: Panel) -> str:
    when = "all day" if p.start is None else f"{p.start}-{p.end}"
    days = "every day" if len(p.days) == 7 else ",".join(p.days)
    return f"panel {p.order} ({p.rule}, {when}, {days})"


def evaluate(sign: Sign, query: Query) -> Verdict:
    """Returns (legal | illegal | ambiguous, governing_panel_order, reason)."""
    if sign.ambiguous:
        return Verdict(verdict="ambiguous", reason="sign flagged ambiguous (panels conflict)")

    stay = stay_intervals(query)
    violations: list[tuple[int, int, str]] = []  # (severity, order, reason)
    for p in sign.panels:
        if p.rule == "unrestricted":
            continue
        inside = overlap_minutes(stay, panel_intervals(p))
        if inside == 0:
            continue
        if p.rule != "permit_only" and p.district and query.permit_district == p.district:
            continue  # permit holders are exempt from this panel

        if p.rule in PROHIBITIONS:
            violations.append((SEVERITY[p.rule], p.order, f"stay overlaps {_describe(p)} by {inside} min"))
        elif p.rule == "permit_only":
            if query.permit_district != p.district:
                violations.append((SEVERITY[p.rule], p.order, f"no district {p.district} permit during {_describe(p)}"))
        elif p.rule in ("time_limited", "metered"):
            if p.limit_min is not None and inside > p.limit_min:
                violations.append(
                    (SEVERITY[p.rule], p.order, f"{inside} min inside {_describe(p)} exceeds {p.limit_min} min limit")
                )

    if not violations:
        return Verdict(verdict="legal", reason="no panel is violated by this stay")
    _, order, reason = min(violations)
    return Verdict(verdict="illegal", governing_panel=order, reason=reason)
