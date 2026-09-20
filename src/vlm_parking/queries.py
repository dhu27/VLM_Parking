"""Phase 4: generate queries for a labeled sign, with ground truth from the evaluator.

Per sign (plan §6): 5 clearly legal, 5 clearly illegal, 3 boundary, 2 permit-dependent.

The types are defined by how stable the answer is, which works for every rule type:
- clearly legal / illegal: the verdict is unchanged when arrival or duration moves by ±30 min
- boundary: the verdict flips somewhere within ±5 min of the query
- permit: the same stay gets different verdicts with and without the district's permit (only 13 of 273
  signs post a district)
- permit_irrelevant: for signs that post no district, the same stay asked with and without a permit for
  some other district. The answer must not change; this probes models inventing an exemption because a
  permit was mentioned.

Signs that can't fill a category (e.g. "NO PARKING ANY TIME" has no legal times) get the shortfall
redistributed to the categories that are possible, so every sign keeps 15 queries.
"""

from __future__ import annotations

import random

from vlm_parking.evaluator import DAY_MIN, evaluate, panel_intervals
from vlm_parking.schema import DAYS, Query, Sign

DURATIONS = [15, 30, 45, 60, 90, 120, 180, 240]
CLEAR_MARGIN = 30  # minutes
BOUNDARY_MARGIN = 5
TARGETS = {"clear_legal": 5, "clear_illegal": 5, "boundary": 3, "permit": 2}
# districts to borrow for permit_irrelevant queries (real LA preferential-parking district numbers)
OTHER_DISTRICTS = ["5", "7", "12", "31", "36", "43"]


def _shift(q: Query, minutes: int = 0, extra: int = 0) -> Query:
    """Move the arrival by `minutes` and change the duration by `extra` (within 5..1440)."""
    t = (DAYS.index(q.day) * DAY_MIN + int(q.time[:2]) * 60 + int(q.time[3:]) + minutes) % (7 * DAY_MIN)
    day, mins = DAYS[t // DAY_MIN], t % DAY_MIN
    return Query(
        day=day,
        time=f"{mins // 60:02d}:{mins % 60:02d}",
        duration_min=max(5, min(24 * 60, q.duration_min + extra)),
        permit_district=q.permit_district,
    )


def _perturbations(q: Query, minutes: int) -> list[Query]:
    return [_shift(q, minutes), _shift(q, -minutes), _shift(q, 0, minutes), _shift(q, 0, -minutes)]


def is_clear(sign: Sign, q: Query, margin: int = CLEAR_MARGIN) -> bool:
    """The verdict survives ±margin minutes of arrival and duration wobble."""
    v = evaluate(sign, q).verdict
    return all(evaluate(sign, p).verdict == v for p in _perturbations(q, margin))


def is_boundary(sign: Sign, q: Query, margin: int = BOUNDARY_MARGIN) -> bool:
    """A few minutes either way changes the answer."""
    v = evaluate(sign, q).verdict
    return any(evaluate(sign, p).verdict != v for p in _perturbations(q, margin))


def permit_districts(sign: Sign) -> list[str]:
    return sorted({p.district for p in sign.panels if p.district})


def boundary_minutes(sign: Sign) -> list[int]:
    """Week-minute positions where a panel's window starts or ends."""
    edges: set[int] = set()
    for p in sign.panels:
        for start, end in panel_intervals(p):
            edges.add(start % (7 * DAY_MIN))
            edges.add(end % (7 * DAY_MIN))
    return sorted(edges)


def _query_at(week_minute: int, duration: int, permit: str | None = None) -> Query:
    week_minute %= 7 * DAY_MIN
    day, mins = DAYS[week_minute // DAY_MIN], week_minute % DAY_MIN
    return Query(day=day, time=f"{mins // 60:02d}:{mins % 60:02d}", duration_min=duration, permit_district=permit)


def _random_query(rng: random.Random, permit: str | None = None) -> Query:
    return _query_at(rng.randrange(7 * DAY_MIN), rng.choice(DURATIONS), permit)


def generate_for_sign(sign: Sign, seed: int = 0, targets: dict[str, int] = TARGETS, attempts: int = 4000) -> list[dict]:
    """Queries for one sign, each with its evaluator verdict. Deterministic given seed."""
    rng = random.Random(f"{sign.sign_id}:{seed}")
    found: dict[str, list[Query]] = {k: [] for k in targets}
    seen: set[tuple] = set()

    def key(q: Query) -> tuple:
        return (q.day, q.time, q.duration_min, q.permit_district)

    def take(kind: str, q: Query) -> bool:
        if len(found[kind]) >= targets[kind] or key(q) in seen:
            return False
        seen.add(key(q))
        found[kind].append(q)
        return True

    total_target = sum(targets.values())
    districts = permit_districts(sign)
    edges = boundary_minutes(sign)
    found.setdefault("permit_irrelevant", [])

    # permit-dependent pairs: same stay, with and without the sign's own district permit
    for _ in range(attempts if districts else 0):
        if len(found["permit"]) >= targets["permit"]:
            break
        d = rng.choice(districts)
        q = _random_query(rng)
        with_permit = Query(**{**q.model_dump(), "permit_district": d})
        if evaluate(sign, q).verdict != evaluate(sign, with_permit).verdict and is_clear(sign, q) and is_clear(sign, with_permit):
            if take("permit", q):
                take("permit", with_permit)

    # no district posted (or no stay where it matters): ask with a permit for some *other* district,
    # which must not change the answer
    targets = dict(targets)
    if len(found["permit"]) < targets["permit"]:
        targets["permit_irrelevant"] = targets["permit"] - len(found["permit"])
        targets["permit"] = len(found["permit"])
        pool = [d for d in OTHER_DISTRICTS if d not in districts]
        for _ in range(attempts):
            if len(found["permit_irrelevant"]) >= targets["permit_irrelevant"]:
                break
            q = _random_query(rng)
            with_permit = Query(**{**q.model_dump(), "permit_district": rng.choice(pool)})
            if evaluate(sign, q).verdict in ("legal", "illegal") and is_clear(sign, q) and is_clear(sign, with_permit):
                if take("permit_irrelevant", q):
                    take("permit_irrelevant", with_permit)

    # boundary queries: start or end a few minutes either side of a window edge
    for _ in range(attempts if edges else 0):
        if len(found["boundary"]) >= targets["boundary"]:
            break
        edge = rng.choice(edges)
        delta = rng.randint(-BOUNDARY_MARGIN, BOUNDARY_MARGIN)
        duration = rng.choice([15, 30, 45, 60])
        q = _query_at(edge + delta if rng.random() < 0.5 else edge + delta - duration, duration)
        if is_boundary(sign, q):
            take("boundary", q)

    # clearly legal / illegal
    for _ in range(attempts):
        if len(found["clear_legal"]) >= targets["clear_legal"] and len(found["clear_illegal"]) >= targets["clear_illegal"]:
            break
        q = _random_query(rng)
        v = evaluate(sign, q).verdict
        if v in ("legal", "illegal") and is_clear(sign, q):
            take(f"clear_{v}", q)

    # Redistribute any shortfall so every sign still gets 15 queries, without letting one class run away
    # (a "NO PARKING ANY TIME" sign has no legal times at all). Permit-distractor pairs first, since they
    # exist for any sign, then the clear classes up to a cap.
    caps = {"clear_legal": 8, "clear_illegal": 8, "boundary": targets.get("boundary", 3), "permit": targets["permit"],
            "permit_irrelevant": 6}
    pool = [d for d in OTHER_DISTRICTS if d not in districts]
    for _ in range(attempts):
        if sum(len(v) for v in found.values()) >= total_target:
            break
        q = _random_query(rng)
        v = evaluate(sign, q).verdict
        if v not in ("legal", "illegal") or not is_clear(sign, q):
            continue
        if len(found["permit_irrelevant"]) + 2 <= caps["permit_irrelevant"] and sum(len(x) for x in found.values()) + 2 <= total_target:
            with_permit = Query(**{**q.model_dump(), "permit_district": rng.choice(pool)})
            if is_clear(sign, with_permit) and key(q) not in seen and key(with_permit) not in seen:
                for x in (q, with_permit):
                    seen.add(key(x))
                    found["permit_irrelevant"].append(x)
                continue
        kind = f"clear_{v}"
        if len(found[kind]) < caps[kind] and key(q) not in seen:
            seen.add(key(q))
            found[kind].append(q)

    rows = []
    for kind, qs in found.items():
        for q in qs:
            verdict = evaluate(sign, q)
            rows.append({
                "sign_id": sign.sign_id,
                "type": kind,
                "day": q.day,
                "time": q.time,
                "duration_min": q.duration_min,
                "permit_district": q.permit_district,
                "verdict": verdict.verdict,
                "governing_panel": verdict.governing_panel,
                "reason": verdict.reason,
            })
    rows.sort(key=lambda r: (r["type"], r["day"], r["time"]))
    for i, r in enumerate(rows):
        r["query_id"] = f"{sign.sign_id}#{i:02d}"
    return rows
