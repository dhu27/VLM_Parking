"""Evaluator tests: every case listed in the plan (section 4.3) plus the agreed semantics
(whole stay must be legal; no holidays; permit exemptions)."""

import pytest
from pydantic import ValidationError

from vlm_parking.evaluator import evaluate
from vlm_parking.schema import DAYS, Panel, Query, Sign

WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI"]
MON_SAT = ["MON", "TUE", "WED", "THU", "FRI", "SAT"]
ALL = list(DAYS)


def sign(*panels: dict, **kw) -> Sign:
    return Sign(sign_id="t", panels=[Panel(order=i, **p) for i, p in enumerate(panels)], **kw)


def q(day: str, time: str, duration: int = 1, permit: str | None = None) -> Query:
    return Query(day=day, time=time, duration_min=duration, permit_district=permit)


def verdict(s: Sign, query: Query) -> str:
    return evaluate(s, query).verdict


# --- each rule type in isolation ------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["no_stopping", "no_standing", "no_parking", "passenger_loading", "commercial_loading"])
def test_prohibitions_in_and_out_of_window(rule):
    s = sign({"rule": rule, "days": WEEKDAYS, "start": "07:00", "end": "09:00"})
    assert verdict(s, q("TUE", "08:00", 30)) == "illegal"
    assert evaluate(s, q("TUE", "08:00", 30)).governing_panel == 0
    assert verdict(s, q("TUE", "10:00", 30)) == "legal"
    assert verdict(s, q("SAT", "08:00", 30)) == "legal"


def test_permit_only():
    s = sign({"rule": "permit_only", "days": ALL, "start": "18:00", "end": "08:00", "district": "7"})
    assert verdict(s, q("WED", "20:00", 60)) == "illegal"
    assert verdict(s, q("WED", "20:00", 60, permit="7")) == "legal"
    assert verdict(s, q("WED", "20:00", 60, permit="12")) == "illegal"  # wrong district
    assert verdict(s, q("WED", "12:00", 60)) == "legal"  # outside the window


def test_time_limited_duration_fits_vs_exceeds():
    s = sign({"rule": "time_limited", "days": MON_SAT, "start": "08:00", "end": "18:00", "limit_min": 120})
    assert verdict(s, q("MON", "10:00", 120)) == "legal"  # exactly the limit
    assert verdict(s, q("MON", "10:00", 121)) == "illegal"
    assert verdict(s, q("SUN", "10:00", 300)) == "legal"  # limit not in force on Sunday
    # only the time spent inside the window counts: 17:00 + 3 h -> 60 min inside
    assert verdict(s, q("MON", "17:00", 180)) == "legal"
    assert verdict(s, q("MON", "07:00", 180)) == "legal"  # 07:00-10:00 -> 120 min inside


def test_time_limit_all_day_when_no_hours():
    s = sign({"rule": "time_limited", "days": ALL, "limit_min": 120})  # "2 HOUR PARKING", no hours
    assert verdict(s, q("SUN", "03:00", 150)) == "illegal"
    assert verdict(s, q("SUN", "03:00", 90)) == "legal"


def test_metered_assumes_paid():
    limited = sign({"rule": "metered", "days": MON_SAT, "start": "08:00", "end": "20:00", "limit_min": 60})
    assert verdict(limited, q("TUE", "12:00", 60)) == "legal"
    assert verdict(limited, q("TUE", "12:00", 90)) == "illegal"
    unlimited = sign({"rule": "metered", "days": MON_SAT, "start": "08:00", "end": "20:00"})
    assert verdict(unlimited, q("TUE", "12:00", 300)) == "legal"


def test_unrestricted():
    assert verdict(sign({"rule": "unrestricted", "days": ALL}), q("FRI", "12:00", 600)) == "legal"


# --- overnight wraparound -------------------------------------------------------------------------


def test_overnight_window_2300_and_0200():
    s = sign({"rule": "no_parking", "days": ALL, "start": "18:00", "end": "08:00"})
    assert verdict(s, q("TUE", "23:00")) == "illegal"
    assert verdict(s, q("TUE", "02:00")) == "illegal"  # tail of Monday night's window
    assert verdict(s, q("TUE", "12:00")) == "legal"


def test_overnight_days_are_start_days():
    # "6PM-8AM MON-FRI": the Friday-night window runs into Saturday morning; Saturday night has none
    s = sign({"rule": "no_parking", "days": WEEKDAYS, "start": "18:00", "end": "08:00"})
    assert verdict(s, q("SAT", "02:00")) == "illegal"
    assert verdict(s, q("SAT", "23:00")) == "legal"
    assert verdict(s, q("MON", "02:00")) == "legal"  # Sunday night has no window
    assert verdict(s, q("MON", "20:00")) == "illegal"


def test_week_wraparound_sunday_night_to_monday():
    s = sign({"rule": "no_parking", "days": ["SUN"], "start": "22:00", "end": "06:00"})
    assert verdict(s, q("MON", "05:00")) == "illegal"
    assert verdict(s, q("SUN", "21:00", 120)) == "illegal"  # runs into the window


# --- boundary minutes -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "time,expected",
    [("09:59", "legal"), ("10:00", "illegal"), ("11:59", "illegal"), ("12:00", "legal"), ("12:01", "legal")],
)
def test_boundary_minutes(time, expected):
    s = sign({"rule": "no_parking", "days": ALL, "start": "10:00", "end": "12:00"})
    assert verdict(s, q("WED", time, 1)) == expected


def test_stay_running_into_restriction_is_illegal():
    # the agreed semantics: arrive Thu 07:30 for 60 min, street cleaning 08:00-10:00 Thursday
    s = sign({"rule": "no_parking", "days": ["THU"], "start": "08:00", "end": "10:00"})
    assert verdict(s, q("THU", "07:30", 60)) == "illegal"
    assert verdict(s, q("THU", "07:30", 30)) == "legal"  # leaves exactly at 08:00


def test_midnight_as_end_of_day():
    s = sign({"rule": "no_parking", "days": ALL, "start": "20:00", "end": "24:00"})
    assert verdict(s, q("MON", "23:59")) == "illegal"
    assert verdict(s, q("TUE", "00:00")) == "legal"


# --- day-of-week edges ----------------------------------------------------------------------------


def test_sunday_against_mon_sat():
    s = sign({"rule": "no_parking", "days": MON_SAT, "start": "08:00", "end": "18:00"})
    assert verdict(s, q("SUN", "12:00", 60)) == "legal"
    assert verdict(s, q("SAT", "12:00", 60)) == "illegal"


def test_saturday_stay_running_into_sunday_only_counts_saturday_window():
    s = sign({"rule": "no_parking", "days": ["SUN"], "start": "00:00", "end": "06:00"})
    assert verdict(s, q("SAT", "23:00", 30)) == "legal"
    assert verdict(s, q("SAT", "23:00", 90)) == "illegal"


# --- permits --------------------------------------------------------------------------------------


def test_permit_exempts_from_time_limit():
    # "2 HR PARKING 8AM-6PM MON-FRI, VEHICLES WITH DISTRICT 7 PERMITS EXEMPT"
    s = sign({"rule": "time_limited", "days": WEEKDAYS, "start": "08:00", "end": "18:00", "limit_min": 120, "district": "7"})
    assert verdict(s, q("TUE", "09:00", 240)) == "illegal"
    assert verdict(s, q("TUE", "09:00", 240, permit="7")) == "legal"
    assert verdict(s, q("TUE", "09:00", 240, permit="8")) == "illegal"


def test_permit_does_not_exempt_other_panels():
    s = sign(
        {"rule": "no_parking", "days": ["TUE"], "start": "08:00", "end": "10:00"},  # street cleaning
        {"rule": "time_limited", "days": WEEKDAYS, "start": "08:00", "end": "18:00", "limit_min": 120, "district": "7"},
    )
    v = evaluate(s, q("TUE", "09:00", 30, permit="7"))
    assert v.verdict == "illegal" and v.governing_panel == 0


# --- multi-panel priority -------------------------------------------------------------------------


def test_most_restrictive_panel_governs():
    s = sign(
        {"rule": "time_limited", "days": WEEKDAYS, "start": "07:00", "end": "19:00", "limit_min": 60},
        {"rule": "no_stopping", "days": WEEKDAYS, "start": "16:00", "end": "19:00", "tow_away": True},
    )
    v = evaluate(s, q("WED", "16:30", 120))  # violates both
    assert v.verdict == "illegal" and v.governing_panel == 1
    v = evaluate(s, q("WED", "10:00", 120))  # only the time limit
    assert v.verdict == "illegal" and v.governing_panel == 0
    assert verdict(s, q("WED", "10:00", 30)) == "legal"


def test_typical_la_stack():
    # anti-gridlock tow-away no stopping + 1 hr parking + street cleaning
    s = sign(
        {"rule": "no_stopping", "days": WEEKDAYS, "start": "07:00", "end": "09:00", "tow_away": True},
        {"rule": "no_stopping", "days": WEEKDAYS, "start": "16:00", "end": "19:00", "tow_away": True},
        {"rule": "time_limited", "days": MON_SAT, "start": "09:00", "end": "16:00", "limit_min": 60},
        {"rule": "no_parking", "days": ["THU"], "start": "10:00", "end": "12:00"},
    )
    assert evaluate(s, q("THU", "10:30", 30)).governing_panel == 3
    assert evaluate(s, q("THU", "15:30", 60)).governing_panel == 1  # runs into 16:00 no stopping
    assert evaluate(s, q("FRI", "13:00", 90)).governing_panel == 2
    assert verdict(s, q("FRI", "13:00", 45)) == "legal"
    assert verdict(s, q("SAT", "08:00", 30)) == "legal"


def test_tie_goes_to_higher_panel():
    s = sign(
        {"rule": "no_parking", "days": ["MON"], "start": "08:00", "end": "10:00"},
        {"rule": "no_parking", "days": ["MON"], "start": "09:00", "end": "11:00"},
    )
    assert evaluate(s, q("MON", "09:30", 10)).governing_panel == 0


# --- ambiguity, holidays, schema validation --------------------------------------------------------


def test_ambiguous_sign():
    s = sign({"rule": "no_parking", "days": ALL}, ambiguous=True)
    assert verdict(s, q("MON", "12:00")) == "ambiguous"


def test_holidays_ignored():
    s = sign({"rule": "time_limited", "days": ALL, "start": "08:00", "end": "18:00", "limit_min": 60, "except_holidays": True})
    assert verdict(s, q("MON", "10:00", 120)) == "illegal"


@pytest.mark.parametrize(
    "panel",
    [
        {"rule": "time_limited", "days": ALL},  # missing limit
        {"rule": "permit_only", "days": ALL},  # missing district
        {"rule": "no_parking", "days": ALL, "start": "08:00"},  # end missing
        {"rule": "no_parking", "days": ALL, "start": "8:00", "end": "10:00"},  # not HH:MM
        {"rule": "no_parking", "days": ALL, "start": "10:00", "end": "10:00"},
        {"rule": "no_parking", "days": []},
    ],
)
def test_invalid_panels_rejected(panel):
    with pytest.raises(ValidationError):
        Panel(order=0, **panel)


def test_sign_panel_orders_and_count():
    with pytest.raises(ValidationError):
        Sign(sign_id="t", panels=[Panel(order=1, rule="no_parking", days=ALL)])
    with pytest.raises(ValidationError):
        Sign(sign_id="t", n_panels=2, panels=[Panel(order=0, rule="no_parking", days=ALL)])
    assert Sign(sign_id="t", panels=[Panel(order=0, rule="no_parking", days=["TUE", "MON", "MON"])]).panels[0].days == ["MON", "TUE"]


def test_annotation_guide_examples_validate():
    import json
    import re
    from pathlib import Path

    guide = (Path(__file__).parents[1] / "ANNOTATION_GUIDE.md").read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", guide, flags=re.S)
    assert len(blocks) >= 3
    for block in blocks:
        Sign(sign_id="guide", panels=[Panel(**p) for p in json.loads(block)])
