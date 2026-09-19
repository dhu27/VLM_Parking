"""Structured transcription of a parking sign, plus the query and verdict types.

Conventions (full rules in ANNOTATION_GUIDE.md):
- Times are 24-hour "HH:MM". "24:00" means midnight at the end of the day.
- start/end both null means the rule applies all day.
- A window with end <= start wraps past midnight (18:00-08:00). Its `days` are the days on which the
  window *starts*.
- `district` on a permit_only panel means a permit for that district is required. On any other panel
  it means vehicles with that district's permit are exempt from the panel.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Day = Literal["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
DAYS: tuple[Day, ...] = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")

Rule = Literal[
    "no_stopping",
    "no_standing",
    "no_parking",
    "passenger_loading",
    "commercial_loading",
    "permit_only",
    "time_limited",
    "metered",
    "unrestricted",
]

_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$|^24:00$")


def to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class Panel(BaseModel):
    order: int = Field(ge=0, description="Top to bottom, 0-indexed")
    rule: Rule
    days: list[Day] = Field(min_length=1, description="Explicit list, never a range string")
    start: str | None = None
    end: str | None = None
    limit_min: int | None = Field(default=None, gt=0, description="For time_limited / metered / loading")
    district: str | None = None
    except_holidays: bool = False
    tow_away: bool = False

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str | None) -> str | None:
        if v is not None and not _HHMM.match(v):
            raise ValueError(f"time must be HH:MM (24h), got {v!r}")
        return v

    @field_validator("days")
    @classmethod
    def _unique_sorted_days(cls, v: list[Day]) -> list[Day]:
        return sorted(set(v), key=DAYS.index)

    @model_validator(mode="after")
    def _consistent(self) -> Panel:
        if (self.start is None) != (self.end is None):
            raise ValueError("fill in both Start and End, or tick all day")
        if self.start is not None and self.start == self.end:
            raise ValueError("Start equals End; tick all day instead")
        if self.start == "24:00":
            raise ValueError("start cannot be 24:00; use 00:00")
        if self.rule == "time_limited" and self.limit_min is None:
            raise ValueError("a time_limited panel needs Limit (min), e.g. 120 for 2 HOUR PARKING")
        if self.rule == "permit_only" and not self.district:
            raise ValueError("a permit_only panel needs a District")
        return self


class ImageQuality(BaseModel):
    angle: Literal["frontal", "oblique", "severe"] = "frontal"
    glare: bool = False
    occlusion: bool = False
    legibility: Literal["clear", "hard"] = "clear"


class Sign(BaseModel):
    sign_id: str
    source: Literal["mapillary"] = "mapillary"
    panels: list[Panel] = Field(min_length=1)
    arrow: Literal["left", "right", "both", "none"] | None = None
    n_panels: int | None = None
    image_quality: ImageQuality = ImageQuality()
    ambiguous: bool = False
    provenance: Literal["gold_manual", "assisted_accepted", "assisted_corrected"] = "gold_manual"
    notes: str = ""

    @model_validator(mode="after")
    def _panels(self) -> Sign:
        orders = [p.order for p in self.panels]
        if sorted(orders) != list(range(len(orders))):
            raise ValueError(f"panel orders must be 0..{len(orders) - 1}, got {orders}")
        self.panels = sorted(self.panels, key=lambda p: p.order)
        if self.n_panels is None:
            self.n_panels = len(self.panels)
        elif self.n_panels != len(self.panels):
            raise ValueError(f"n_panels={self.n_panels} but {len(self.panels)} panels given")
        return self


class Query(BaseModel):
    day: Day
    time: str = Field(description="Arrival time, HH:MM")
    duration_min: int = Field(gt=0, le=24 * 60)
    permit_district: str | None = None  # the district permit the driver holds, if any

    @field_validator("time")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not _HHMM.match(v) or v == "24:00":
            raise ValueError(f"time must be HH:MM in 00:00-23:59, got {v!r}")
        return v


class Verdict(BaseModel):
    verdict: Literal["legal", "illegal", "ambiguous"]
    governing_panel: int | None = None  # order of the panel that decides it; None when legal or ambiguous
    reason: str = ""
