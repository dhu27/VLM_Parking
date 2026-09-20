"""The prompt and answer schema, shared by both conditions and all models.

One prompt, fixed across models and conditions (plan §7.4): only the sign representation differs.
  condition A: the sign crop (image)
  condition B: the ground-truth transcription rendered as text

No LA-convention primer: whether a primer helps is an extension experiment, so the MVP leaves it out.

Answer fields are generated in order, so `reason` comes before `verdict`: the model writes its
justification before committing to an answer, and the text feeds the error taxonomy in Phase 6.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from vlm_parking.schema import DAYS, Panel, Query, Sign

DAY_NAMES = {"MON": "Monday", "TUE": "Tuesday", "WED": "Wednesday", "THU": "Thursday", "FRI": "Friday", "SAT": "Saturday", "SUN": "Sunday"}


class Answer(BaseModel):
    """The response schema for guided JSON decoding."""

    reason: str = Field(max_length=300, description="One or two sentences: which posted rule decides this.")
    governing_panel: int | None = Field(description="0-based index from the top of the sign of the panel that decides it; null if parking is legal.")
    verdict: Literal["legal", "illegal"] = Field(description="Is parking legal for the whole stay?")


TASK = """You are given a Los Angeles parking sign and a parking question.

Decide whether parking is legal for the ENTIRE stay described in the question.
- The stay must be legal from arrival until departure; if any part of it breaks a posted rule, it is illegal.
- A restriction that starts exactly when the driver leaves does not make the stay illegal.
- Panels are numbered from the top of the sign, starting at 0.
- Assume the driver pays any meter, the day is not a holiday, and no rule applies except what the sign posts.

Answer with JSON only: {"reason": "...", "governing_panel": <int or null>, "verdict": "legal" | "illegal"}."""


def render_query(q: Query) -> str:
    permit = f"a residential permit for district {q.permit_district}" if q.permit_district else "no parking permit"
    hours, minutes = divmod(q.duration_min, 60)
    length = " ".join(part for part in (f"{hours} hour{'s' if hours != 1 else ''}" if hours else "", f"{minutes} minutes" if minutes else "") if part)
    return (
        f"Question: A driver arrives on {DAY_NAMES[q.day]} at {q.time} and wants to park for {length}. "
        f"The driver has {permit}. Is parking legal?"
    )


def render_panel(p: Panel) -> str:
    rule = {
        "no_parking": "No parking",
        "no_stopping": "No stopping",
        "no_standing": "No standing",
        "passenger_loading": "Passenger loading only",
        "commercial_loading": "Commercial loading only",
        "permit_only": "Permit parking only",
        "time_limited": "Time-limited parking",
        "metered": "Metered parking",
        "unrestricted": "Parking allowed",
    }[p.rule]
    bits = [rule]
    if p.rule in ("time_limited", "metered") and p.limit_min:
        bits.append(f"limit {p.limit_min} minutes")
    bits.append("all day" if p.start is None else f"{p.start}-{p.end}" + (" (overnight, into the next day)" if p.end <= p.start else ""))
    bits.append("every day" if len(p.days) == 7 else ", ".join(DAY_NAMES[d] for d in sorted(p.days, key=DAYS.index)))
    if p.district:
        bits.append(f"district {p.district} permit holders required" if p.rule == "permit_only" else f"district {p.district} permit holders exempt")
    if p.tow_away:
        bits.append("tow-away")
    if p.except_holidays:
        bits.append("except holidays")
    return "; ".join(bits)


def render_sign(sign: Sign) -> str:
    """Condition B's sign representation: the ground-truth transcription as text."""
    lines = [f"Panel {p.order}: {render_panel(p)}" for p in sign.panels]
    return "The sign has the following panels, top to bottom:\n" + "\n".join(lines)


def build_prompt(query: Query, sign: Sign | None = None) -> str:
    """The text part of the prompt. Condition A passes sign=None and supplies the image instead."""
    sign_part = "The sign is shown in the image." if sign is None else render_sign(sign)
    return f"{TASK}\n\n{sign_part}\n\n{render_query(query)}"


def messages(query: Query, sign: Sign | None = None) -> list[dict]:
    """Chat messages for vLLM. The image goes first so the per-sign prefix can be cached across queries."""
    if sign is None:
        content = [{"type": "image"}, {"type": "text", "text": build_prompt(query)}]
    else:
        content = [{"type": "text", "text": build_prompt(query, sign)}]
    return [{"role": "user", "content": content}]
