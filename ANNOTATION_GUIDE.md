# Annotation Guide

How to transcribe an LA parking sign into the structured schema in [`src/vlm_parking/schema.py`](src/vlm_parking/schema.py). Decide every case with these rules, and never deviate. When a new kind of case comes up, add a rule here *before* continuing.

This guide is also the system prompt for the pre-fill labeler model, so it's written to be followed literally.

---

## 1. Accept or reject the sign first

**Reject the whole sign** (and log the reason) if any of these hold:

| Reason code | When |
|---|---|
| `illegible_panel` | Any panel can't be fully read: text, times, days, or limits. |
| `illegible_fine_print` | The main text is readable but fine print that could change the rule isn't (e.g. a blurred "EXCEPT SUNDAYS" line). |
| `cut_off` | Part of the stack is outside the crop or the photo. |
| `multiple_signs` | The crop mixes panels from two unrelated signs and can't be re-cropped cleanly. |
| `unsupported_condition` | The sign has a condition the schema can't express: school days, specific dates, week-of-month ("1ST & 3RD TUE"), events, "when posted", vehicle types other than permits (commercial vehicles only, oversize vehicles), or a temporary / paper sign. |
| `not_parking` | Not a parking regulation sign after all. |

When in doubt about legibility, reject. The dataset's scope is signs a careful human can *fully* read.

---

## 2. Sign-level fields

| Field | Rule |
|---|---|
| `panels` | One entry per panel, **top to bottom**, `order` starting at 0. A panel is one rule: usually one physical plate, but a plate with two separate rules (e.g. two unrelated time windows with different rules) is two panels. |
| `arrow` | Record `left`, `right`, `both`, or `none` for the arrow on the sign. It is **not** used to change the rules. |
| `n_panels` | Leave blank; it's computed from `panels`. |
| `ambiguous` | `true` only if two panels genuinely conflict and the conflict can't be resolved by the priority rule below. Still transcribe every panel. |
| `image_quality` | `angle`: frontal / oblique / severe. `glare`, `occlusion`: true if present at all. `legibility`: `clear`, or `hard` if you could read it only with effort. |
| `notes` | Anything unusual, in a few words. |

---

## 3. Panel fields

### `rule`

| Sign text | `rule` |
|---|---|
| NO STOPPING (ANY TIME / hours) | `no_stopping` |
| NO STANDING | `no_standing` |
| NO PARKING (including the red P-with-slash symbol, and STREET CLEANING / SWEEPING) | `no_parking` |
| PASSENGER LOADING ONLY / white-zone style | `passenger_loading` |
| LOADING ZONE, COMMERCIAL LOADING, TRUCK LOADING | `commercial_loading` |
| PERMIT PARKING ONLY, DISTRICT n | `permit_only` |
| n HOUR / n MIN PARKING | `time_limited` |
| PAY TO PARK / METER / pay station | `metered` |
| An explicit "PARKING" panel with no restriction | `unrestricted` |

### `days`

- Always an **explicit list**: `["MON","TUE","WED","THU","FRI"]`, never `"MON-FRI"`.
- **No days listed** → all seven days.
- "EXCEPT SUNDAY" → list the six included days. "NIGHTLY" / "DAILY" / "ANY TIME" → all seven days.
- For an overnight window, `days` are the days on which the window **starts** ("6PM–8AM MON–FRI" means the Friday window runs into Saturday morning).

### `start`, `end`

- 24-hour `"HH:MM"`. `12 NOON` → `"12:00"`. `MIDNIGHT` at the end of a window → `"24:00"`; at the start → `"00:00"`.
- **No hours listed** (e.g. "2 HOUR PARKING", "NO PARKING ANY TIME") → both `null`, meaning all day.
- Overnight windows are **one panel** with `start > end` ("6PM TO 8AM" → `18:00`, `08:00`). Don't split them.
- A panel with **two time windows and the same rule** ("7AM–9AM 4PM–7PM") becomes two panels with the same rule, one per window, in reading order.

### `limit_min`

- Required for `time_limited`: "2 HOUR" → `120`, "30 MIN" → `30`.
- Optional for `metered` and loading rules; set it only if the sign states a limit.

### `district`

- On `permit_only`: the district number or name required to park.
- On any other rule: the district whose permit holders are **exempt** ("VEHICLES WITH DISTRICT 7 PERMITS EXEMPT" under a time limit → `district: "7"` on that time-limit panel).
- Record the district exactly as printed, without "No." or "#" ("DISTRICT No. 7" → `"7"`).

### `except_holidays`, `tow_away`

- `except_holidays: true` if the panel says holidays are excepted. Don't list dates.
- `tow_away: true` if the tow-away warning belongs to that panel ("TOW-AWAY NO STOPPING 7–9AM"). It's a flag on that panel, **not** a separate panel.

---

## 4. Worked examples

**Street cleaning + time limit**
```
NO PARKING 8AM TO 10AM THURSDAY STREET CLEANING
2 HOUR PARKING 8AM TO 6PM EXCEPT SUNDAY
```
```json
[
  {"order": 0, "rule": "no_parking", "days": ["THU"], "start": "08:00", "end": "10:00"},
  {"order": 1, "rule": "time_limited", "days": ["MON","TUE","WED","THU","FRI","SAT"], "start": "08:00", "end": "18:00", "limit_min": 120}
]
```

**Anti-gridlock tow-away stack**
```
ANTI-GRIDLOCK ZONE  TOW-AWAY NO STOPPING 7AM–9AM 4PM–7PM MON–FRI
1 HOUR PARKING 9AM–4PM MON–SAT
```
```json
[
  {"order": 0, "rule": "no_stopping", "days": ["MON","TUE","WED","THU","FRI"], "start": "07:00", "end": "09:00", "tow_away": true},
  {"order": 1, "rule": "no_stopping", "days": ["MON","TUE","WED","THU","FRI"], "start": "16:00", "end": "19:00", "tow_away": true},
  {"order": 2, "rule": "time_limited", "days": ["MON","TUE","WED","THU","FRI","SAT"], "start": "09:00", "end": "16:00", "limit_min": 60}
]
```

**Overnight permit district**
```
NO PARKING 2AM TO 6AM NIGHTLY  VEHICLES WITH DISTRICT No. 36 PERMITS EXEMPT
```
```json
[{"order": 0, "rule": "no_parking", "days": ["MON","TUE","WED","THU","FRI","SAT","SUN"], "start": "02:00", "end": "06:00", "district": "36"}]
```

---

## 5. Evaluator semantics

These define the correct answer to a query. They're implemented in [`src/vlm_parking/evaluator.py`](src/vlm_parking/evaluator.py) and tested in [`tests/test_evaluator.py`](tests/test_evaluator.py).

1. **A query is a stay:** arrival day and time plus a duration. **The whole stay must be legal.** Arriving Thursday 7:30am for 60 minutes under "NO PARKING 8–10AM THURSDAY" is illegal.
2. **Windows are half-open:** "10AM–12PM" is in force at 10:00 and over at 12:00. Leaving exactly when a restriction starts is legal.
3. **Prohibitions** (no stopping, no standing, no parking, loading zones) are violated by any overlap with the stay.
4. **permit_only** is violated by any overlap unless the driver holds that district's permit.
5. **Time limits** count only the time spent inside the window: under "2 HR 8AM–6PM", arriving at 5pm for 3 hours is legal (60 minutes inside).
6. **Meters are assumed paid.** A metered panel only matters if it states a time limit.
7. **Permit exemptions** apply only to the panel they're written on.
8. **Queries never fall on holidays**, so `except_holidays` never changes an answer.
9. **Arrows are ignored.**
10. **Most restrictive panel governs** when several are violated: no stopping > no standing > no parking > loading > permit-only > time limit. Ties go to the higher panel. That panel is the `governing_panel` for rule attribution.
11. **Ambiguous signs** (`ambiguous: true`) always evaluate to `ambiguous` and are excluded from query generation.
