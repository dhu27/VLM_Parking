"""Phase 1 collection helpers: stack merging, OCR keyword filter, and duplicate-sign clustering."""

from __future__ import annotations

import difflib
import re

from vlm_parking.prescreen import THUMB, _point_dist_m

# --- stack merging ------------------------------------------------------------------------------


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe_boxes(boxes: list[list[float]], thresh: float = 0.6) -> list[list[float]]:
    """Mapillary often returns the same panel twice; keep the larger of heavily overlapping boxes."""
    kept: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(iou(b, k) < thresh for k in kept):
            kept.append(b)
    return kept


def same_stack(a: list[float], b: list[float]) -> bool:
    """Two panels belong to one stack if they overlap horizontally and sit close vertically."""
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    if x_overlap < 0.5 * min(a[2] - a[0], b[2] - b[0]):
        return False
    gap = max(a[1], b[1]) - min(a[3], b[3])  # negative when they overlap vertically
    return gap < 0.5 * min(a[3] - a[1], b[3] - b[1])


def merge_stacks(boxes: list[list[float]]) -> list[dict]:
    """Group panel boxes into stacks (connected components under same_stack).
    Returns [{"bbox": union box, "n_panels": number of detected panels, "members": [indices]}]."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if same_stack(boxes[i], boxes[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    stacks = []
    for members in groups.values():
        bs = [boxes[i] for i in members]
        union = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
        stacks.append({"bbox": union, "n_panels": len(members), "members": members})
    return stacks


def padded(box: list[float], width: int, height: int, pad: float = 0.12) -> tuple[int, int, int, int]:
    w, h = box[2] - box[0], box[3] - box[1]
    return (
        int(max(0, box[0] - pad * w)),
        int(max(0, box[1] - pad * h)),
        int(min(width, box[2] + pad * w)),
        int(min(height, box[3] + pad * h)),
    )


def scale_box(box: list[float], native_w: int, native_h: int, w: int, h: int) -> list[float]:
    sx, sy = w / native_w, h / native_h
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


# --- OCR keyword filter -------------------------------------------------------------------------

# Phrases that on their own mark a parking-regulation sign (matched on uppercase text with spaces removed,
# because OCR often drops spaces: "2HOUR", "8AM10AM").
STRONG = [
    "PARKING", "NOSTOPPING", "NOSTANDING", "TOWAWAY", "TOW-AWAY", "STREETCLEANING", "STREETSWEEPING",
    "PERMIT", "DISTRICT", "LOADING", "PASSENGER", "HOURPARK", "HRPARK", "MINUTE", "METER", "VIOLATORS",
    "ANTI-GRIDLOCK", "ANTIGRIDLOCK",
]
# Weaker evidence: times and days. Two or more of these also count (a no-parking pictogram sign
# may only have "8AM TO 10AM THURSDAY" as text).
WEAK = ["AM", "PM", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN", "NOON", "MIDNIGHT", "HOUR", "HR"]
# Text that marks a sign as clearly something else, when no parking keyword is present.
NEGATIVE = [
    "SPEED", "LIMIT", "SCHOOL", "DONOTBLOCK", "ONLY", "ONEWAY", "YIELD", "STOP", "BUSES", "AHEAD", "XING",
    "METRO", "DASH", "BIKE", "LANE", "TURN", "ENTER", "PED", "CROSSING", "END",
]
# Single words OCR often mangles ("FARKING", "AOUR"); matched fuzzily, token by token.
FUZZY = ["PARKING", "STOPPING", "STANDING", "TOWAWAY", "CLEANING", "SWEEPING", "PERMIT", "DISTRICT", "LOADING", "HOUR"]

_TIME = re.compile(r"\d{1,2}(:\d{2})?(AM|PM)")


def _fuzzy_hits(texts: list[str], min_ratio: float = 0.75) -> list[str]:
    tokens = {t for text in texts for t in re.findall(r"[A-Z0-9-]{4,}", text.upper())}
    return sorted({k for k in FUZZY for t in tokens if difflib.SequenceMatcher(None, t.replace("-", ""), k).ratio() >= min_ratio})


def parking_text_score(texts: list[str]) -> dict:
    """Classify OCR output into parking / other / unknown.

    OCR misses many genuine parking signs (faded, small, low contrast), so it is only trusted to
    *reject* crops it can clearly read as something else ("other"). "unknown" crops (including no
    text at all) go on to human triage. Using OCR as a positive gate would also bias the dataset
    toward signs that are easy to read.
    """
    joined = "".join(t.upper().replace(" ", "") for t in texts)
    strong = [k for k in STRONG if k in joined] + [f"~{k}" for k in _fuzzy_hits(texts)]
    weak = {k for k in WEAK if k in joined}
    times = len(_TIME.findall(joined))
    negative = [k for k in NEGATIVE if k in joined]
    if strong or (len(weak) + times >= 2 and not negative):
        status = "parking"
    elif negative:
        status = "other"
    else:
        status = "unknown"
    return {"status": status, "keep": status != "other", "strong": strong, "weak": sorted(weak), "n_times": times, "negative": negative}


# --- duplicate physical signs -------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Z0-9]{2,}", text.upper()))


def text_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def cluster_duplicates(
    items: list[dict], radius_m: float = 25, max_phash: int = 10, min_text_sim: float = 0.6
) -> list[int]:
    """Assign a cluster id to each crop; crops of the same physical sign share one.
    Two crops match if their cameras were within radius_m and either the crops look alike (perceptual
    hash distance) or their OCR text largely agrees. items need: lon, lat, phash (int), ocr_text."""
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: items[i]["lat"])
    deg = radius_m / 110_540
    for a_pos, i in enumerate(order):
        for j in order[a_pos + 1 :]:
            if items[j]["lat"] - items[i]["lat"] > deg:
                break
            a, b = items[i], items[j]
            if _point_dist_m(a["lon"], a["lat"], b["lon"], b["lat"]) > radius_m:
                continue
            if bin(a["phash"] ^ b["phash"]).count("1") <= max_phash or text_similarity(a["ocr_text"], b["ocr_text"]) >= min_text_sim:
                parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def thumb_height_to_native(h_2048: float, width: int, height: int) -> float:
    return h_2048 * max(width, height) / THUMB
