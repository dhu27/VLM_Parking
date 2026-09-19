"""Cheap filters that decide which Mapillary images are worth a human look.

Stages (all before any image is downloaded):
  1. metadata: resolution
  2. location: drop images on or beside freeways (OpenStreetMap via Overpass)
  3. detections: keep images with at least one large, upright, uncut front-facing sign detection

Capture year is deliberately not a filter: ground truth is transcribed from the image itself,
so an old sign is as valid as a new one. Year is kept for stratification.
"""

from __future__ import annotations

import collections
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests

from vlm_parking.mapillary import BBox, MapillaryClient, decode_detection_geometry, ring_bbox

# Approximate target areas [VERIFY on a map]: (min_lon, min_lat, max_lon, max_lat)
AREAS = {
    "westwood": (-118.455, 34.045, -118.435, 34.065),
    "koreatown": (-118.310, 34.052, -118.290, 34.070),
    "downtown": (-118.258, 34.035, -118.225, 34.055),  # Historic Core + Arts District
    "hollywood": (-118.345, 34.088, -118.285, 34.105),  # Hollywood + East Hollywood
    "venice_mar_vista": (-118.480, 33.985, -118.420, 34.015),
}

THUMB = 2048  # sizes are expressed on the 2048-px thumbnail scale unless noted

SIGN_VALUES = {"object--traffic-sign--front", "object--traffic-sign--information-parking"}
PARKING_PREFIXES = ("regulatory--no-parking", "regulatory--no-stopping", "regulatory--parking", "information--parking")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


@dataclass
class Criteria:
    min_long_side: int = 1920  # native pixels; drops 1280x720 dashcams and tiny thumbnails
    min_sign_px: float = 60  # detection height on the 2048 scale; deliberately loose, tighten after review
    min_aspect: float = 1.1  # height / width; parking signs are portrait
    edge_margin: float = 0.01  # fraction of the frame; boxes this close to an edge are likely cut off
    overhead_frac: float = 0.2  # boxes entirely in the top 20% of the frame are usually overhead signs
    freeway_buffer_m: float = 30


def is_sign_value(value: str) -> bool:
    return value in SIGN_VALUES or value.startswith(PARKING_PREFIXES)


def metadata_ok(img: dict, c: Criteria) -> bool:
    return max(img.get("width") or 0, img.get("height") or 0) >= c.min_long_side


def sign_candidates(detections: list[dict], width: int, height: int, c: Criteria) -> list[dict]:
    """Sign detections in one image that pass the size / shape / position checks."""
    scale = THUMB / max(width, height)
    out = []
    for det in detections:
        if not is_sign_value(det["value"]):
            continue
        for ring in decode_detection_geometry(det["geometry"], width, height):
            x0, y0, x1, y1 = ring_bbox(ring)
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h * scale < c.min_sign_px or h / w < c.min_aspect:
                continue
            mx, my = c.edge_margin * width, c.edge_margin * height
            if x0 < mx or y0 < my or x1 > width - mx or y1 > height - my:
                continue
            if y1 < c.overhead_frac * height:
                continue
            out.append({"id": det["id"], "value": det["value"], "bbox": [x0, y0, x1, y1], "height_2048": h * scale})
    return out


# --- freeway filter -----------------------------------------------------------------------------


FREEWAY_CACHE = Path("data/phase0/cache")


def load_freeways(bbox: BBox, pad_deg: float = 0.002, retries: int = 6) -> list[list[tuple[float, float]]]:
    """Motorway centrelines (lon, lat) in and around bbox from OpenStreetMap, cached on disk.
    The public Overpass server rate-limits (429) and times out (504) under load, so retry with backoff."""
    cache = FREEWAY_CACHE / f"freeways_{bbox.param().replace(',', '_')}.json"
    if cache.exists():
        return [[tuple(p) for p in line] for line in json.loads(cache.read_text())]
    s, w, n, e = bbox.min_lat - pad_deg, bbox.min_lon - pad_deg, bbox.max_lat + pad_deg, bbox.max_lon + pad_deg
    query = f'[out:json][timeout:60];way["highway"~"^(motorway|motorway_link)$"]({s},{w},{n},{e});out geom;'
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, headers={"User-Agent": "vlm-parking-research/0.1"}, timeout=90)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(min(120, 10 * 2**attempt) + random.random() * 5)
    lines = [[(p["lon"], p["lat"]) for p in way["geometry"]] for way in resp.json()["elements"]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(lines))
    return lines


def _dist_m(lon: float, lat: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance in metres from a point to segment ab, in a local equirectangular projection."""
    kx = 111_320 * math.cos(math.radians(lat))
    ky = 110_540
    px, py = lon * kx, lat * ky
    ax, ay, bx, by = a[0] * kx, a[1] * ky, b[0] * kx, b[1] * ky
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _point_dist_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return _dist_m(lon1, lat1, (lon2, lat2), (lon2, lat2))


def near_freeway(lon: float, lat: float, freeways: list[list[tuple[float, float]]], buffer_m: float) -> bool:
    deg = buffer_m / 100_000 * 2  # cheap bounding-box reject before exact distance
    for line in freeways:
        for a, b in zip(line, line[1:]):
            if min(a[0], b[0]) - deg > lon or max(a[0], b[0]) + deg < lon:
                continue
            if min(a[1], b[1]) - deg > lat or max(a[1], b[1]) + deg < lat:
                continue
            if _dist_m(lon, lat, a, b) <= buffer_m:
                return True
    return False


# --- sampling -----------------------------------------------------------------------------------


def sample_images(images: list[dict], n: int, per_sequence: int, seed: int) -> list[dict]:
    """Random sample capped per capture sequence, so one drive can't dominate."""
    rng = random.Random(seed)
    by_seq: dict[str, list[dict]] = collections.defaultdict(list)
    for img in images:
        by_seq[img.get("sequence") or img["id"]].append(img)
    pool = []
    for seq_imgs in by_seq.values():
        seq_imgs = seq_imgs[:]
        rng.shuffle(seq_imgs)
        pool.extend(seq_imgs[:per_sequence])
    rng.shuffle(pool)
    return pool[:n]


def spatial_dedup(images: list[dict], min_dist_m: float = 15) -> list[dict]:
    """Drop images within min_dist_m of an already-kept image from the same sequence
    (consecutive frames of one drive usually show the same sign)."""
    kept: list[dict] = []
    for img in images:
        lon, lat = img["geometry"]["coordinates"]
        same_spot = any(
            k.get("sequence") == img.get("sequence") and _point_dist_m(lon, lat, *k["geometry"]["coordinates"]) < min_dist_m
            for k in kept
        )
        if not same_spot:
            kept.append(img)
    return kept


def fetch_detections(client: MapillaryClient, image_ids: list[str], workers: int = 8) -> dict[str, list[dict]]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(image_ids, pool.map(client.detections, image_ids)))


def screen(
    client: MapillaryClient, images: list[dict], freeways: list, c: Criteria, n_screen: int, seed: int
) -> tuple[dict, list[dict], dict[str, list[dict]]]:
    """Run all pre-screen stages. Returns (funnel counts, passing images with their candidates, raw detections)."""
    funnel = {"images": len(images)}
    pool = [i for i in images if metadata_ok(i, c)]
    funnel["metadata_ok"] = len(pool)
    pool = [i for i in pool if not near_freeway(*i["geometry"]["coordinates"], freeways, c.freeway_buffer_m)]
    funnel["off_freeway"] = len(pool)

    # Spread the detection budget across sequences: cap per sequence so it sums to ~n_screen
    n_seq = max(1, len({i.get("sequence") for i in pool}))
    per_seq = max(2, math.ceil(n_screen / n_seq))
    screened = sample_images(pool, n_screen, per_seq, seed)
    funnel["screened"] = len(screened)

    detections = fetch_detections(client, [i["id"] for i in screened])
    passing = []
    for img in screened:
        cands = sign_candidates(detections[img["id"]], img["width"], img["height"], c)
        if cands:
            passing.append({**img, "candidates": cands})
    funnel["passing"] = len(passing)
    funnel["passing_sequences"] = len({i.get("sequence") for i in passing})
    funnel["pass_rate"] = round(len(passing) / max(1, len(screened)), 3)
    funnel["est_passing_in_area"] = round(funnel["pass_rate"] * funnel["off_freeway"])
    return funnel, passing, detections
