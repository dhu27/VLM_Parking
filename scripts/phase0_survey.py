"""Phase 0 area survey: compare Mapillary coverage across the plan's target neighbourhoods without
downloading images. For each area, report the image population and a legibility proxy: the number
of large front-facing sign detections per 100 sampled images.

    uv run python scripts/phase0_survey.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from phase0_probe import sample_images
from vlm_parking.mapillary import BBox, MapillaryClient, decode_detection_geometry, ring_bbox

# Approximate boxes [VERIFY on a map]; each is (min_lon, min_lat, max_lon, max_lat)
AREAS = {
    "westwood": (-118.455, 34.045, -118.435, 34.065),
    "koreatown": (-118.310, 34.052, -118.290, 34.070),
    "downtown": (-118.258, 34.035, -118.225, 34.055),  # Historic Core + Arts District
    "hollywood": (-118.345, 34.088, -118.285, 34.105),  # Hollywood + East Hollywood
    "venice_mar_vista": (-118.480, 33.985, -118.420, 34.015),
}

THUMB = 2048  # detection sizes are reported on the 2048-px thumbnail scale, as reviewed in notebook 01
SIGN = "object--traffic-sign--front"


def sign_heights(client: MapillaryClient, img: dict) -> list[float]:
    """Heights (px, thumbnail scale) of front-facing sign detections in one image."""
    w, h = img.get("width") or THUMB, img.get("height") or THUMB
    scale = THUMB / max(w, h)
    heights = []
    for det in client.detections(img["id"]):
        if det["value"] == SIGN:
            for ring in decode_detection_geometry(det["geometry"], w, h):
                x0, y0, x1, y1 = ring_bbox(ring)
                heights.append((y1 - y0) * scale)
    return heights


def summarize(client: MapillaryClient, images: list[dict], n: int, seed: int) -> dict:
    sample = sample_images(images, n, per_sequence=2, seed=seed)
    per_image = [sign_heights(client, img) for img in tqdm(sample, leave=False)]
    flat = [h for hs in per_image for h in hs]
    k = len(sample) or 1
    return {
        "sampled": len(sample),
        "signs_per_100": round(100 * len(flat) / k, 1),
        "median_sign_px": round(float(pd.Series(flat).median()), 1) if flat else None,
        # images with at least one sign detection this tall: a rough upper bound on "legible sign" images
        "imgs_with_sign_ge_60px_per_100": round(100 * sum(any(h >= 60 for h in hs) for hs in per_image) / k, 1),
        "imgs_with_sign_ge_100px_per_100": round(100 * sum(any(h >= 100 for h in hs) for hs in per_image) / k, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recent-year", type=int, default=2023)
    ap.add_argument("--out", type=Path, default=Path("data/phase0/survey.json"))
    args = ap.parse_args()

    client = MapillaryClient()
    rows = []
    for name, box in AREAS.items():
        print(f"== {name}")
        images = client.search_images(BBox(*box))
        year = lambda i: datetime.fromtimestamp(i["captured_at"] / 1000, tz=timezone.utc).year
        recent = [i for i in images if year(i) >= args.recent_year]
        base = {
            "area": name,
            "images": len(images),
            "sequences": len({i.get("sequence") for i in images}),
            "recent_images": len(recent),
            "recent_sequences": len({i.get("sequence") for i in recent}),
            "hi_res_share": round(sum(max(i.get("width") or 0, i.get("height") or 0) >= 3000 for i in images) / max(1, len(images)), 2),
        }
        for pool_name, pool in [("all", images), (f"{args.recent_year}+", recent)]:
            rows.append({**base, "pool": pool_name, **summarize(client, pool, args.n, args.seed)})
            print(json.dumps(rows[-1]))

    args.out.write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows).set_index(["area", "pool"])
    pd.set_option("display.width", 200)
    print(df.to_string())


if __name__ == "__main__":
    main()
