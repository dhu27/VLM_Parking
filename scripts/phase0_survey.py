"""Phase 0 area survey: run the pre-screen (resolution, freeway, sign-detection checks) on a sample
of images from each target area, without downloading any images, and compare the yield.

    uv run python scripts/phase0_survey.py                      # all areas
    uv run python scripts/phase0_survey.py --areas koreatown downtown --n-screen 800

Writes data/phase0/survey/<area>.json (funnel + passing images) and prints a comparison table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from vlm_parking.mapillary import BBox, MapillaryClient
from vlm_parking.prescreen import AREAS, Criteria, load_freeways, screen

OUT = Path("data/phase0/survey")
CACHE = Path("data/phase0/cache")


def area_images(client: MapillaryClient, name: str, refresh: bool) -> list[dict]:
    """All image records in an area, cached on disk (thumbnail URLs in the cache expire; metadata doesn't)."""
    path = CACHE / f"{name}_images.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    images = client.search_images(BBox(*AREAS[name]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(images))
    return images


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--areas", nargs="+", default=list(AREAS), choices=list(AREAS))
    ap.add_argument("--n-screen", type=int, default=500, help="images per area to fetch detections for")
    ap.add_argument("--min-sign-px", type=float, default=Criteria.min_sign_px)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true", help="re-run the image search instead of using the cache")
    args = ap.parse_args()

    criteria = Criteria(min_sign_px=args.min_sign_px)
    client = MapillaryClient()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.areas:
        print(f"== {name}: searching ...", flush=True)
        images = area_images(client, name, args.refresh)
        freeways = load_freeways(BBox(*AREAS[name]))
        print(f"   {len(images):,} images, {len(freeways)} freeway segments; screening ...", flush=True)
        funnel, passing, _ = screen(client, images, freeways, criteria, args.n_screen, args.seed)

        year = lambda i: datetime.fromtimestamp(i["captured_at"] / 1000, tz=timezone.utc).year
        funnel["passing_2023plus"] = sum(year(i) >= 2023 for i in passing)
        funnel["passing_median_sign_px"] = (
            round(float(pd.Series([max(c["height_2048"] for c in i["candidates"]) for i in passing]).median()), 1)
            if passing else None
        )
        (OUT / f"{name}.json").write_text(
            json.dumps({"area": name, "bbox": AREAS[name], "criteria": dataclasses.asdict(criteria), "funnel": funnel, "passing": passing})
        )
        rows.append({"area": name, **funnel})
        print("   " + json.dumps(funnel), flush=True)

    df = pd.DataFrame(rows).set_index("area")
    pd.set_option("display.width", 250)
    print("\n" + df.to_string())
    df.to_csv(OUT / "summary.csv")


if __name__ == "__main__":
    main()
