"""Phase 0 go/no-go probe: sample ~100 Mapillary images from one area, download them, and
collect Mapillary's own detections and map features so they can be reviewed in
notebooks/01_phase0_go_no_go.ipynb.

    uv run python scripts/phase0_probe.py                 # Westwood defaults
    uv run python scripts/phase0_probe.py --bbox -118.455 34.055 -118.435 34.075 --out data/phase0_plan_box
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vlm_parking.mapillary import BBox, MapillaryClient

# Westwood Village, the Wilshire corridor, and residential permit streets to the south.
# Shifted south of the plan's box (lat 34.055–34.075), which is mostly UCLA campus north of Le Conte.
DEFAULT_BBOX = (-118.455, 34.045, -118.435, 34.065)

PARKING_HINTS = ("parking", "stopping", "standing", "tow", "loading", "permit")


def sample_images(images: list[dict], n: int, per_sequence: int, seed: int) -> list[dict]:
    """Random sample capped per capture sequence, so one dashcam drive can't dominate."""
    rng = random.Random(seed)
    by_seq: dict[str, list[dict]] = collections.defaultdict(list)
    for img in images:
        by_seq[img.get("sequence", img["id"])].append(img)
    pool = []
    for seq_imgs in by_seq.values():
        rng.shuffle(seq_imgs)
        pool.extend(seq_imgs[:per_sequence])
    rng.shuffle(pool)
    if len(pool) < n:
        print(f"warning: only {len(pool)} images after capping at {per_sequence}/sequence")
    return pool[:n]


def to_row(idx: int, img: dict) -> dict:
    captured = datetime.fromtimestamp(img["captured_at"] / 1000, tz=timezone.utc)
    lon, lat = img["geometry"]["coordinates"]
    return {
        "idx": idx,
        "image_id": img["id"],
        "captured_at": captured.isoformat(),
        "year": captured.year,
        "lat": lat,
        "lon": lon,
        "compass_angle": img.get("compass_angle"),
        "is_pano": img.get("is_pano"),
        "width": img.get("width"),
        "height": img.get("height"),
        "creator_username": img.get("creator", {}).get("username"),
        "sequence": img.get("sequence"),
        "source_url": f"https://www.mapillary.com/app/?pKey={img['id']}",
        "license": "CC BY-SA 4.0",
        "path": f"images/{img['id']}.jpg",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-sequence", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/phase0"))
    args = ap.parse_args()

    bbox = BBox(*args.bbox)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    client = MapillaryClient()

    print(f"Searching images in {bbox.param()} ...")
    images = client.search_images(bbox)
    years = collections.Counter(datetime.fromtimestamp(i["captured_at"] / 1000, tz=timezone.utc).year for i in images)
    population = {
        "bbox": bbox.param(),
        "n_images": len(images),
        "n_sequences": len({i.get("sequence") for i in images}),
        "n_creators": len({i.get("creator", {}).get("username") for i in images}),
        "n_pano": sum(bool(i.get("is_pano")) for i in images),
        "by_year": dict(sorted(years.items())),
    }
    print(json.dumps(population, indent=1))

    sample = sample_images(images, args.n, args.per_sequence, args.seed)
    rows = [to_row(idx, img) for idx, img in enumerate(sample)]

    print(f"Downloading {len(sample)} images and their detections ...")
    detections = {}
    for img in tqdm(sample):
        path = out / "images" / f"{img['id']}.jpg"
        if not path.exists():
            client.download(img["thumb_2048_url"], path)
        detections[img["id"]] = client.detections(img["id"])
    pd.DataFrame(rows).to_csv(out / "sample.csv", index=False)
    (out / "detections.json").write_text(json.dumps(detections))

    print("Searching map features ...")
    features = client.map_features(bbox)
    (out / "map_features.json").write_text(json.dumps(features))
    values = collections.Counter(f["object_value"] for f in features)
    parking_values = {v: c for v, c in values.most_common() if any(h in v for h in PARKING_HINTS)}
    print(f"{len(features)} map features, {len(values)} distinct object_values; parking-related:")
    for v, c in parking_values.items():
        print(f"  {c:>5}  {v}")

    (out / "population.json").write_text(json.dumps(population, indent=1))
    print(f"Done. Review in notebooks/01_phase0_go_no_go.ipynb (data in {out}/)")


if __name__ == "__main__":
    main()
