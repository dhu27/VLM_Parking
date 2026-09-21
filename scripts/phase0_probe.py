"""Phase 0 go/no-go probe: build a ~100-image review sample, download it, and save Mapillary's
detections alongside it, so the images can be eyeballed for legible parking signs.

Two modes:
  --prescreen (recommended): sample from images that passed the pre-screen in
      scripts/phase0_survey.py (run that first for the area).
  default: a plain random sample of the area, as a no-filter baseline.

    uv run python scripts/phase0_probe.py --area koreatown --prescreen
    uv run python scripts/phase0_probe.py --area westwood            # random baseline
"""

from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vlm_parking.mapillary import BBox, MapillaryClient
from vlm_parking.prescreen import AREAS, fetch_detections, sample_images, spatial_dedup

PARKING_HINTS = ("parking", "stopping", "standing", "tow", "loading", "permit")


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
        "n_candidates": len(img.get("candidates", [])),
        "max_sign_px": round(max((c["height_2048"] for c in img.get("candidates", [])), default=0), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", choices=list(AREAS), default="westwood")
    ap.add_argument("--prescreen", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-sequence", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="default: data/phase0/<area>[_prescreen]")
    ap.add_argument("--map-features", action="store_true", help="also dump Mapillary map features for the area")
    args = ap.parse_args()

    out: Path = args.out or Path("data/phase0") / (args.area + ("_prescreen" if args.prescreen else ""))
    out.mkdir(parents=True, exist_ok=True)
    client = MapillaryClient()
    bbox = BBox(*AREAS[args.area])

    if args.prescreen:
        survey = Path("data/phase0/survey") / f"{args.area}.json"
        if not survey.exists():
            raise SystemExit(f"{survey} not found; run scripts/phase0_survey.py --areas {args.area} first")
        passing = json.loads(survey.read_text())["passing"]
        deduped = spatial_dedup(passing)
        print(f"{len(passing)} pre-screened images, {len(deduped)} after dropping repeat views of the same spot")
        sample = sample_images(deduped, args.n, args.per_sequence, args.seed)
    else:
        sample = sample_images(client.search_images(bbox), args.n, args.per_sequence, args.seed)
    if len(sample) < args.n:
        print(f"warning: only {len(sample)} images available for the sample")

    print(f"Downloading {len(sample)} images and their detections ...")
    for img in tqdm(sample):
        path = out / "images" / f"{img['id']}.jpg"
        if not path.exists():
            url = client.image(img["id"], ["thumb_2048_url"])["thumb_2048_url"]  # cached URLs expire
            client.download(url, path)
    detections = fetch_detections(client, [i["id"] for i in sample])
    pd.DataFrame([to_row(idx, img) for idx, img in enumerate(sample)]).to_csv(out / "sample.csv", index=False)
    (out / "detections.json").write_text(json.dumps(detections))
    (out / "candidates.json").write_text(json.dumps({i["id"]: i.get("candidates", []) for i in sample}))

    if args.map_features:
        features = client.map_features(bbox)
        (out / "map_features.json").write_text(json.dumps(features))
        values = collections.Counter(f["object_value"] for f in features)
        print(f"{len(features)} map features; parking-related:")
        for v, c in values.most_common():
            if any(h in v for h in PARKING_HINTS):
                print(f"  {c:>5}  {v}")

    print(f"Done. Images, metadata and detections are in {out}")


if __name__ == "__main__":
    main()
