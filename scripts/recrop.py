"""Re-cut one sign's crop from the original photo, when the detector's box missed part of the stack.

    uv run python scripts/recrop.py --sign hollywood/1979970682169839_0 --view        # inspect first
    uv run python scripts/recrop.py --sign hollywood/1979970682169839_0 --box 3085 853 3315 1334

Without --box, --view saves a wide view around the current box (with a pixel grid) so the right box can
be read off it. Applying a box rewrites data/collect/crops/<area>/<id>.jpg and appends the change to
data/collect/recrops.csv, so every manual fix stays on the record.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

from vlm_parking.collect import padded
from vlm_parking.mapillary import MapillaryClient

COLLECT = Path("data/collect")
LOG = COLLECT / "recrops.csv"


def stack_box(area: str, image_id: str, k: int) -> list[float]:
    for line in (COLLECT / area / "stacks.jsonl").open():
        row = json.loads(line)
        if row.get("image_id") == image_id and row.get("k") == k:
            return row["bbox_native"]
    raise SystemExit(f"no stack for {image_id}_{k}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sign", required=True, help="candidate_id, e.g. hollywood/1979970682169839_0")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="new box in original-image pixels")
    ap.add_argument("--view", action="store_true", help="save a wide view with a coordinate grid")
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--out", type=Path, default=Path("recrop_view.png"))
    args = ap.parse_args()

    area, stem = args.sign.split("/")
    image_id, k = stem.rsplit("_", 1)
    old = stack_box(area, image_id, int(k))
    client = MapillaryClient()
    im = Image.open(io.BytesIO(client._get(client.image(image_id, ["thumb_original_url"])["thumb_original_url"]).content)).convert("RGB")
    print(f"{args.sign}: original {im.size}, current box {[round(v) for v in old]}")

    if args.view or not args.box:
        x0, y0, x1, y1 = old
        w, h = x1 - x0, y1 - y0
        view_box = (max(0, x0 - 2 * w), max(0, y0 - h), min(im.width, x1 + 2 * w), min(im.height, y1 + 4 * h))
        view = im.crop(tuple(int(v) for v in view_box)).copy()
        draw = ImageDraw.Draw(view)
        draw.rectangle([x0 - view_box[0], y0 - view_box[1], x1 - view_box[0], y1 - view_box[1]], outline="red", width=3)
        step = 100
        for gx in range(0, view.width, step):  # grid labelled in original-image coordinates
            draw.line([(gx, 0), (gx, view.height)], fill=(255, 255, 0), width=1)
            draw.text((gx + 2, 2), str(int(view_box[0]) + gx), fill=(255, 255, 0))
        for gy in range(0, view.height, step):
            draw.line([(0, gy), (view.width, gy)], fill=(255, 255, 0), width=1)
            draw.text((2, gy + 2), str(int(view_box[1]) + gy), fill=(255, 255, 0))
        view.save(args.out)
        print(f"view saved to {args.out} (red = current box, grid = original-image pixels)")
        if not args.box:
            return

    crop_path = COLLECT / "crops" / area / f"{stem}.jpg"
    before = Image.open(crop_path).size
    crop = im.crop(padded(list(args.box), im.width, im.height, args.pad))
    crop.save(crop_path, quality=95)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["candidate_id", "old_box", "new_box", "old_size", "new_size", "changed_at"])
        w.writerow([args.sign, [round(v) for v in old], [round(v) for v in args.box], before, crop.size,
                    datetime.now(timezone.utc).isoformat(timespec="seconds")])
    print(f"re-cropped {crop_path}: {before} -> {crop.size}; logged in {LOG}")


if __name__ == "__main__":
    main()
