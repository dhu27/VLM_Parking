"""Phase 1 collection: turn Mapillary images into deduplicated parking-sign crops ready for triage.

Stages (each resumable; re-running skips work already on disk):
  screen    fetch detections for up to --n-screen images per area; keep images with a large sign
  crop      download 2048-px thumbnails of passing images, merge stacked panels, crop, OCR each crop
  dedup     drop crops OCR clearly reads as non-parking signs (OCR never gates *in*: it misses many real
            signs, and gating on it would bias toward easy-to-read signs); cluster views of the same
            physical sign; pick the best view
  final     download originals for the chosen views and save full-resolution crops + candidates.csv
  sheets    contact sheets per area for fast triage

    uv run python scripts/collect.py                                  # all stages, all areas
    uv run python scripts/collect.py --stages screen crop --areas koreatown
"""

from __future__ import annotations

import argparse
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from vlm_parking.collect import cluster_duplicates, dedupe_boxes, merge_stacks, padded, parking_text_score, scale_box
from vlm_parking.mapillary import BBox, MapillaryClient, decode_detection_geometry, ring_bbox
from vlm_parking.prescreen import (
    AREAS,
    THUMB,
    Criteria,
    is_sign_value,
    load_freeways,
    metadata_ok,
    near_freeway,
    sample_images,
    sign_candidates,
)

ROOT = Path("data/collect")  # overridable with --root
CACHE = Path("data/phase0/cache")
STAGES = ["screen", "crop", "dedup", "final", "sheets"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def area_images(client: MapillaryClient, area: str) -> list[dict]:
    path = CACHE / f"{area}_images.json"
    if path.exists():
        return json.loads(path.read_text())
    images = client.search_images(BBox(*AREAS[area]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(images))
    return images


# --- stage 1: screen ------------------------------------------------------------------------------


def stage_screen(client: MapillaryClient, area: str, c: Criteria, n_screen: int, seed: int, workers: int) -> None:
    out = ROOT / area / "screened.jsonl"
    done = {r["id"] for r in read_jsonl(out)}
    images = area_images(client, area)
    freeways = load_freeways(BBox(*AREAS[area]))
    pool = [i for i in images if metadata_ok(i, c) and not near_freeway(*i["geometry"]["coordinates"], freeways, c.freeway_buffer_m)]
    per_seq = max(2, math.ceil(n_screen / max(1, len({i.get("sequence") for i in pool}))))
    todo = [i for i in sample_images(pool, n_screen, per_seq, seed) if i["id"] not in done]
    print(f"[{area}] screen: {len(pool):,} eligible, {len(done):,} done, {len(todo):,} to go")

    def work(img: dict) -> dict:
        dets = client.detections(img["id"])
        w, h = img["width"], img["height"]
        signs = []
        for d in dets:
            if is_sign_value(d["value"]):
                signs += [list(ring_bbox(r)) for r in decode_detection_geometry(d["geometry"], w, h)]
        cands = sign_candidates(dets, w, h, c)
        meta = {k: img.get(k) for k in ("id", "captured_at", "compass_angle", "is_pano", "width", "height", "sequence")}
        return {
            **meta,
            "lon": img["geometry"]["coordinates"][0],
            "lat": img["geometry"]["coordinates"][1],
            "creator_username": (img.get("creator") or {}).get("username"),
            "passing": bool(cands),
            "candidates": [cd["bbox"] for cd in cands],
            "signs": signs if cands else [],
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in tqdm(range(0, len(todo), 200), desc=f"{area} screen", unit="batch"):
            append_jsonl(out, list(ex.map(work, todo[start : start + 200])))
    rows = read_jsonl(out)
    print(f"[{area}] screened {len(rows):,}, passing {sum(r['passing'] for r in rows):,}")


# --- stage 2: crop + OCR --------------------------------------------------------------------------

_ocr = None


def ocr_engine():
    global _ocr
    if _ocr is None:
        import logging

        from rapidocr import RapidOCR

        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        _ocr = RapidOCR()
    return _ocr


def stacks_for(row: dict) -> list[dict]:
    """Merged stacks that contain at least one pre-screen candidate (native pixel coords)."""
    boxes = dedupe_boxes([b for b in row["signs"] if (b[3] - b[1]) * THUMB / max(row["width"], row["height"]) >= 12])
    out = []
    for s in merge_stacks(boxes):
        if any(s["bbox"][0] <= (c[0] + c[2]) / 2 <= s["bbox"][2] and s["bbox"][1] <= (c[1] + c[3]) / 2 <= s["bbox"][3] for c in row["candidates"]):
            out.append(s)
    return out


def stage_crop(client: MapillaryClient, area: str, workers: int) -> None:
    out = ROOT / area / "stacks.jsonl"
    crops_dir = ROOT / area / "thumb_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    done = {r["image_id"] for r in read_jsonl(out)}
    todo = [r for r in read_jsonl(ROOT / area / "screened.jsonl") if r["passing"] and r["id"] not in done]
    print(f"[{area}] crop: {len(done):,} done, {len(todo):,} to go")

    def fetch(row: dict) -> tuple[dict, Image.Image | None]:
        try:
            url = client.image(row["id"], ["thumb_2048_url"])["thumb_2048_url"]
            return row, Image.open(io.BytesIO(client._get(url).content)).convert("RGB")
        except (requests.RequestException, OSError, KeyError):
            return row, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for start in tqdm(range(0, len(todo), 100), desc=f"{area} crop", unit="batch"):
            results = []
            for row, im in ex.map(fetch, todo[start : start + 100]):
                if im is None:
                    results.append({"image_id": row["id"], "k": -1, "error": "download failed"})
                    continue
                for k, s in enumerate(stacks_for(row)):
                    box = scale_box(s["bbox"], row["width"], row["height"], im.width, im.height)
                    crop = im.crop(padded(box, im.width, im.height))
                    name = f"{row['id']}_{k}.jpg"
                    crop.save(crops_dir / name, quality=92)
                    # OCR struggles on tiny crops; upscale to a readable height first
                    ocr_in = crop if crop.height >= 320 else crop.resize((max(1, int(crop.width * 320 / crop.height)), 320), Image.LANCZOS)
                    ocr = ocr_engine()(ocr_in)
                    texts = list(ocr.txts or [])
                    scores = [float(x) for x in (ocr.scores or [])]
                    results.append({
                        "image_id": row["id"], "k": k, "crop": name,
                        "bbox_native": s["bbox"], "n_panels_detected": s["n_panels"],
                        "height_native": s["bbox"][3] - s["bbox"][1],
                        "ocr_text": " | ".join(texts), "ocr_conf": sum(scores) / len(scores) if scores else 0.0,
                        "phash": int(str(imagehash.phash(crop)), 16),
                        **parking_text_score(texts),
                    })
                if not stacks_for(row):
                    results.append({"image_id": row["id"], "k": -1, "error": "no stack"})
            append_jsonl(out, results)
    rows = [r for r in read_jsonl(out) if r["k"] >= 0]
    print(f"[{area}] {len(rows):,} stacks, {sum(r['keep'] for r in rows):,} pass OCR keyword filter")


# --- stage 3: dedup -------------------------------------------------------------------------------


def stage_dedup(area: str) -> pd.DataFrame:
    screened = {r["id"]: r for r in read_jsonl(ROOT / area / "screened.jsonl")}
    stacks = [r for r in read_jsonl(ROOT / area / "stacks.jsonl") if r.get("k", -1) >= 0]
    # re-score from saved OCR text so rule changes don't need a re-run of OCR
    for s in stacks:
        s.update(parking_text_score([t for t in s["ocr_text"].split(" | ") if t]))
    n_other = sum(s["status"] == "other" for s in stacks)
    stacks = [s for s in stacks if s["status"] != "other"]
    for s in stacks:
        img = screened[s["image_id"]]
        s.update(lon=img["lon"], lat=img["lat"])
    if not stacks:
        print(f"[{area}] dedup: nothing to do")
        return pd.DataFrame()
    clusters = cluster_duplicates(stacks)
    df = pd.DataFrame(stacks).assign(cluster=clusters)
    # Crops already triaged stay the representative of their sign, so adding more images later never
    # re-shows a triaged sign or orphans a decision. If new views merge two triaged crops into one
    # cluster, both are kept and flagged (cluster_n_triaged > 1) for a duplicate check at labeling.
    triage = ROOT / "triage.csv"
    triaged = set(pd.read_csv(triage).candidate_id) if triage.exists() else set()
    df["triaged"] = [f"{area}/{i}_{k}" in triaged for i, k in zip(df.image_id, df.k)]
    df["cluster_n_triaged"] = df.groupby("cluster").triaged.transform("sum")
    # otherwise the best view: largest sign in native pixels, weighted by OCR confidence, then most panels
    df["rank_score"] = df.height_native * (0.5 + df.ocr_conf)
    ranked = df.sort_values(["rank_score", "n_panels_detected"], ascending=False)
    best = pd.concat([ranked[ranked.triaged], ranked[ranked.cluster_n_triaged == 0].groupby("cluster").head(1)])
    best = best.assign(n_views=df.groupby("cluster").size().reindex(best.cluster).values)
    best.to_csv(ROOT / area / "dedup.csv", index=False)
    print(f"[{area}] dedup: {n_other:,} rejected as clearly non-parking; {len(df):,} kept -> {len(best):,} unique signs "
          f"({(best.status == 'parking').sum():,} OCR-read as parking, {(best.status == 'unknown').sum():,} unknown)")
    return best


# --- stage 4: full-resolution crops ---------------------------------------------------------------


def stage_final(client: MapillaryClient, area: str, workers: int) -> None:
    best = pd.read_csv(ROOT / area / "dedup.csv", dtype={"image_id": str})
    screened = {r["id"]: r for r in read_jsonl(ROOT / area / "screened.jsonl")}
    out_dir = ROOT / "crops" / area
    out_dir.mkdir(parents=True, exist_ok=True)

    def work(r) -> dict | None:
        name = f"{r.image_id}_{r.k}.jpg"
        img = screened[r.image_id]
        if not (out_dir / name).exists():
            try:
                url = client.image(r.image_id, ["thumb_original_url"])["thumb_original_url"]
                im = Image.open(io.BytesIO(client._get(url).content)).convert("RGB")
            except (requests.RequestException, OSError, KeyError):
                return None
            box = scale_box(json.loads(r.bbox_native) if isinstance(r.bbox_native, str) else r.bbox_native, img["width"], img["height"], im.width, im.height)
            crop = im.crop(padded(box, im.width, im.height))
            crop.save(out_dir / name, quality=95)
        captured = datetime.fromtimestamp(img["captured_at"] / 1000, tz=timezone.utc)
        return {
            "candidate_id": f"{area}/{r.image_id}_{r.k}",
            "area": area,
            "crop_path": f"crops/{area}/{name}",
            "image_id": r.image_id,
            "captured_at": captured.isoformat(),
            "year": captured.year,
            "lat": img["lat"],
            "lon": img["lon"],
            "compass_angle": img["compass_angle"],
            "is_pano": img["is_pano"],
            "creator_username": img["creator_username"],
            "source_url": f"https://www.mapillary.com/app/?pKey={r.image_id}",
            "license": "CC BY-SA 4.0",
            "n_panels_detected": r.n_panels_detected,
            "height_native": round(r.height_native),
            "n_views": r.n_views,
            "ocr_status": r.status,
            "ocr_text": r.ocr_text if isinstance(r.ocr_text, str) else "",
            "ocr_conf": round(r.ocr_conf, 3),
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = [x for x in tqdm(ex.map(work, best.itertuples()), total=len(best), desc=f"{area} final") if x]
    pd.DataFrame(rows).to_csv(ROOT / area / "candidates.csv", index=False)
    print(f"[{area}] final: {len(rows):,} full-resolution crops in {out_dir}")


# --- stage 5: contact sheets ----------------------------------------------------------------------


def stage_sheets(area: str, per_sheet: int = 60, tile_h: int = 220, width: int = 1800) -> None:
    # likely parking signs first, then larger stacks
    cands = pd.read_csv(ROOT / area / "candidates.csv")
    cands = cands.assign(_p=cands.ocr_status.eq("parking")).sort_values(["_p", "n_panels_detected", "height_native"], ascending=False)
    out_dir = ROOT / "sheets" / area
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    for p, start in enumerate(range(0, len(cands), per_sheet)):
        tiles, index = [], []
        for n, r in enumerate(cands.iloc[start : start + per_sheet].itertuples()):
            im = Image.open(ROOT / r.crop_path).convert("RGB")
            im = im.resize((max(1, int(im.width * tile_h / im.height)), tile_h), Image.LANCZOS)
            ImageDraw.Draw(im).text((3, 2), f"{n}", fill="yellow", font=font, stroke_width=2, stroke_fill="black")
            tiles.append(im)
            index.append({"sheet": p, "n": n, "candidate_id": r.candidate_id})
        pd.DataFrame(index).to_csv(out_dir / f"sheet_{p:02d}.csv", index=False)
        rows, row, x = [], [], 0
        for t in tiles:
            if x + t.width > width and row:
                rows.append(row); row, x = [], 0
            row.append(t); x += t.width + 6
        rows.append(row)
        sheet = Image.new("RGB", (width, (tile_h + 6) * len(rows)), "white")
        for i, rr in enumerate(rows):
            x = 0
            for t in rr:
                sheet.paste(t, (x, i * (tile_h + 6))); x += t.width + 6
        sheet.save(out_dir / f"sheet_{p:02d}.jpg", quality=85)
    print(f"[{area}] sheets: {math.ceil(len(cands) / per_sheet)} in {out_dir}")


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--areas", nargs="+", default=list(AREAS), choices=list(AREAS))
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--n-screen", type=int, default=5000)
    ap.add_argument("--min-sign-px", type=float, default=Criteria.min_sign_px)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()

    ROOT = args.root
    client = MapillaryClient()
    c = Criteria(min_sign_px=args.min_sign_px)
    for area in args.areas:
        if "screen" in args.stages:
            stage_screen(client, area, c, args.n_screen, args.seed, args.workers)
        if "crop" in args.stages:
            stage_crop(client, area, args.workers)
        if "dedup" in args.stages:
            stage_dedup(area)
        if "final" in args.stages:
            stage_final(client, area, args.workers)
        if "sheets" in args.stages:
            stage_sheets(area)
    summary = []
    for area in args.areas:
        p = ROOT / area / "candidates.csv"
        if p.exists():
            summary.append({"area": area, "candidates": len(pd.read_csv(p))})
    if summary:
        print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
