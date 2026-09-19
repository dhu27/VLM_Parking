"""Minimal Mapillary Graph API v4 client: bbox search with tiling, retries, detections, downloads.

API reference: https://www.mapillary.com/developer/api-documentation
"""

from __future__ import annotations

import base64
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mapbox_vector_tile
import requests
from dotenv import load_dotenv

GRAPH_URL = "https://graph.mapillary.com"

# Search endpoints return at most this many results per request.
MAX_LIMIT = 2000

# The API requires bbox area < 0.01 square degrees; 0.005° tiles are well inside that.
DEFAULT_TILE_DEG = 0.005
MIN_TILE_DEG = 0.000625

IMAGE_FIELDS = [
    "id",
    "thumb_2048_url",
    "captured_at",
    "compass_angle",
    "is_pano",
    "geometry",
    "creator",
    "sequence",
    "width",
    "height",
]

MAP_FEATURE_FIELDS = ["id", "object_value", "object_type", "geometry", "first_seen_at", "last_seen_at"]


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def param(self) -> str:
        return f"{self.min_lon:.6f},{self.min_lat:.6f},{self.max_lon:.6f},{self.max_lat:.6f}"

    def tiles(self, step: float = DEFAULT_TILE_DEG) -> Iterator[BBox]:
        """Split into a grid of tiles no larger than step × step degrees."""
        lon = self.min_lon
        while lon < self.max_lon - 1e-9:
            lat = self.min_lat
            next_lon = min(lon + step, self.max_lon)
            while lat < self.max_lat - 1e-9:
                next_lat = min(lat + step, self.max_lat)
                yield BBox(lon, lat, next_lon, next_lat)
                lat = next_lat
            lon = next_lon

    def quarters(self) -> list[BBox]:
        mid_lon = (self.min_lon + self.max_lon) / 2
        mid_lat = (self.min_lat + self.max_lat) / 2
        return [
            BBox(self.min_lon, self.min_lat, mid_lon, mid_lat),
            BBox(mid_lon, self.min_lat, self.max_lon, mid_lat),
            BBox(self.min_lon, mid_lat, mid_lon, self.max_lat),
            BBox(mid_lon, mid_lat, self.max_lon, self.max_lat),
        ]


class MapillaryClient:
    def __init__(self, token: str | None = None, max_retries: int = 5, timeout: float = 60):
        if token is None:
            load_dotenv()
            token = os.environ.get("MAPILLARY_TOKEN", "")
        if not token:
            raise RuntimeError("MAPILLARY_TOKEN is not set; add it to .env")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"OAuth {token}"
        self.max_retries = max_retries
        self.timeout = timeout

    def _get(
        self, url: str, params: dict | None = None, stream: bool = False, max_retries: int | None = None
    ) -> requests.Response:
        """GET with exponential backoff on rate limits, server errors, and connection failures."""
        max_retries = self.max_retries if max_retries is None else max_retries
        for attempt in range(max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout, stream=stream)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} from {url}", response=resp)
                resp.raise_for_status()
                return resp
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as err:
                status = getattr(err.response, "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt == max_retries:
                    raise
                time.sleep(min(60, 2**attempt) + random.random())
        raise AssertionError("unreachable")

    def _search(self, endpoint: str, bbox: BBox, fields: list[str], params: dict, step: float) -> list[dict]:
        """Query every tile of bbox. A tile that hits the result cap, or keeps failing with server
        errors (which dense tiles tend to do), is split into quarters so nothing is silently lost."""
        results: dict[str, dict] = {}
        pending = list(bbox.tiles(step))
        while pending:
            tile = pending.pop()
            splittable = (tile.max_lon - tile.min_lon) > MIN_TILE_DEG
            try:
                data = self._get(
                    f"{GRAPH_URL}/{endpoint}",
                    {**params, "bbox": tile.param(), "fields": ",".join(fields), "limit": MAX_LIMIT},
                    max_retries=2 if splittable else None,
                ).json()["data"]
            except requests.HTTPError as err:
                if splittable and err.response is not None and err.response.status_code >= 500:
                    pending.extend(tile.quarters())
                    continue
                raise
            if len(data) >= MAX_LIMIT and splittable:
                pending.extend(tile.quarters())
                continue
            for item in data:
                results[item["id"]] = item
        return list(results.values())

    def search_images(
        self, bbox: BBox, fields: list[str] = IMAGE_FIELDS, step: float = DEFAULT_TILE_DEG, **filters
    ) -> list[dict]:
        """All images in bbox. filters: e.g. start_captured_at="2020-01-01T00:00:00Z", is_pano=False."""
        return self._search("images", bbox, fields, filters, step)

    def map_features(
        self, bbox: BBox, fields: list[str] = MAP_FEATURE_FIELDS, step: float = DEFAULT_TILE_DEG, **filters
    ) -> list[dict]:
        """All map features (traffic signs and points) in bbox. filters: e.g. object_values="regulatory--*"."""
        return self._search("map_features", bbox, fields, filters, step)

    def image(self, image_id: str, fields: list[str] = IMAGE_FIELDS) -> dict:
        """One image's fields; use to refresh expired thumbnail URLs."""
        return self._get(f"{GRAPH_URL}/{image_id}", {"fields": ",".join(fields)}).json()

    def detections(self, image_id: str) -> list[dict]:
        """Object detections for one image; geometry is a base64 vector tile (see decode_detection_geometry)."""
        return self._get(f"{GRAPH_URL}/{image_id}/detections", {"fields": "id,value,geometry"}).json()["data"]

    def download(self, url: str, path: Path) -> None:
        """Download to path atomically; thumbnail URLs are signed and expire, so fetch soon after searching."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with self._get(url, stream=True) as resp, open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        tmp.replace(path)


def decode_detection_geometry(geometry_b64: str, width: int, height: int) -> list[list[tuple[float, float]]]:
    """Decode a detection's base64 vector-tile geometry into polygon rings in pixel coordinates.

    Tile coordinates run 0..extent (4096) with y pointing down, matching image coordinates.
    """
    tile = mapbox_vector_tile.decode(base64.b64decode(geometry_b64), default_options={"y_coord_down": True})
    rings = []
    for layer in tile.values():
        extent = layer.get("extent", 4096)
        for feature in layer["features"]:
            geom = feature["geometry"]
            polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
            for polygon in polygons:
                for ring in polygon:
                    rings.append([(x / extent * width, y / extent * height) for x, y in ring])
    return rings


def ring_bbox(ring: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs, ys = zip(*ring)
    return min(xs), min(ys), max(xs), max(ys)
