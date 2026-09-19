import base64

import mapbox_vector_tile

from vlm_parking.mapillary import BBox, decode_detection_geometry
from vlm_parking.prescreen import Criteria, near_freeway, sign_candidates, spatial_dedup


def encode_box(x0, y0, x1, y1, extent=4096):
    """Encode a pixel-free box (tile units) the way Mapillary encodes detection geometry."""
    polygon = f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"
    tile = mapbox_vector_tile.encode(
        [{"name": "mpy-or", "features": [{"geometry": polygon, "properties": {}}]}],
        default_options={"extents": extent, "y_coord_down": True},
    )
    return base64.b64encode(tile).decode()


def det(value, box, id_="1"):
    return {"id": id_, "value": value, "geometry": encode_box(*box)}


def test_tiles_cover_bbox_within_step():
    tiles = list(BBox(0, 0, 0.012, 0.007).tiles(0.005))
    assert len(tiles) == 3 * 2
    assert all(t.max_lon - t.min_lon <= 0.005 + 1e-12 and t.max_lat - t.min_lat <= 0.005 + 1e-12 for t in tiles)
    assert min(t.min_lon for t in tiles) == 0 and max(t.max_lon for t in tiles) == 0.012


def test_decode_geometry_is_y_down_pixels():
    (ring,) = decode_detection_geometry(encode_box(1024, 2048, 2048, 3072), 2048, 1536)
    xs, ys = zip(*ring)
    assert (min(xs), max(xs)) == (512, 1024)
    assert (min(ys), max(ys)) == (768, 1152)  # lower half of the frame, not flipped


def test_sign_candidates_filters():
    c = Criteria()
    W, H = 2048, 2048  # 1 tile unit = 0.5 px
    ok = det("object--traffic-sign--front", (1000, 2000, 1200, 2400))  # 200 px tall, portrait
    small = det("object--traffic-sign--front", (1000, 2000, 1040, 2080))  # 40 px
    wide = det("object--traffic-sign--front", (1000, 2000, 1800, 2300))  # landscape street-name sign
    edge = det("object--traffic-sign--front", (0, 2000, 200, 2400))  # touches the left edge
    overhead = det("object--traffic-sign--front", (1000, 100, 1200, 500))  # top of frame
    stop = det("regulatory--stop--g1", (1000, 2000, 1200, 2400))  # classified non-parking sign
    got = sign_candidates([ok, small, wide, edge, overhead, stop], W, H, c)
    assert len(got) == 1 and got[0]["height_2048"] == 200


def test_near_freeway_buffer():
    line = [[(-118.0, 34.0), (-118.0, 34.01)]]
    assert near_freeway(-118.0002, 34.005, line, 30)  # ~18 m east
    assert not near_freeway(-118.001, 34.005, line, 30)  # ~92 m east


def test_spatial_dedup_same_sequence_only():
    def img(i, seq, lon):
        return {"id": str(i), "sequence": seq, "geometry": {"coordinates": [lon, 34.0]}}

    kept = spatial_dedup([img(1, "a", -118.0), img(2, "a", -118.00005), img(3, "b", -118.00005), img(4, "a", -118.001)])
    assert [k["id"] for k in kept] == ["1", "3", "4"]
