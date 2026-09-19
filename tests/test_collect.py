from vlm_parking.collect import cluster_duplicates, dedupe_boxes, merge_stacks, parking_text_score


def test_dedupe_boxes_drops_duplicate_detections():
    boxes = [[100, 100, 200, 300], [102, 98, 201, 302], [400, 100, 500, 300]]
    assert len(dedupe_boxes(boxes)) == 2


def test_merge_stacks_groups_vertical_panels_only():
    top = [100, 100, 200, 250]
    bottom = [105, 260, 198, 400]  # 10 px gap below, same column -> same stack
    beside = [400, 100, 500, 250]  # different pole
    far_below = [100, 700, 200, 850]  # same column but far below -> separate
    stacks = merge_stacks([top, bottom, beside, far_below])
    sizes = sorted(s["n_panels"] for s in stacks)
    assert sizes == [1, 1, 2]
    merged = next(s for s in stacks if s["n_panels"] == 2)
    assert merged["bbox"] == [100, 100, 200, 400]


def test_parking_text_score_statuses():
    assert parking_text_score(["2HOUR", "PARKING", "8AM6PM"])["status"] == "parking"
    assert parking_text_score(["R", "4ouk", "FARKING"])["status"] == "parking"  # fuzzy match
    assert parking_text_score(["8AM10AM", "THURSDAY"])["status"] == "parking"  # times + day, no negatives
    assert parking_text_score(["SPEED", "LIMIT"])["status"] == "other"
    assert parking_text_score(["AHEAD"])["status"] == "other"
    assert parking_text_score([])["status"] == "unknown"  # OCR saw nothing: goes to human triage
    assert parking_text_score(["TOW-AWAY", "NO", "STOPPING", "STOP"])["status"] == "parking"  # parking beats negative


def test_cluster_duplicates_needs_proximity_and_similarity():
    base = {"lon": -118.3, "lat": 34.06, "phash": 0b1010, "ocr_text": "2 HOUR PARKING 8AM 6PM"}
    same_sign_other_drive = {**base, "lon": -118.30005, "phash": 0b1111_0000_1111}  # 5 m away, text agrees
    different_sign_nearby = {**base, "lon": -118.30005, "phash": (1 << 40) - 1, "ocr_text": "NO STOPPING TOW AWAY"}
    same_text_far_away = {**base, "lat": 34.07}  # ~1 km away
    c = cluster_duplicates([base, same_sign_other_drive, different_sign_nearby, same_text_far_away])
    assert c[0] == c[1]
    assert c[2] != c[0] and c[3] != c[0]
