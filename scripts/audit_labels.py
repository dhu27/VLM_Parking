"""Local OCR-vs-label audit: find labels whose rule contradicts the text on the sign.

    uv run python scripts/audit_labels.py           # uses the cached OCR if present
    uv run python scripts/audit_labels.py --reocr   # re-run OCR from the crops

Runs RapidOCR on CPU (already a dependency; no GPU, nothing leaves the machine) over every labeled
crop and compares the words on the sign against the rules in the label. The output is a worklist for
`scripts/label.py --only`, never an edit: OCR is the accuser, the human is the judge.

Only one check earns a place on the worklist, because only one survived hand-verification:

  rule_conflict - the sign says STOPPING / STANDING but no panel carries that rule.

Checks that were tried and dropped, so they don't get re-added:

  except_day    - "EXCEPT SAT & SUN" belongs to the panel it sits under, not the whole sign. On an
                  anti-gridlock stack the exception governs the no-stopping plate while a separate
                  SATURDAY plate legitimately includes SAT. Every flag was a false positive.
  limit_missing - OCR returns "HOUR | 4PARKING" as often as "4 HOUR PARKING", and reads HOUR as
                  AOUR; 103 flags, no label errors.
  time_missing  - OCR reads 1PM as IPM and 11AM as lLAM. 60 flags, no label errors.

`tow_away` is deliberately not audited: it is documented as an unannotated field (see
ANNOTATION_GUIDE.md). The summary still counts it so the claim stays honest.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("RapidOCR").setLevel(logging.ERROR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

COLLECT = Path("data/collect")
LABELS = Path("data/labels")
QUERIES = Path("data/queries")
CACHE = LABELS / "ocr_cache.json"
WORKLIST = LABELS / "audit_worklist.csv"

UPSCALE = 3  # crops average 362 px on the long side; the recogniser needs the pixels

# A word distinctive enough that seeing it means the rule belongs somewhere on the sign. Matched
# against text with all separators stripped, so a split "NOS | STOPPING" still hits.
RULE_WORDS = {"no_stopping": "STOPPING", "no_standing": "STANDING"}


def norm(text: str) -> str:
    """Uppercase, alphanumerics only, so '8AM  TO 6PM' and '8AM=6PM' compare equal."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def run_ocr(signs: pd.DataFrame, reocr: bool) -> dict[str, list[str]]:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() and not reocr else {}
    todo = [r for r in signs.itertuples() if r.sign_id not in cache]
    if todo:
        from rapidocr import RapidOCR

        engine = RapidOCR()
        for i, r in enumerate(todo, 1):
            im = Image.open(COLLECT / r.crop_path).convert("RGB")
            res = engine(np.array(im.resize((im.width * UPSCALE, im.height * UPSCALE), Image.LANCZOS)))
            cache[r.sign_id] = list(res.txts) if res and res.txts else []
            if i % 25 == 0 or i == len(todo):
                print(f"  OCR {i}/{len(todo)}", flush=True)
        CACHE.write_text(json.dumps(cache))
    return cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reocr", action="store_true")
    args = ap.parse_args()

    signs = pd.read_parquet(QUERIES / "signs_v1.parquet")
    queries = pd.read_parquet(QUERIES / "queries_v1.parquet")
    n_queries = queries.groupby("sign_id").size().to_dict()
    print(f"auditing {len(signs)} labeled signs")
    cache = run_ocr(signs, args.reocr)

    rows, no_ocr, tow_unset = [], 0, 0
    for r in signs.itertuples():
        sign = json.loads(r.sign_json)
        text = " | ".join(cache.get(r.sign_id, []))
        flat = norm(text)
        if not flat:
            no_ocr += 1
            continue
        rules = {p["rule"] for p in sign["panels"]}
        if "TOWAWAY" in flat and not any(p.get("tow_away") for p in sign["panels"]):
            tow_unset += 1
        for rule, word in RULE_WORDS.items():
            if word in flat and rule not in rules:
                rows.append({"sign_id": r.sign_id, "issue": f"sign says {word}, no {rule} panel",
                             "n_panels": len(sign["panels"]), "n_queries": n_queries.get(r.sign_id, 0),
                             "labeled_rules": ",".join(sorted(rules)), "crop_path": r.crop_path, "ocr": text[:160]})

    cols = ["sign_id", "issue", "n_panels", "n_queries", "labeled_rules", "crop_path", "ocr"]
    out = pd.DataFrame(rows, columns=cols).sort_values(["issue", "sign_id"])  # columns kept for the clean case
    LABELS.mkdir(parents=True, exist_ok=True)
    out.to_csv(WORKLIST, index=False)

    print(f"\n{len(out)} signs to review ({out.n_queries.sum()} queries, "
          f"{out.n_queries.sum() / len(queries):.1%} of the set) -> {WORKLIST}")
    for r in out.itertuples():
        print(f"  {r.sign_id} ({r.n_panels}p, {r.n_queries}q) has {r.labeled_rules}\n     OCR: {r.ocr[:120]}")
    print(f"\nnot on the worklist: {tow_unset} signs show TOW-AWAY with tow_away unset "
          f"(field documented as unannotated); {no_ocr} signs OCR read nothing from")
    if len(out):
        print(f"\nnext: uv run python scripts/label.py --only {WORKLIST}")


if __name__ == "__main__":
    main()
