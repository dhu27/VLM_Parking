"""Phase 4: build the query set from the labeled signs.

    uv run python scripts/make_queries.py            # writes data/queries/{signs,queries}_v1.parquet

Ground truth for every query comes from the evaluator, so the query set can be rebuilt at any time:
it is a deterministic function of the labels, the seed, and the evaluator.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from vlm_parking.dataset import accepted_candidates
from vlm_parking.queries import generate_for_sign
from vlm_parking.schema import Sign

LABELS_DB = Path("data/labels/labels.sqlite")
OUT = Path("data/queries")


def labeled_signs(round_: int = 1) -> pd.DataFrame:
    """One row per labeled sign in the accepted cohort, with its crop and attribution."""
    with sqlite3.connect(LABELS_DB) as con:
        labels = pd.read_sql(
            "SELECT candidate_id, sign_json, seconds, saved_at FROM labels WHERE round = ? AND status = 'labeled'",
            con, params=(round_,),
        )
    cands = accepted_candidates()
    df = labels.merge(cands, on="candidate_id")
    df["sign"] = df.sign_json.map(lambda s: Sign(**json.loads(s)))
    df["n_panels"] = df.sign.map(lambda s: s.n_panels)
    df["ambiguous"] = df.sign.map(lambda s: s.ambiguous)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    signs = labeled_signs(args.round)
    usable = signs[~signs.ambiguous]
    print(f"{len(signs)} labeled signs, {len(usable)} usable (ambiguous signs generate no queries)")

    rows = []
    for r in usable.itertuples():
        rows += [{**q, "area": r.area, "n_panels": r.n_panels} for q in generate_for_sign(r.sign, args.seed)]
    queries = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    sign_cols = ["candidate_id", "area", "crop_path", "source_url", "creator_username", "license", "captured_at",
                 "year", "lat", "lon", "height_native", "n_panels", "sign_json"]
    usable[sign_cols].rename(columns={"candidate_id": "sign_id"}).to_parquet(OUT / f"signs_{args.version}.parquet", index=False)
    queries.to_parquet(OUT / f"queries_{args.version}.parquet", index=False)

    n_signs = queries.sign_id.nunique()
    print(f"\n{len(queries):,} queries over {n_signs} signs ({len(queries) / n_signs:.1f} per sign)")
    print("\nby type:\n", queries.groupby("type").verdict.value_counts().unstack(fill_value=0).to_string())
    print("\nbase rate (share legal):", round((queries.verdict == "legal").mean(), 3))
    print("\nby panel count:\n", queries.groupby(queries.n_panels.clip(upper=3)).agg(
        queries=("verdict", "size"), signs=("sign_id", "nunique"), share_legal=("verdict", lambda v: round((v == "legal").mean(), 2))).to_string())
    short = queries.groupby("sign_id").size().pipe(lambda s: s[s < 15])
    print(f"\nsigns with fewer than 15 queries: {len(short)} (min {short.min() if len(short) else '-'})")
    print(f"written to {OUT}/queries_{args.version}.parquet and signs_{args.version}.parquet")


if __name__ == "__main__":
    main()
