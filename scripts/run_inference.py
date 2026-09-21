"""Phase 5: run one model over the query set in one condition.

    uv run python scripts/run_inference.py --model qwen3-vl-8b --condition A --limit 20 --dry-run
    uv run python scripts/run_inference.py --model qwen3-vl-8b --condition A          # needs a GPU + vllm

Condition A gives the model the sign crop; condition B gives the ground-truth transcription as text.
Everything else — prompt, schema, sampling — is identical (plan §7.4).

Results stream to data/runs/<model>_<condition>.jsonl and are cached in data/runs/cache.sqlite keyed on
(model, condition, image/sign hash, prompt hash, sampling params), so a re-run resumes for free.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image
from pydantic import ValidationError

from vlm_parking.prompts import Answer, build_prompt, render_sign
from vlm_parking.schema import Query, Sign

QUERIES = Path("data/queries")
COLLECT = Path("data/collect")
RUNS = Path("data/runs")

# Hugging Face ids [VERIFY against the model cards before renting a GPU]
MODELS = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3_5-8b": "OpenGVLab/InternVL3_5-8B",
    "minicpm-v-4_5": "openbmb/MiniCPM-V-4_5",
}


def sha(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode())
    return h.hexdigest()


def load_data(version: str) -> tuple[pd.DataFrame, dict[str, Sign], dict[str, str]]:
    queries = pd.read_parquet(QUERIES / f"queries_{version}.parquet")
    signs_df = pd.read_parquet(QUERIES / f"signs_{version}.parquet")
    signs = {r.sign_id: Sign(**json.loads(r.sign_json)) for r in signs_df.itertuples()}
    crops = {r.sign_id: str(COLLECT / r.crop_path) for r in signs_df.itertuples()}
    return queries, signs, crops


def prepare_image(path: str, max_side: int) -> tuple[bytes, tuple[int, int]]:
    """One fixed resize policy for every model, so token counts differ only by each model's own scheme."""
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), im.size


def cache_db() -> sqlite3.Connection:
    RUNS.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(RUNS / "cache.sqlite")
    con.execute(
        """CREATE TABLE IF NOT EXISTS responses (
             key TEXT PRIMARY KEY, model TEXT, condition TEXT, query_id TEXT, raw TEXT, answer_json TEXT,
             parse_ok INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, finish_reason TEXT,
             seconds REAL, created_at TEXT, vllm_version TEXT, model_revision TEXT, gpu TEXT)"""
    )
    have = {r[1] for r in con.execute("PRAGMA table_info(responses)")}
    for col in ("vllm_version", "model_revision", "gpu"):  # caches written before provenance was recorded
        if col not in have:
            con.execute(f"ALTER TABLE responses ADD COLUMN {col} TEXT")
    return con


def provenance(model_id: str, revision: str | None) -> dict[str, str]:
    """What produced these answers. Unrecoverable once the pod is deleted, so it goes in every row.

    The cache key deliberately does not include any of this: mixing engines is caught by the check in
    main() instead, so an engine upgrade doesn't silently invalidate a paid-for cache.
    """
    import torch
    import vllm

    sha = revision or "unknown"
    try:  # the id on its own is mutable; the commit sha is what someone else would have to check out
        from huggingface_hub import model_info

        sha = model_info(model_id, revision=revision).sha
    except Exception as err:  # offline, gated repo, or hub API change - the run is still valid
        print(f"  [warn] could not resolve the revision of {model_id}: {err}")
    return {"vllm_version": vllm.__version__, "model_revision": sha,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def parse_answer(text: str) -> tuple[dict | None, bool]:
    try:
        return Answer.model_validate_json(text).model_dump(), True
    except (ValidationError, ValueError):
        return None, False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--condition", required=True, choices=["A", "B"])
    ap.add_argument("--version", default="v1")
    ap.add_argument("--limit", type=int, help="first N queries (smoke test)")
    ap.add_argument("--max-side", type=int, default=1024, help="longest image side in pixels (condition A)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--revision", help="pin the model to one HF commit sha (recorded either way)")
    ap.add_argument("--allow-mixed-engine", action="store_true",
                    help="resume a run whose cached rows came from a different vLLM version")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and exit; no GPU, no model")
    args = ap.parse_args()

    queries, signs, crops = load_data(args.version)
    if args.limit:
        queries = queries.head(args.limit)
    model_id = MODELS[args.model]
    schema = Answer.model_json_schema()
    settings = json.dumps({"t": args.temperature, "max_tokens": args.max_tokens, "seed": args.seed,
                           "max_side": args.max_side, "schema": sha(json.dumps(schema, sort_keys=True))}, sort_keys=True)

    # Build every request first: text prompt, image (condition A), and the cache key.
    requests = []
    image_cache: dict[str, tuple[bytes, tuple[int, int]]] = {}
    for q in queries.itertuples():
        permit = q.permit_district if isinstance(q.permit_district, str) else None  # parquet nulls arrive as NaN
        query = Query(day=q.day, time=q.time, duration_min=int(q.duration_min), permit_district=permit)
        sign = signs[q.sign_id]
        if args.condition == "A":
            if q.sign_id not in image_cache:
                image_cache[q.sign_id] = prepare_image(crops[q.sign_id], args.max_side)
            image, size = image_cache[q.sign_id]
            prompt, sign_hash = build_prompt(query), sha(image)
        else:
            image, size = None, None
            prompt, sign_hash = build_prompt(query, sign), sha(render_sign(sign))
        requests.append({
            "query_id": q.query_id, "sign_id": q.sign_id, "prompt": prompt, "image": image, "image_size": size,
            "key": sha(model_id, args.condition, sign_hash, prompt, settings),
        })

    out_path = RUNS / f"{args.model}_{args.condition}.jsonl"
    con = cache_db()
    cached = {r[0] for r in con.execute("SELECT key FROM responses WHERE model = ? AND condition = ?", (model_id, args.condition))}
    todo = [r for r in requests if r["key"] not in cached]
    print(f"{args.model} condition {args.condition}: {len(requests):,} queries, {len(requests) - len(todo):,} cached, {len(todo):,} to run")

    if args.dry_run:
        RUNS.mkdir(parents=True, exist_ok=True)
        preview = RUNS / f"prompts_{args.model}_{args.condition}.jsonl"
        with preview.open("w") as f:
            for r in requests:
                f.write(json.dumps({k: v for k, v in r.items() if k != "image"} | {"has_image": r["image"] is not None}) + "\n")
        example = requests[0]
        print(f"\n--- example prompt ({len(example['prompt'].split())} words, image={example['image_size']}) ---\n{example['prompt']}\n---")
        print(f"answer schema: {json.dumps(schema['properties'], indent=1)}")
        print(f"prompts written to {preview}; nothing was run")
        return

    from vllm import LLM, SamplingParams  # imported late: only needed on the GPU box

    prov = provenance(model_id, args.revision)
    print(f"vLLM {prov['vllm_version']} | {prov['gpu']} | {args.model} @ {prov['model_revision'][:12]}")

    # Resuming under a different engine would append new answers beside old ones in one file, with
    # nothing marking the boundary. Better to stop and let the human decide.
    prior = {v for (v,) in con.execute(
        "SELECT DISTINCT vllm_version FROM responses WHERE model = ? AND condition = ? AND vllm_version IS NOT NULL",
        (model_id, args.condition))}
    if prior - {prov["vllm_version"]} and not args.allow_mixed_engine:
        raise SystemExit(
            f"cached answers for {args.model} condition {args.condition} came from vLLM "
            f"{', '.join(sorted(prior))}, but this is {prov['vllm_version']}.\n"
            f"Re-run under the pinned version (requirements-gpu.txt), delete those rows, "
            f"or pass --allow-mixed-engine if you accept mixing them.")

    try:  # the structured-output API was renamed between vLLM versions [VERIFY]
        from vllm.sampling_params import GuidedDecodingParams

        structured = {"guided_decoding": GuidedDecodingParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import StructuredOutputsParams

        structured = {"structured_outputs": StructuredOutputsParams(json=schema)}

    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True, revision=args.revision,
              limit_mm_per_prompt={"image": 1} if args.condition == "A" else {"image": 0}, seed=args.seed)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, **structured)

    conversations = []
    for r in todo:
        content = [{"type": "text", "text": r["prompt"]}]
        if r["image"] is not None:
            data_uri = "data:image/jpeg;base64," + base64.b64encode(r["image"]).decode()
            content = [{"type": "image_url", "image_url": {"url": data_uri}}, *content]
        conversations.append([{"role": "user", "content": content}])

    started = time.time()
    outputs = llm.chat(conversations, params) if conversations else []
    elapsed = time.time() - started

    n_fail = 0
    with out_path.open("a") as f:
        for r, out in zip(todo, outputs):
            text = out.outputs[0].text
            answer, ok = parse_answer(text)
            n_fail += not ok
            row = {
                "query_id": r["query_id"], "sign_id": r["sign_id"], "model": args.model, "model_id": model_id,
                "condition": args.condition, "answer": answer, "raw": text, "parse_ok": ok,
                "prompt_tokens": len(out.prompt_token_ids), "completion_tokens": len(out.outputs[0].token_ids),
                "finish_reason": out.outputs[0].finish_reason, **prov,
            }
            f.write(json.dumps(row) + "\n")
            con.execute(
                """INSERT OR REPLACE INTO responses
                     (key, model, condition, query_id, raw, answer_json, parse_ok, prompt_tokens,
                      completion_tokens, finish_reason, seconds, created_at, vllm_version, model_revision, gpu)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'),?,?,?)""",
                (r["key"], model_id, args.condition, r["query_id"], text, json.dumps(answer), ok,
                 row["prompt_tokens"], row["completion_tokens"], row["finish_reason"], elapsed / max(1, len(todo)),
                 prov["vllm_version"], prov["model_revision"], prov["gpu"]),
            )
        con.commit()

    tokens = sum(len(o.prompt_token_ids) for o in outputs)
    print(f"ran {len(todo):,} in {elapsed / 60:.1f} min ({len(todo) / max(elapsed, 1e-9):.1f}/s) | "
          f"parse failures {n_fail} ({n_fail / max(1, len(todo)):.2%}) | mean prompt tokens {tokens / max(1, len(outputs)):.0f}")

    # One line per invocation: what ran, under what, and how it went. The writeup's methods section.
    with (RUNS / "manifest.jsonl").open("a") as f:
        f.write(json.dumps({
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": args.model,
            "model_id": model_id, "condition": args.condition, "version": args.version, "n_run": len(todo),
            "n_cached": len(requests) - len(todo), "parse_failures": n_fail, "seconds": round(elapsed, 1),
            "max_side": args.max_side, "max_tokens": args.max_tokens, "temperature": args.temperature,
            "seed": args.seed, "max_model_len": args.max_model_len, **prov,
        }) + "\n")
    print(f"appended to {out_path} and {RUNS / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
