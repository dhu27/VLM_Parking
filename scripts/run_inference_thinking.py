"""Phase 5b: run a thinking model over the query set, kept separate from the original runs.

    uv run python scripts/run_inference_thinking.py --condition B --seed 0 --limit 20 --dry-run
    uv run python scripts/run_inference_thinking.py --condition B --seed 0 --limit 20     # smoke test, GPU
    for s in 0 1 2; do uv run python scripts/run_inference_thinking.py --condition B --seed $s; done

A separate script so that run_inference.py, which produced the six original runs, stays exactly as it was.
Everything that isn't about thinking is imported from it unchanged: data loading, the image policy, the
response cache, provenance and answer parsing. What differs, and why:

- The model thinks before it answers. vLLM's reasoning parser holds the JSON grammar back until </think>
  (StructuredOutputsConfig.enable_in_reasoning defaults to False), so the thinking is free text and only the
  final answer is constrained. Without it, the grammar would force JSON from the first token and the model
  would never think. Verified in the vLLM 0.29.0 source; the smoke test is what confirms it works.
- Sampling, not greedy. The model card says greedy='false', temperature 1.0, top_p 0.95, top_k 20; running
  greedy would test the model against its own instructions. Sampled answers vary from run to run, so run
  several seeds, and each seed gets its own file.
- A budget for thinking: 16,384 output tokens. The card allows up to 40,960, sized for competition math;
  answers that run out mid-thought are counted, so the smoke test shows whether this is enough.
- The trace is saved with every answer, so you can read whether the model catches its own mistakes.

Results go to data/runs/<model>_<condition>_s<seed>.jsonl and the shared data/runs/cache.sqlite. The cache key
includes the model id, every sampling setting and the seed, so these runs can never collide with the originals.
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from datetime import datetime, timezone

from run_inference import RUNS, cache_db, load_data, parse_answer, prepare_image, provenance, sha

from vlm_parking.prompts import Answer, build_prompt, render_sign
from vlm_parking.schema import Query
from vlm_parking.thinking import END_THINK, MODEL_CARD_VL, settings_key, split_thinking

# name -> (Hugging Face id, vLLM reasoning parser)
MODELS = {"qwen3-vl-8b-thinking": ("Qwen/Qwen3-VL-8B-Thinking", "qwen3")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="qwen3-vl-8b-thinking")
    ap.add_argument("--condition", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, required=True, help="sampling seed; each seed is its own run and file")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--limit", type=int, help="first N queries (smoke test)")
    ap.add_argument("--max-side", type=int, default=1024, help="longest image side in pixels (condition A)")
    ap.add_argument("--max-tokens", type=int, default=16384, help="thinking plus answer")
    ap.add_argument("--max-model-len", type=int, default=20480)
    ap.add_argument("--temperature", type=float, default=MODEL_CARD_VL["temperature"])
    ap.add_argument("--top-p", type=float, default=MODEL_CARD_VL["top_p"])
    ap.add_argument("--top-k", type=int, default=MODEL_CARD_VL["top_k"])
    ap.add_argument("--presence-penalty", type=float, default=MODEL_CARD_VL["presence_penalty"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--revision", help="pin the model to one HF commit sha (recorded either way)")
    ap.add_argument("--allow-mixed-engine", action="store_true",
                    help="resume a run whose cached rows came from a different vLLM version")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and exit; no GPU, no model")
    args = ap.parse_args()

    queries, signs, crops = load_data(args.version)
    if args.limit:
        queries = queries.head(args.limit)
    model_id, parser = MODELS[args.model]
    schema = Answer.model_json_schema()
    settings = settings_key(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                            presence_penalty=args.presence_penalty, max_tokens=args.max_tokens, seed=args.seed,
                            max_side=args.max_side, schema_hash=sha(json.dumps(schema, sort_keys=True)),
                            reasoning_parser=parser)

    # Build every request. This loop mirrors run_inference.py's, so the prompts are byte-identical to the
    # original runs'; only how the model decodes differs.
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

    run = f"{args.model}_{args.condition}_s{args.seed}"
    out_path = RUNS / f"{run}.jsonl"
    con = cache_db()
    cached = {r[0] for r in con.execute("SELECT key FROM responses WHERE model = ? AND condition = ?", (model_id, args.condition))}
    todo = [r for r in requests if r["key"] not in cached]
    print(f"{run}: {len(requests):,} queries, {len(requests) - len(todo):,} cached, {len(todo):,} to run | "
          f"temperature {args.temperature}, top_p {args.top_p}, top_k {args.top_k}, "
          f"presence_penalty {args.presence_penalty}, max_tokens {args.max_tokens:,}, reasoning parser {parser!r}")

    if args.dry_run:
        RUNS.mkdir(parents=True, exist_ok=True)
        preview = RUNS / f"prompts_{run}.jsonl"
        with preview.open("w") as f:
            for r in requests:
                f.write(json.dumps({k: v for k, v in r.items() if k != "image"} | {"has_image": r["image"] is not None}) + "\n")
        example = requests[0]
        print(f"\n--- example prompt ({len(example['prompt'].split())} words, image={example['image_size']}) ---\n{example['prompt']}\n---")
        print(f"prompts written to {preview}; nothing was run")
        return

    from vllm import LLM, SamplingParams  # imported late: only needed on the GPU box
    from vllm.sampling_params import StructuredOutputsParams

    prov = provenance(model_id, args.revision)
    print(f"vLLM {prov['vllm_version']} | {prov['gpu']} | {args.model} @ {prov['model_revision'][:12]}")

    prior = {v for (v,) in con.execute(
        "SELECT DISTINCT vllm_version FROM responses WHERE model = ? AND condition = ? AND vllm_version IS NOT NULL",
        (model_id, args.condition))}
    if prior - {prov["vllm_version"]} and not args.allow_mixed_engine:
        raise SystemExit(
            f"cached answers for {args.model} condition {args.condition} came from vLLM "
            f"{', '.join(sorted(prior))}, but this is {prov['vllm_version']}.\n"
            f"Re-run under the pinned version (requirements-gpu.txt), delete those rows, "
            f"or pass --allow-mixed-engine if you accept mixing them.")

    # The same answer grammar as the original runs, whitespace locked, applied only once thinking ends.
    structured = StructuredOutputsParams(json=schema, disable_any_whitespace=True)
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True, revision=args.revision,
              limit_mm_per_prompt={"image": 1} if args.condition == "A" else {"image": 0}, seed=args.seed,
              structured_outputs_config={"reasoning_parser": parser})
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                            presence_penalty=args.presence_penalty, max_tokens=args.max_tokens,
                            seed=args.seed,  # per request, so a sampled run is reproducible from its seed
                            structured_outputs=structured)
    end_think_id = llm.get_tokenizer().convert_tokens_to_ids(END_THINK)

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

    n_fail = n_unfinished = 0
    thinking_counts = []
    with out_path.open("a") as f:
        for r, out in zip(todo, outputs):
            o = out.outputs[0]
            thinking, answer_text = split_thinking(o.text)
            answer, ok = parse_answer(answer_text) if answer_text is not None else (None, False)
            ids = list(o.token_ids)
            thinking_tokens = ids.index(end_think_id) if end_think_id in ids else len(ids)
            thinking_counts.append(thinking_tokens)
            n_fail += not ok
            n_unfinished += answer_text is None
            row = {
                "query_id": r["query_id"], "sign_id": r["sign_id"], "model": args.model, "model_id": model_id,
                "condition": args.condition, "seed": args.seed, "answer": answer, "raw": o.text, "parse_ok": ok,
                "thinking": thinking, "thinking_tokens": thinking_tokens, "finished_thinking": answer_text is not None,
                "prompt_tokens": len(out.prompt_token_ids), "completion_tokens": len(ids),
                "finish_reason": o.finish_reason, **prov,
            }
            f.write(json.dumps(row) + "\n")
            con.execute(
                """INSERT OR REPLACE INTO responses
                     (key, model, condition, query_id, raw, answer_json, parse_ok, prompt_tokens,
                      completion_tokens, finish_reason, seconds, created_at, vllm_version, model_revision, gpu)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'),?,?,?)""",
                (r["key"], model_id, args.condition, r["query_id"], o.text, json.dumps(answer), ok,
                 row["prompt_tokens"], row["completion_tokens"], row["finish_reason"], elapsed / max(1, len(todo)),
                 prov["vllm_version"], prov["model_revision"], prov["gpu"]),
            )
        con.commit()

    mean_think = statistics.fmean(thinking_counts) if thinking_counts else 0
    max_think = max(thinking_counts, default=0)
    print(f"ran {len(todo):,} in {elapsed / 60:.1f} min ({len(todo) / max(elapsed, 1e-9):.1f}/s) | "
          f"parse failures {n_fail} ({n_fail / max(1, len(todo)):.2%}), of which ran out mid-thought {n_unfinished} | "
          f"thinking tokens mean {mean_think:,.0f}, max {max_think:,}")

    with (RUNS / "manifest.jsonl").open("a") as f:
        f.write(json.dumps({
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": args.model,
            "model_id": model_id, "condition": args.condition, "version": args.version, "n_run": len(todo),
            "n_cached": len(requests) - len(todo), "parse_failures": n_fail, "seconds": round(elapsed, 1),
            "max_side": args.max_side, "max_tokens": args.max_tokens, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k, "presence_penalty": args.presence_penalty,
            "seed": args.seed, "max_model_len": args.max_model_len, "reasoning_parser": parser,
            "grammar_whitespace": "locked", "ran_out_mid_thought": n_unfinished,
            "thinking_tokens_mean": round(mean_think, 1), "thinking_tokens_max": max_think, **prov,
        }) + "\n")
    print(f"appended to {out_path} and {RUNS / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
