#!/usr/bin/env python3
"""
Integrated Meta-Leaderboard with API Performance
==================================================
Combines the main coding benchmark pipeline (scraper.py) with API-level
performance metrics (TPS, TTFT, total response time) to create a unified
leaderboard that captures both capability and efficiency.

Match priority:
  1. Explicit alias mapping (model_alias_mapping.json) - source of truth
  2. Fuzzy matching (fuzzywuzzy, threshold >= 85) - fallback
  3. No match - API perf fields remain null

Scoring Formula:
  - Benchmark_efficiency (E_bt) = Score² / Completion_Time  [from scraper.py]
  - API_speed_score = TPS / (1.0 + TTFT * 0.5) * 10
  - Combined_score = (norm_benchmark * 0.7) + (norm_speed * 0.3)
  - Speed bonus: +0-10% for models with API speed_rank <= 50

Usage:
  python integrated_pipeline.py           # Full pipeline
  python integrated_pipeline.py --api-only # Only API perf data
  python integrated_pipeline.py --bench-only # Only benchmarks (skip API)
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process

import scraper
import api_perf_scraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("integrated-leaderboard")

BENCHMARK_WEIGHT = 0.7
API_SPEED_WEIGHT = 0.3
SPEED_BONUS_CAP = 50
FUZZY_THRESHOLD = 85
FUZZY_SECOND_BEST_GAP = 15

ALIAS_MAP_PATH = Path(__file__).parent / "data" / "model_alias_mapping.json"


def normalize_to_100(values: list[float], epsilon: float = 0.5) -> list[float]:
    arr = np.array(values)
    v_min, v_max = arr.min(), arr.max()
    if v_max - v_min < 1e-12:
        return [50.0] * len(values)
    normalized = ((arr - v_min) / (v_max - v_min)) * 100.0
    return normalized.clip(min=epsilon).tolist()


def load_alias_mapping() -> dict:
    if ALIAS_MAP_PATH.exists():
        try:
            data = json.loads(ALIAS_MAP_PATH.read_text())
            return data.get("leaderboard_to_api", {})
        except Exception as e:
            log.warning("Failed to load alias mapping: %s", e)
    return {}


def build_api_map(api_perf: list[dict]) -> dict:
    api_map = {}
    for r in api_perf:
        canonical = r["model_canonical"]
        api_map[canonical] = r
        api_map[f"provider/{canonical}"] = r
        api_map[canonical.replace("-", "").replace("/", "")] = r

    return api_map


def find_api_match(lb_model: str, api_map: dict, alias_map: dict) -> tuple[dict | None, str]:
    if lb_model in alias_map:
        api_name = alias_map[lb_model]
        if api_name in api_map:
            return api_map[api_name], "explicit"
        if f"provider/{api_name}" in api_map:
            return api_map[f"provider/{api_name}"], "explicit"
        for key in api_map:
            if api_name in key or key in api_name:
                return api_map[key], "explicit"

    api_keys = list(api_map.keys())
    if not api_keys:
        return None, "no_api_data"

    match_result = process.extractOne(
        lb_model,
        api_keys,
        scorer=fuzz.token_sort_ratio
    )

    if match_result:
        best_match = match_result[0]
        score = match_result[1]
        if score >= FUZZY_THRESHOLD:
            best_matches = process.extract(
                lb_model, api_keys, scorer=fuzz.token_sort_ratio, limit=2
            )
            if len(best_matches) >= 2:
                second_score = best_matches[1][1]
                gap = score - second_score
                if gap >= FUZZY_SECOND_BEST_GAP:
                    return api_map[best_match], f"fuzzy({score})"
                else:
                    log.debug("Fuzzy match %s=%d rejected: gap=%d < %d", lb_model, score, gap, FUZZY_SECOND_BEST_GAP)
                    return None, f"fuzzy_rejected(gap)"

            return api_map[best_match], f"fuzzy({score})"

    return None, "no_match"


def compute_integrated_scores(leaderboard: list[dict], api_perf: list[dict]) -> list[dict]:
    alias_map = load_alias_mapping()
    api_map = build_api_map(api_perf)

    log.info("  -> Explicit alias mappings: %d", len(alias_map))
    log.info("  -> API perf models available: %d", len(api_map))

    match_stats = {"explicit": 0, "fuzzy": 0, "no_match": 0, "no_api_data": 0, "fuzzy_rejected": 0}
    matched_models = []

    bench_scores = [e.get("composite_efficiency_score", 50) for e in leaderboard]
    bench_norm = normalize_to_100(bench_scores)

    speed_scores = []
    match_sources = []
    api_matches = {}

    for entry in leaderboard:
        model = entry.get("model", "")
        perf, source = find_api_match(model, api_map, alias_map)
        match_sources.append(source)
        match_stats[source] = match_stats.get(source, 0) + 1

        if perf:
            speed_scores.append(perf.get("speed_score", 0))
            api_matches[model] = perf
            if source != "no_match":
                matched_models.append(model)
        else:
            speed_scores.append(0)

    speed_norm = normalize_to_100(speed_scores)

    log.info("  -> Match stats: %s", {k: v for k, v in match_stats.items() if v > 0})

    integrated = []
    for i, entry in enumerate(leaderboard):
        model = entry.get("model", "")
        perf = api_matches.get(model, {})
        source = match_sources[i]

        bench_score = bench_norm[i]
        speed_score = speed_norm[i]
        combined = (bench_score * BENCHMARK_WEIGHT) + (speed_score * API_SPEED_WEIGHT)

        speed_rank = perf.get("speed_rank", 999) if perf else 999
        if speed_rank <= SPEED_BONUS_CAP:
            bonus = 1.0 + ((SPEED_BONUS_CAP - speed_rank) / (SPEED_BONUS_CAP * 20))
            combined *= bonus

        entry["integrated_score"] = round(combined, 4)
        entry["benchmark_norm"] = round(bench_score, 4)
        entry["speed_norm"] = round(speed_score, 4)
        entry["speed_rank"] = speed_rank if speed_rank < 999 else None
        entry["ttft_s"] = perf.get("ttft_s") if perf else None
        entry["tps"] = perf.get("tps") if perf else None
        entry["top_provider"] = perf.get("top_provider") if perf else None
        entry["api_match_source"] = source
        integrated.append(entry)

    integrated.sort(key=lambda x: x["integrated_score"], reverse=True)
    for i, entry in enumerate(integrated, 1):
        entry["integrated_rank"] = i

    return integrated


def run_integrated_pipeline(skip_api: bool = False):
    log.info("=" * 60)
    log.info("Integrated Meta-Leaderboard Pipeline")
    log.info("=" * 60)

    log.info("--- Step 1: Run benchmark pipeline ---")
    benchmark_data = scraper.run_pipeline_return()
    leaderboard = benchmark_data.get("leaderboard", [])

    if not leaderboard:
        log.error("Benchmark pipeline returned no data")
        sys.exit(1)

    log.info("  -> %d models in benchmark leaderboard", len(leaderboard))

    if skip_api:
        log.info("Skipping API perf (--bench-only flag)")
        return benchmark_data

    log.info("--- Step 2: Fetch API performance data ---")
    local_records = api_perf_scraper.load_speedrun_data()
    web_records = api_perf_scraper.scrape_third_party(api_perf_scraper._session())
    all_records = local_records + web_records
    log.info("  -> %d API perf records (local=%d, web=%d)", len(all_records), len(local_records), len(web_records))

    api_perf_list = api_perf_scraper.aggregate_api_perf(all_records)
    log.info("  -> %d unique models with API perf", len(api_perf_list))

    log.info("--- Step 3: Compute integrated scores ---")
    integrated = compute_integrated_scores(leaderboard, api_perf_list)

    matched = sum(1 for e in integrated if e.get("ttft_s") is not None)
    log.info("  -> %d/%d models matched with API perf", matched, len(integrated))

    output = {
        "generated_at": benchmark_data.get("generated_at"),
        "pipeline_version": "1.2.0",
        "benchmark_sources": ["vals.ai", "artificial-analysis"],
        "api_sources": ["llm-speedrun", "kickllm"],
        "match_config": {
            "fuzzy_threshold": FUZZY_THRESHOLD,
            "fuzzy_second_best_gap": FUZZY_SECOND_BEST_GAP,
        },
        "weights": {"benchmark": BENCHMARK_WEIGHT, "api_speed": API_SPEED_WEIGHT},
        "integrated_leaderboard": integrated,
        "leaderboard": leaderboard,
    }

    output_path = scraper.DATA_DIR / "integrated_leaderboard.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    log.info("Wrote integrated leaderboard to %s", output_path)

    csv_rows = []
    for entry in integrated:
        csv_rows.append({
            "rank": entry["integrated_rank"],
            "model": entry["model"],
            "integrated_score": entry["integrated_score"],
            "benchmark_score": entry["composite_efficiency_score"],
            "benchmark_norm": entry["benchmark_norm"],
            "speed_norm": entry["speed_norm"],
            "ttft_s": entry.get("ttft_s"),
            "tps": entry.get("tps"),
            "speed_rank": entry.get("speed_rank"),
            "top_provider": entry.get("top_provider"),
            "match_source": entry.get("api_match_source"),
            "n_benchmarks": entry["n_benchmarks"],
        })

    csv_path = scraper.DATA_DIR / "integrated_leaderboard.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    log.info("Wrote CSV to %s", csv_path)

    print("\nTop 15 by Integrated Score:")
    print(f"{'Rank':<5} {'Model':<35} {'Integ':<8} {'Bench':<8} {'Speed':<8} {'TTFT':<8} {'TPS':<10} {'Match'}")
    print("-" * 105)
    for entry in integrated[:15]:
        ttft = f"{entry['ttft_s']:.3f}s" if entry.get("ttft_s") else "N/A"
        tps = f"{entry['tps']:.1f}" if entry.get("tps") else "N/A"
        provider = entry.get("top_provider", "-")
        match = entry.get("api_match_source", "-")
        print(f"{entry['integrated_rank']:<5} {entry['model']:<35} {entry['integrated_score']:<8.2f} {entry['benchmark_norm']:<8.2f} {entry['speed_norm']:<8.2f} {ttft:<8} {tps:<10} {match}")

    print("\n\nAPI Match Summary:")
    explicit = sum(1 for e in integrated if e.get("api_match_source") == "explicit")
    fuzzy = sum(1 for e in integrated if str(e.get("api_match_source", "")).startswith("fuzzy"))
    no_match = sum(1 for e in integrated if e.get("api_match_source") == "no_match")
    print(f"  Explicit: {explicit}")
    print(f"  Fuzzy (safe): {fuzzy}")
    print(f"  No match: {no_match}")
    print(f"  Total with API perf: {explicit + fuzzy}")

    return output


if __name__ == "__main__":
    if "--api-only" in sys.argv:
        api_perf_scraper.run_api_perf_pipeline()
    elif "--bench-only" in sys.argv:
        result = scraper.run_pipeline_return()
        print(f"Processed {len(result.get('leaderboard', []))} models")
    else:
        run_integrated_pipeline()