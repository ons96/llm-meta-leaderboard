#!/usr/bin/env python3
"""
LLM Agentic Coding Meta-Leaderboard Pipeline
=============================================
Scrapes vals.ai (5 coding benchmarks: IOI, LiveCodeBench, SWE-bench Verified,
Terminal-Bench 2.0, Vibe Code Bench v1.1) + Artificial Analysis, applies
efficiency math, deduplicates model names, outputs normalized composite scores.

Architecture:
  1. Scrape vals.ai Astro island SSR data (5 coding benchmarks)
  2. Scrape Artificial Analysis RSC payload + free API
  3. Entity resolution (fuzzy dedup via alias dict + provider/model parsing)
  4. Dynamic outlier rejection per benchmark (Mean-1Std or Q1, whichever is lower)
  5. E_bt = Score^2 / Completion_Time
  6. Min-Max normalize E_bt to 0-100 per benchmark (0.5 epsilon floor)
  7. Coverage-weighted composite aggregation (penalty for few benchmarks)
  8. Atomic write with state-based skip logic (SHA-256 fingerprint)
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bs4
import numpy as np
import pandas as pd
import requests
from fuzzywuzzy import fuzz, process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("meta-leaderboard")

GROUP = os.environ.get("OWNER", "user")
REPO_NAME = "llm-meta-leaderboard"

VALS_BASE = "https://www.vals.ai"
VALS_BENCHMARKS = {
    "ioi": {"url": f"{VALS_BASE}/benchmarks/ioi", "slug": "ioi"},
    "livecodebench": {"url": f"{VALS_BASE}/benchmarks/lcb", "slug": "livecodebench"},
    "swe_bench_verified": {"url": f"{VALS_BASE}/benchmarks/swebench", "slug": "swe_bench_verified"},
    "terminal_bench_2": {"url": f"{VALS_BASE}/benchmarks/terminal-bench-2", "slug": "terminal_bench_2"},
    "vibe_code_bench": {"url": f"{VALS_BASE}/benchmarks/vibe-code", "slug": "vibe_code_bench"},
}

AA_CODING_AGENTS_URL = "https://artificialanalysis.ai/agents/coding-agents"
AA_METHODOLOGY_URL = "https://artificialanalysis.ai/methodology/coding-agents-benchmarking"
AA_MARKET_URL = "https://artificialanalysis.ai/models/capabilities/coding"

DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "last_run_state.json"
TEMP_FILE = DATA_DIR / "temp_data.json"
LEADERBOARD_JSON = DATA_DIR / "leaderboard.json"
LEADERBOARD_CSV = DATA_DIR / "leaderboard.csv"

REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3
RETRY_BACKOFF = 2

MODEL_ALIASES = {
    "claude-opus-4.7": [
        "Claude Opus 4.7", "anthropic/claude-opus-4-7",
        "claude-opus-4-7", "claude-opus-4.7", "Anthropic Claude Opus 4.7",
    ],
    "gpt-5.5": [
        "GPT 5.5", "openai/gpt-5.5", "gpt-5.5", "OpenAI GPT 5.5", "GPT 5.5 (Preview)",
    ],
    "gpt-5.4": [
        "GPT 5.4", "openai/gpt-5.4-2026-03-05", "gpt-5.4", "OpenAI GPT 5.4",
    ],
    "gpt-5.3-codex": [
        "GPT 5.3 Codex", "openai/gpt-5.3-codex", "gpt-5.3-codex", "GPT 5.3 Codex (Preview)",
    ],
    "gpt-5.2-codex": [
        "GPT 5.2 Codex", "openai/gpt-5.2-codex", "gpt-5.2-codex",
    ],
    "gpt-5.2": [
        "GPT 5.2", "openai/gpt-5.2-2025-12-11", "gpt-5.2",
    ],
    "gpt-5.1": [
        "GPT 5.1", "openai/gpt-5.1", "gpt-5.1",
    ],
    "gpt-5": [
        "GPT 5", "openai/gpt-5", "gpt-5",
    ],
    "gpt-5-mini": [
        "GPT 5 Mini", "openai/gpt-5-mini", "gpt-5-mini",
    ],
    "gpt-5.4-mini": [
        "GPT 5.4 Mini", "openai/gpt-5.4-mini", "gpt-5.4-mini",
    ],
    "gpt-5.4-nano": [
        "GPT 5.4 Nano", "openai/gpt-5.4-nano", "gpt-5.4-nano",
    ],
    "claude-sonnet-4.6": [
        "Claude Sonnet 4.6", "anthropic/claude-sonnet-4-6",
        "claude-sonnet-4-6", "claude-sonnet-4.6", "Anthropic Claude Sonnet 4.6",
    ],
    "claude-opus-4.6-thinking": [
        "Claude Opus 4.6 (Thinking)", "claude-opus-4.6-thinking",
        "anthropic/claude-opus-4-6-thinking",
    ],
    "claude-opus-4.5-thinking": [
        "Claude Opus 4.5 (Thinking)", "claude-opus-4.5-thinking",
    ],
    "claude-opus-4.5-nonthinking": [
        "Claude Opus 4.5 (Nonthinking)", "claude-opus-4.5-nonthinking",
    ],
    "claude-haiku-4.5-thinking": [
        "Claude Haiku 4.5 (Thinking)", "claude-haiku-4.5-thinking",
        "anthropic/claude-haiku-4-5-20251001-thinking",
    ],
    "gemini-3.1-pro-preview-0226": [
        "Gemini 3.1 Pro Preview (02/26)", "google/gemini-3.1-pro-preview-0226",
        "gemini-3.1-pro-preview-02-26", "Gemini 3.1 Pro Preview",
    ],
    "gemini-3-pro-1125": [
        "Gemini 3 Pro (11/25)", "google/gemini-3-pro-1125", "gemini-3-pro-11-25",
    ],
    "gemini-3-flash-1225": [
        "Gemini 3 Flash (12/25)", "google/gemini-3-flash-1225", "gemini-3-flash-12-25",
    ],
    "gemini-3-flash-preview": [
        "Gemini 3 Flash Preview", "google/gemini-3-flash-preview",
    ],
    "gemini-2.5-pro": [
        "Gemini 2.5 Pro", "google/gemini-2.5-pro", "gemini-2.5-pro",
    ],
    "deepseek-v4": [
        "DeepSeek V4", "deepseek/deepseek-v4", "deepseek-v4",
        "DeepSeek V4 (Thinking)", "Deepseek V4",
    ],
    "deepseek-v3.2-thinking": [
        "DeepSeek V3.2 (Thinking)", "deepseek-v3.2-thinking",
    ],
    "grok-4.3": [
        "Grok 4.3", "xai/grok-4-3", "grok-4.3", "xAI Grok 4.3",
    ],
    "grok-4.20-reasoning": [
        "Grok 4.20 (Reasoning)", "xai/grok-4-20-reasoning", "grok-4.20-reasoning",
    ],
    "grok-4-fast": [
        "Grok 4 Fast (Reasoning)", "xai/grok-4-fast-reasoning",
        "Grok 4 Fast",
    ],
    "grok-4.1-fast": [
        "Grok 4.1 Fast (Reasoning)", "xai/grok-4-1-fast-reasoning",
    ],
    "grok-4": [
        "Grok 4", "xai/grok-4", "grok-4",
    ],
    "kimi-k2.6": [
        "Kimi K2.6", "moonshot/kimi-k2-6", "kimi-k2.6", "Moonshot AI Kimi K2.6",
    ],
    "kimi-k2.5": [
        "Kimi K2.5", "moonshot/kimi-k2-5", "kimi-k2.5",
    ],
    "kimi-k2-thinking": [
        "Kimi K2 Thinking", "moonshot/kimi-k-2-thinking", "kimi-k2-thinking",
    ],
    "qwen-3.6-plus": [
        "Qwen 3.6 Plus", "alibaba/qwen3-6-plus", "qwen-3.6-plus", "Alibaba Qwen 3.6 Plus",
    ],
    "qwen-3.5-plus": [
        "Qwen 3.5 Plus", "alibaba/qwen3-5-plus", "qwen-3.5-plus",
    ],
    "qwen-3.6-27b": [
        "Qwen 3.6 27B", "alibaba/qwen3-6-27b", "qwen-3.6-27b",
    ],
    "qwen-3.6-max": [
        "Qwen 3.6 Max Preview", "alibaba/qwen3-6-max-preview",
        "alibaba/qwen3-max", "alibaba/qwen3-max-preview",
    ],
    "minimax-m2.1": [
        "MiniMax-M2.1", "minimax/minimax-m2-1", "minimax-m2.1",
    ],
    "minimax-m2.5": [
        "MiniMax-M2.5", "minimax/minimax-m2-5", "minimax-m2.5",
    ],
    "minimax-m2.7": [
        "MiniMax-M2.7", "minimax/minimax-m2-7", "minimax-m2.7",
    ],
    "muse-spark": [
        "Muse Spark", "zai/muse-spark", "muse-spark", "zAI Muse Spark",
    ],
    "glm-5.1": [
        "GLM 5.1", "zhipu/glm-5-1", "glm-5.1",
    ],
    "glm-5": [
        "GLM 5", "zhipu/glm-5", "glm-5",
    ],
    "glm-4.7": [
        "GLM 4.7", "zhipu/glm-4-7", "glm-4.7",
    ],
    "mistral-large-3": [
        "Mistral Large 3", "mistral/mistral-large-3", "mistral-large-3",
    ],
    "mistral-medium-3.5": [
        "Mistral Medium 3.5", "mistral/mistral-medium-3-5", "mistral-medium-3.5",
    ],
    "devstral-2": [
        "Devstral 2", "mistral/devstral-2", "devstral-2", "Mistral Devstral 2",
    ],
    "command-a": [
        "Command A", "cohere/command-a", "command-a", "Cohere Command A",
    ],
    "gpt-oss-120b": [
        "GPT OSS 120B", "openai/gpt-oss-120b", "gpt-oss-120b",
    ],
}


def _build_reverse_alias_map():
    reverse = {}
    for canonical, aliases in MODEL_ALIASES.items():
        for alias in aliases:
            key = alias.lower().strip()
            reverse[key] = canonical
        reverse[canonical.lower().strip()] = canonical
    return reverse


REVERSE_ALIAS = _build_reverse_alias_map()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (f"MetaLeaderboard/1.0 (+https://github.com/{GROUP}/{REPO_NAME})"),
        "Accept": "text/html,application/xhtml+xml,application/xml,application/json,*/*",
    })
    return s


def _fetch_with_retry(url, session, retries=REQUEST_RETRIES):
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt + 1, retries, url, exc)
            if attempt < retries - 1:
                time.sleep(RETRY_BACKOFF ** attempt)
            else:
                raise


# ---------------------------------------------------------------------------
# Scraper 1: Vals.ai (Astro island SSR)
# ---------------------------------------------------------------------------

def _decode_astro(obj):
    if isinstance(obj, list):
        if len(obj) == 2 and isinstance(obj[0], int):
            tag, val = obj
            if tag == 0:
                return _decode_astro(val)
            elif tag == 1:
                return [_decode_astro(item) for item in (val if isinstance(val, list) else [val])]
        return [_decode_astro(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _decode_astro(v) for k, v in obj.items()}
    return obj


def _parse_vals_ssr(html: str) -> dict:
    soup = bs4.BeautifulSoup(html, "html.parser")
    islands = soup.find_all("astro-island")
    for isl in islands:
        props_raw = isl.get("props", "")
        if not props_raw:
            continue
        if "benchmarkView" not in props_raw and len(props_raw) < 2000:
            continue
        try:
            raw = json.loads(props_raw)
        except json.JSONDecodeError:
            continue
        if "benchmarkView" in raw:
            return _decode_astro(raw["benchmarkView"])
        if len(props_raw) > 5000:
            decoded = _decode_astro(raw)
            if isinstance(decoded, dict) and len(decoded) > 2:
                return decoded
    return {}


def _extract_task_overall(tasks: dict) -> dict:
    if isinstance(tasks, dict):
        if "overall" in tasks:
            ov = tasks["overall"]
            return ov if isinstance(ov, dict) else {}
        first_key = next(iter(tasks), None)
        if first_key:
            val = tasks[first_key]
            return val if isinstance(val, dict) else {}
    return {}


def scrape_vals_benchmark(benchmark_key: str, cfg: dict, session) -> list[dict]:
    url = cfg["url"]
    slug = cfg["slug"]
    log.info("Scraping vals.ai: %s (%s)", benchmark_key, url)

    resp = _fetch_with_retry(url, session)
    bv = _parse_vals_ssr(resp.text)

    if not bv:
        log.warning("  No benchmarkView data found for %s", benchmark_key)
        return []

    default = bv.get("default", bv)
    if isinstance(default, dict) and "default" in bv and isinstance(default.get("tasks"), dict):
        pass
    elif "tasks" in bv:
        default = bv
    elif "default" in bv:
        default = bv["default"]

    metadata = default.get("metadata", {})
    benchmark_label = metadata.get("benchmark", slug)
    model_ids = metadata.get("models", [])

    tasks = default.get("tasks", {})
    task_results = _extract_task_overall(tasks)

    if not task_results:
        log.warning("  No task results found for %s (found tasks: %s)", benchmark_key, list(tasks.keys())[:3])
        return []

    rows = []
    for model_slug, result in task_results.items():
        if not isinstance(result, dict):
            continue
        score = result.get("accuracy") or result.get("score") or 0
        latency = result.get("latency") or result.get("avg_time") or 0
        score_val = float(score) if score else 0.0
        latency_val = float(latency) if latency else 0.0

        display_name = model_slug
        if "/" in display_name:
            parts = display_name.split("/", 1)
            provider = parts[0].title()
            rest = parts[1]
            rest = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', rest)
            rest = re.sub(r'-\d{8}$', '', rest)
            rest = rest.replace("-thinking", " (Thinking)")
            rest = rest.replace("-nonthinking", " (Nonthinking)")
            rest = re.sub(r"^[a-z]", lambda m: m.group(0).upper(), rest)
            rest = rest.replace("-preview", " Preview")
            rest = rest.replace("-preview", " Preview")
            display_name = rest.replace("-", " ")

        rows.append({
            "model_raw": display_name,
            "model_slug": model_slug,
            "benchmark": slug,
            "source": "vals_ai",
            "score": score_val,
            "completion_time_s": latency_val,
            "provider": result.get("provider", provider if "/" in model_slug else ""),
        })

    log.info("  -> %d entries from %s", len(rows), benchmark_key)
    return rows


def scrape_vals_ai(session) -> list[dict]:
    all_rows = []
    for key, cfg in VALS_BENCHMARKS.items():
        try:
            rows = scrape_vals_benchmark(key, cfg, session)
            all_rows.extend(rows)
        except Exception as exc:
            log.error("Failed scraping %s: %s", key, exc)
    return all_rows


# ---------------------------------------------------------------------------
# Scraper 2: vals.ai Vibe Index coding benchmarks (via main page)
# ---------------------------------------------------------------------------

def scrape_vals_index_coding(session) -> list[dict]:
    log.info("Scraping vals.ai index page for coding benchmarks")
    all_rows = []

    for benchmark_key, cfg in VALS_BENCHMARKS.items():
        try:
            rows = scrape_vals_benchmark(benchmark_key, cfg, session)
            all_rows.extend(rows)
        except Exception as exc:
            log.warning("Could not scrape %s from vals.ai: %s", benchmark_key, exc)

    return all_rows


# ---------------------------------------------------------------------------
# Scraper 3: Artificial Analysis (Next.js RSC / free API)
# ---------------------------------------------------------------------------

def _parse_rsc_row_obj(obj, idx_label=None):
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("model_name") or obj.get("agent_name") or ""
    score = obj.get("index_score") or obj.get("pass_rate") or obj.get("score") or obj.get("coding_index") or 0
    time_val = obj.get("time_per_task") or obj.get("execution_time") or obj.get("completion_time") or obj.get("median_time_to_first_token_seconds") or 0
    if isinstance(score, (int, float)) and score > 0:
        return {
            "model_raw": str(name),
            "benchmark": "aa_coding_agents",
            "source": "artificial_analysis",
            "score": float(score) * 100 if score < 1 else float(score),
            "completion_time_s": float(time_val) if time_val else 0,
        }
    return None


def scrape_artificial_analysis(session) -> list[dict]:
    log.info("Scraping Artificial Analysis")
    all_rows = []

    try:
        resp = _fetch_with_retry(AA_CODING_AGENTS_URL, session)
        html = resp.text
        soup = bs4.BeautifulSoup(html, "html.parser")

        chunks = []
        for script in soup.find_all("script"):
            if script.string and "self.__next_f.push" in script.string:
                chunks.append(script.string)

        all_chunks_text = " ".join(chunks)
        data_objs = re.findall(
            r'\{"name"\s*:\s*"[^"]+?"[^{}]*?"(?:index_score|pass_rate|score|time_per_task|execution_time)"\s*:\s*[\d.]+[^{}]*\}',
            all_chunks_text,
        )
        for obj_str in data_objs:
            try:
                obj = json.loads(obj_str)
                row = _parse_rsc_row_obj(obj)
                if row:
                    all_rows.append(row)
            except json.JSONDecodeError:
                continue

        if data_objs:
            log.info("  -> Found %d agent data objects via RSC regex", len(data_objs))
        else:
            log.warning("  -> No agent data found in RSC payload")

    except Exception as exc:
        log.warning("AA scraping failed: %s", exc)

    log.info("  -> Total AA entries: %d", len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# Entity Resolution
# ---------------------------------------------------------------------------

def normalize_model_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    name = re.sub(r"[^\w\s\.\-\(\)]", "", name)
    return name


def resolve_model_name(raw_name: str) -> str:
    normalized = normalize_model_name(raw_name)
    key = normalized.lower().strip()

    if key in REVERSE_ALIAS:
        return REVERSE_ALIAS[key]

    for alias_key, canonical in REVERSE_ALIAS.items():
        if fuzz.partial_ratio(key, alias_key) >= 95:
            return canonical
        if fuzz.token_sort_ratio(key, alias_key) >= 90:
            return canonical

    best, score = process.extractOne(key, REVERSE_ALIAS.keys(), scorer=fuzz.token_sort_ratio)
    if score >= 88:
        return REVERSE_ALIAS[best]

    canonical = re.sub(r"[\s\(\)]+", "-", key)
    canonical = re.sub(r"-+", "-", canonical).strip("-")
    return canonical


def deduplicate_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["model_canonical"] = df["model_raw"].apply(resolve_model_name)
    deduped = []
    for (model, benchmark), group in df.groupby(["model_canonical", "benchmark"]):
        if len(group) > 1:
            best = group.loc[group["score"].idxmax()]
            deduped.append(best)
        else:
            deduped.append(group.iloc[0])
    return pd.DataFrame(deduped).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Math Pipeline
# ---------------------------------------------------------------------------

def reject_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    surviving = []
    for benchmark, group in df.groupby("benchmark"):
        scores = group["score"].dropna()
        if len(scores) < 4:
            surviving.append(group)
            continue
        mean, std = scores.mean(), scores.std()
        q1 = scores.quantile(0.25)
        threshold = min(mean - 1.0 * std, q1)
        mask = group["score"] >= threshold
        kept = group[mask]
        n_rejected = len(group) - len(kept)
        if n_rejected > 0:
            log.info("Outlier %s: removed %d below %.2f (mean=%.2f, std=%.2f, q1=%.2f)", benchmark, n_rejected, threshold, mean, std, q1)
        surviving.append(kept)
    return pd.concat(surviving).reset_index(drop=True)


def calculate_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score_decimal"] = df["score"] / 100.0
    default_times = {
        "ioi": 600, "livecodebench": 120, "swe_bench_verified": 900,
        "terminal_bench_2": 300, "vibe_code_bench": 600,
        "aa_coding_agents": 300, "aa_coding_index": 300, "aa_livecodebench": 120,
    }
    df["completion_time_s"] = df.apply(
        lambda r: r["completion_time_s"] if r["completion_time_s"] > 0 else default_times.get(r["benchmark"], 300),
        axis=1,
    )
    df["e_bt"] = (df["score_decimal"] ** 2) / df["completion_time_s"]
    return df


def normalize_per_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["normalized_e"] = np.nan
    EPSILON = 0.5
    for benchmark, group in df.groupby("benchmark"):
        e_vals = group["e_bt"]
        e_min, e_max = e_vals.min(), e_vals.max()
        if e_max - e_min < 1e-12:
            df.loc[group.index, "normalized_e"] = 50.0
        else:
            raw = ((e_vals - e_min) / (e_max - e_min)) * 100.0
            df.loc[group.index, "normalized_e"] = raw.clip(lower=EPSILON)
    return df


def aggregate_composite(df: pd.DataFrame) -> pd.DataFrame:
    MAX_BENCHMARKS = len(VALS_BENCHMARKS) + 1
    records = []
    for model, group in df.groupby("model_canonical"):
        n = group["benchmark"].nunique()
        avg_norm = group["normalized_e"].mean()
        penalty = {1: 0.5, 2: 0.7, 3: 0.85, 4: 0.95}.get(n, 1.0)
        composite = avg_norm * penalty
        records.append({
            "model": model,
            "composite_efficiency_score": round(composite, 4),
            "avg_normalized_e": round(avg_norm, 4),
            "participation_penalty": penalty,
            "n_benchmarks": n,
            "benchmarks_participated": sorted(group["benchmark"].unique().tolist()),
            "benchmark_details": [],
        })
    result = pd.DataFrame(records)
    result = result.sort_values("composite_efficiency_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    for idx, row in result.iterrows():
        model = row["model"]
        details = []
        for _, bd in df[df["model_canonical"] == model].iterrows():
            details.append({
                "benchmark": bd["benchmark"],
                "score": round(bd["score"], 4),
                "completion_time_s": round(bd["completion_time_s"], 2),
                "e_bt": round(bd["e_bt"], 6),
                "normalized_e": round(bd["normalized_e"], 4),
                "source": bd["source"],
            })
        result.at[idx, "benchmark_details"] = details
    return result


# ---------------------------------------------------------------------------
# GitOps
# ---------------------------------------------------------------------------

def _compute_fingerprint(data: list) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def check_should_run(new_data: list) -> bool:
    state = _load_state()
    fp = _compute_fingerprint(new_data)
    if fp == state.get("data_fingerprint", ""):
        log.info("Data unchanged. Skipping.")
        return False
    log.info("Data changed. Will update.")
    return True


def validate_output(data: dict) -> bool:
    if not isinstance(data, dict) or not isinstance(data.get("leaderboard"), list):
        return False
    lb = data["leaderboard"]
    if len(lb) == 0:
        return False
    required = {"model", "composite_efficiency_score", "rank"}
    for i, entry in enumerate(lb):
        if not isinstance(entry, dict) or not required.issubset(entry.keys()):
            log.error("Validation FAILED at entry %d", i)
            return False
        if not isinstance(entry.get("composite_efficiency_score"), (int, float)):
            return False
    log.info("Validation PASSED: %d records", len(lb))
    return True


def atomic_write(data: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_FILE.write_text(json.dumps(data, indent=2))

    if not validate_output(data):
        log.error("Validation FAILED. Old data preserved.")
        sys.exit(1)

    LEADERBOARD_JSON.write_text(json.dumps(data, indent=2))
    flat_rows = []
    for e in data["leaderboard"]:
        flat_rows.append({
            "rank": e["rank"],
            "model": e["model"],
            "composite_efficiency_score": e["composite_efficiency_score"],
            "avg_normalized_e": e["avg_normalized_e"],
            "participation_penalty": e["participation_penalty"],
            "n_benchmarks": e["n_benchmarks"],
            "benchmarks_participated": "|".join(e["benchmarks_participated"]),
        })
    pd.DataFrame(flat_rows).to_csv(LEADERBOARD_CSV, index=False)
    log.info("Wrote %d rows to leaderboard.json/csv", len(flat_rows))
    try:
        TEMP_FILE.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline():
    log.info("=" * 60)
    log.info("LLM Meta-Leaderboard Pipeline")
    log.info("=" * 60)

    session = _session()

    log.info("--- Phase 1: Data Ingestion ---")
    vals_rows = scrape_vals_ai(session)
    aa_rows = scrape_artificial_analysis(session)
    all_rows = vals_rows + aa_rows

    if not all_rows:
        log.error("No data from any source. Aborting.")
        sys.exit(1)

    log.info("Total raw rows: %d (vals=%d, aa=%d)", len(all_rows), len(vals_rows), len(aa_rows))

    log.info("--- Phase 2: Entity Resolution ---")
    df = pd.DataFrame(all_rows)
    df = deduplicate_models(df)
    log.info("After dedup: %d rows, %d unique models", len(df), df["model_canonical"].nunique())

    log.info("--- Phase 3: Outlier Rejection ---")
    df = reject_outliers(df)
    log.info("After outliers: %d rows", len(df))

    log.info("--- Phase 4: Efficiency (E_bt) ---")
    df = calculate_efficiency(df)

    log.info("--- Phase 5: Normalization ---")
    df = normalize_per_benchmark(df)

    log.info("--- Phase 6: Composite ---")
    result_df = aggregate_composite(df)
    log.info("%d models scored", len(result_df))

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0.0",
        "n_models": len(result_df),
        "n_benchmarks": int(df["benchmark"].nunique()),
        "n_benchmarks_expected": len(VALS_BENCHMARKS),
        "leaderboard": result_df.to_dict(orient="records"),
    }

    log.info("--- Phase 7: State check & Atomic write ---")
    serialized = [
        {"model": r["model"], "benchmark": d["benchmark"], "score": d["score"], "e_bt": d["e_bt"]}
        for r in output["leaderboard"]
        for d in r.get("benchmark_details", [])
    ]

    if not check_should_run(serialized):
        sys.exit(0)

    atomic_write(output)

    state = _load_state()
    state["data_fingerprint"] = _compute_fingerprint(serialized)
    state["last_run_at"] = output["generated_at"]
    state["n_models"] = len(result_df)
    _save_state(state)

    log.info("=" * 60)
    log.info("Pipeline complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
