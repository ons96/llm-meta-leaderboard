#!/usr/bin/env python3
"""
LLM Agentic Coding Meta-Leaderboard Pipeline
=============================================
Scrapes vals.ai + Artificial Analysis, applies efficiency math,
deduplicates model names, and outputs normalized composite scores.

Architecture:
  1. Scrape vals.ai benchmark pages (5 coding benchmarks)
  2. Scrape Artificial Analysis Coding Agent Index
  3. Entity resolution (fuzzy dedup via alias dict)
  4. Dynamic outlier rejection per benchmark
  5. E_bt = Score^2 / Completion_Time
  6. Min-Max normalize E_bt to 0-100 per benchmark
  7. Coverage-weighted composite aggregation
  8. Atomic write with state-based skip logic
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALS_BASE = "https://www.vals.ai"
VALS_BENCHMARKS = {
    "ioi": {"url": f"{VALS_BASE}/benchmarks/ioi", "slug": "ioi"},
    "livecodebench": {"url": f"{VALS_BASE}/benchmarks/lcb", "slug": "livecodebench"},
    "swe_bench_verified": {
        "url": f"{VALS_BASE}/benchmarks/swebench",
        "slug": "swe_bench_verified",
    },
    "terminal_bench_2": {
        "url": f"{VALS_BASE}/benchmarks/terminal-bench-2",
        "slug": "terminal_bench_2",
    },
    "vibe_code_bench_v1_1": {
        "url": f"{VALS_BASE}/benchmarks/vibe-code-bench",
        "slug": "vibe_code_bench_v1_1",
    },
}

AA_CODING_AGENTS_URL = "https://artificialanalysis.ai/agents/coding-agents"
AA_API_BASE = "https://artificialanalysis.ai/api"

DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "last_run_state.json"
TEMP_FILE = DATA_DIR / "temp_data.json"
LEADERBOARD_JSON = DATA_DIR / "leaderboard.json"
LEADERBOARD_CSV = DATA_DIR / "leaderboard.csv"

REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3
RETRY_BACKOFF = 2

# ---------------------------------------------------------------------------
# Alias dictionary for entity resolution
# ---------------------------------------------------------------------------

MODEL_ALIASES = {
    "claude-opus-4.7": [
        "Claude Opus 4.7",
        "Claude Opus 4.7 (Thinking)",
        "claude-opus-4-7",
        "claude-opus-4.7",
        "Anthropic Claude Opus 4.7",
    ],
    "gpt-5.5": [
        "GPT 5.5",
        "GPT-5.5",
        "gpt-5.5",
        "OpenAI GPT 5.5",
        "GPT 5.5 (Preview)",
    ],
    "gpt-5.4": [
        "GPT 5.4",
        "GPT-5.4",
        "gpt-5.4",
        "OpenAI GPT 5.4",
    ],
    "gpt-5.3-codex": [
        "GPT 5.3 Codex",
        "GPT-5.3 Codex",
        "gpt-5.3-codex",
        "GPT 5.3 Codex (Preview)",
    ],
    "gpt-5.2-codex": [
        "GPT 5.2 Codex",
        "GPT-5.2 Codex",
        "gpt-5.2-codex",
    ],
    "gpt-5.2": [
        "GPT 5.2",
        "GPT-5.2",
        "gpt-5.2",
    ],
    "gpt-5.1": [
        "GPT 5.1",
        "GPT-5.1",
        "gpt-5.1",
    ],
    "gpt-5": [
        "GPT 5",
        "GPT-5",
        "gpt-5",
    ],
    "gpt-5-mini": [
        "GPT 5 Mini",
        "GPT-5 Mini",
        "gpt-5-mini",
    ],
    "gpt-5.4-mini": [
        "GPT 5.4 Mini",
        "GPT-5.4 Mini",
        "gpt-5.4-mini",
    ],
    "gpt-5.4-nano": [
        "GPT 5.4 Nano",
        "GPT-5.4 Nano",
        "gpt-5.4-nano",
    ],
    "claude-sonnet-4.6": [
        "Claude Sonnet 4.6",
        "claude-sonnet-4-6",
        "claude-sonnet-4.6",
        "Anthropic Claude Sonnet 4.6",
    ],
    "claude-sonnet-4.5-thinking": [
        "Claude Sonnet 4.5 (Thinking)",
        "claude-sonnet-4.5-thinking",
    ],
    "claude-opus-4.6-thinking": [
        "Claude Opus 4.6 (Thinking)",
        "claude-opus-4.6-thinking",
        "Claude Opus 4.6 Thinking",
    ],
    "claude-opus-4.5-thinking": [
        "Claude Opus 4.5 (Thinking)",
        "claude-opus-4.5-thinking",
    ],
    "claude-opus-4.5-nonthinking": [
        "Claude Opus 4.5 (Nonthinking)",
        "Claude Opus 4.5 Nonthinking",
        "claude-opus-4.5-nonthinking",
    ],
    "claude-haiku-4.5-thinking": [
        "Claude Haiku 4.5 (Thinking)",
        "claude-haiku-4.5-thinking",
    ],
    "gemini-3.1-pro-preview-0226": [
        "Gemini 3.1 Pro Preview (02/26)",
        "gemini-3.1-pro-preview-02-26",
        "Gemini 3.1 Pro Preview",
    ],
    "gemini-3-pro-1125": [
        "Gemini 3 Pro (11/25)",
        "gemini-3-pro-11-25",
    ],
    "gemini-3-flash-1225": [
        "Gemini 3 Flash (12/25)",
        "gemini-3-flash-12-25",
    ],
    "gemini-2.5-pro": [
        "Gemini 2.5 Pro",
        "gemini-2.5-pro",
    ],
    "deepseek-v4": [
        "DeepSeek V4",
        "DeepSeek V4 (Thinking)",
        "deepseek-v4",
        "Deepseek V4",
    ],
    "deepseek-v3.2-thinking": [
        "DeepSeek V3.2 (Thinking)",
        "deepseek-v3.2-thinking",
    ],
    "grok-4.3": [
        "Grok 4.3",
        "grok-4.3",
        "xAI Grok 4.3",
    ],
    "grok-4.20-reasoning": [
        "Grok 4.20 (Reasoning)",
        "grok-4.20-reasoning",
    ],
    "grok-4": [
        "Grok 4",
        "grok-4",
    ],
    "kimi-k2.6": [
        "Kimi K2.6",
        "kimi-k2.6",
        "Moonshot AI Kimi K2.6",
    ],
    "kimi-k2.5": [
        "Kimi K2.5",
        "kimi-k2.5",
    ],
    "kimi-k2-thinking": [
        "Kimi K2 Thinking",
        "kimi-k2-thinking",
    ],
    "qwen-3.6-plus": [
        "Qwen 3.6 Plus",
        "qwen-3.6-plus",
        "Alibaba Qwen 3.6 Plus",
    ],
    "qwen-3.5-plus": [
        "Qwen 3.5 Plus",
        "qwen-3.5-plus",
    ],
    "qwen-3.6-27b": [
        "Qwen 3.6 27B",
        "qwen-3.6-27b",
    ],
    "minimax-m2.1": [
        "MiniMax-M2.1",
        "minimax-m2.1",
    ],
    "minimax-m2.5": [
        "MiniMax-M2.5",
        "minimax-m2.5",
    ],
    "minimax-m2.7": [
        "MiniMax-M2.7",
        "minimax-m2.7",
    ],
    "muse-spark": [
        "Muse Spark",
        "muse-spark",
        "zAI Muse Spark",
    ],
    "glm-5.1": [
        "GLM 5.1",
        "glm-5.1",
    ],
    "glm-5": [
        "GLM 5",
        "glm-5",
    ],
    "glm-4.7": [
        "GLM 4.7",
        "glm-4.7",
    ],
    "mistral-large-3": [
        "Mistral Large 3",
        "mistral-large-3",
    ],
    "mistral-medium-3.5": [
        "Mistral Medium 3.5",
        "mistral-medium-3.5",
    ],
    "devstral-2": [
        "Devstral 2",
        "devstral-2",
        "Mistral Devstral 2",
    ],
    "command-a": [
        "Command A",
        "command-a",
        "Cohere Command A",
    ],
    "gpt-oss-120b": [
        "GPT OSS 120B",
        "gpt-oss-120b",
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
# HTTP helpers
# ---------------------------------------------------------------------------


def _session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "MetaLeaderboard/1.0 (+https://github.com/user/llm-meta-leaderboard)",
            "Accept": "text/html,application/xhtml+xml,application/xml,application/json,*/*",
        }
    )
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
# Scraper 1: Vals.ai (Next.js __NEXT_DATA__ extraction)
# ---------------------------------------------------------------------------


def _extract_next_data(html: str) -> dict:
    soup = bs4.BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script:
        raise ValueError("No __NEXT_DATA__ script tag found in page HTML")
    return json.loads(script.string)


def _parse_vals_benchmark_page(benchmark_key: str, cfg: dict, session) -> list[dict]:
    url = cfg["url"]
    slug = cfg["slug"]
    log.info("Scraping vals.ai benchmark: %s (%s)", benchmark_key, url)

    resp = _fetch_with_retry(url, session)
    html = resp.text

    next_data = _extract_next_data(html)

    props = next_data.get("props", {}).get("pageProps", {})
    if not props:
        props = (
            next_data.get("props", {})
            .get("pageProps", {})
            .get("benchmark", {})
        )

    models_data = props.get("models", props.get("results", []))
    if not models_data:
        models_key = None
        for k, v in props.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                if any(score_key in v[0] for score_key in ("score", "accuracy", "pass_rate")):
                    models_key = k
                    break
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, list) and len(sv) > 0 and isinstance(sv[0], dict):
                        if any(
                            score_key in sv[0]
                            for score_key in ("score", "accuracy", "pass_rate")
                        ):
                            models_key = sk
                            models_data = sv
                            break
        if models_key:
            log.info("Found models data under key: %s", models_key)

    rows = []
    if isinstance(models_data, list):
        for entry in models_data:
            model_name = entry.get("model_name", entry.get("name", entry.get("model", "")))
            score = float(
                entry.get("score", entry.get("accuracy", entry.get("pass_rate", 0)))
            )
            latency = float(
                entry.get(
                    "latency",
                    entry.get("completion_time", entry.get("time", entry.get("avg_time", 0))),
                )
            )
            if latency == 0:
                latency = float(
                    entry.get(
                        "avg_latency",
                        entry.get("mean_time", 0),
                    )
                )
            rows.append(
                {
                    "model_raw": model_name,
                    "benchmark": slug,
                    "source": "vals_ai",
                    "score": score,
                    "completion_time_s": latency,
                }
            )
    elif isinstance(models_data, dict):
        for model_name, data in models_data.items():
            if isinstance(data, dict):
                score = float(data.get("score", data.get("accuracy", 0)))
                latency = float(
                    data.get("latency", data.get("completion_time", data.get("time", 0)))
                )
            else:
                continue
            rows.append(
                {
                    "model_raw": model_name,
                    "benchmark": slug,
                    "source": "vals_ai",
                    "score": score,
                    "completion_time_s": latency,
                }
            )

    log.info("  -> Extracted %d model entries for %s", len(rows), benchmark_key)
    return rows


def _fallback_scrape_vals_table(benchmark_key: str, cfg: dict, session) -> list[dict]:
    url = cfg["url"]
    slug = cfg["slug"]
    log.info("Fallback: scraping vals.ai HTML table for %s", benchmark_key)

    resp = _fetch_with_retry(url, session)
    soup = bs4.BeautifulSoup(resp.text, "html.parser")

    rows = []
    tables = soup.find_all("table")
    for table in tables:
        tbody = table.find("tbody")
        if not tbody:
            continue
        for tr in tbody.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                model_name = tds[0].get_text(strip=True)
                score_text = tds[1].get_text(strip=True).rstrip("%")
                try:
                    score = float(score_text)
                except ValueError:
                    continue
                latency = 0.0
                if len(tds) >= 3:
                    lat_text = tds[2].get_text(strip=True).replace("s", "").replace(",", "")
                    try:
                        latency = float(lat_text)
                    except ValueError:
                        pass
                rows.append(
                    {
                        "model_raw": model_name,
                        "benchmark": slug,
                        "source": "vals_ai",
                        "score": score,
                        "completion_time_s": latency,
                    }
                )

    log.info("  -> Fallback extracted %d entries for %s", len(rows), benchmark_key)
    return rows


def scrape_vals_ai(session) -> list[dict]:
    all_rows = []
    for key, cfg in VALS_BENCHMARKS.items():
        try:
            rows = _parse_vals_benchmark_page(key, cfg, session)
            if not rows:
                rows = _fallback_scrape_vals_table(key, cfg, session)
            all_rows.extend(rows)
        except Exception as exc:
            log.error("Failed to scrape vals.ai/%s: %s", key, exc)
            try:
                rows = _fallback_scrape_vals_table(key, cfg, session)
                all_rows.extend(rows)
            except Exception as exc2:
                log.error("Fallback also failed for vals.ai/%s: %s", key, exc2)
    return all_rows


# ---------------------------------------------------------------------------
# Scraper 2: Artificial Analysis Coding Agent Index
# ---------------------------------------------------------------------------


def scrape_artificial_analysis(session) -> list[dict]:
    log.info("Scraping Artificial Analysis coding agents page")
    all_rows = []

    try:
        resp = _fetch_with_retry(
            f"{AA_API_BASE}/data/llms/models", session
        )
        api_data = resp.json()
        models = api_data.get("data", [])
        for m in models:
            evals = m.get("evaluations", {})
            coding_idx = evals.get("artificial_analysis_coding_index", 0)
            lcb = evals.get("livecodebench", 0)
            speed = m.get("median_output_tokens_per_second", 0)
            ttft = m.get("median_time_to_first_token_seconds", 0)
            if coding_idx > 0:
                est_time = 1.0 / speed if speed > 0 else 0.0
                all_rows.append(
                    {
                        "model_raw": m.get("name", ""),
                        "benchmark": "aa_coding_index",
                        "source": "artificial_analysis",
                        "score": coding_idx,
                        "completion_time_s": est_time,
                    }
                )
            if lcb > 0:
                est_time = 1.0 / speed if speed > 0 else 0.0
                all_rows.append(
                    {
                        "model_raw": m.get("name", ""),
                        "benchmark": "aa_livecodebench",
                        "source": "artificial_analysis",
                        "score": lcb * 100,
                        "completion_time_s": est_time,
                    }
                )
    except Exception as exc:
        log.error("Failed to scrape AA API: %s", exc)

    try:
        resp = _fetch_with_retry(AA_CODING_AGENTS_URL, session)
        html = resp.text
        next_data = _extract_next_data(html)
        props = next_data.get("props", {}).get("pageProps", {})
        agent_results = props.get("agents", props.get("results", []))
        if isinstance(agent_results, list):
            for entry in agent_results:
                model_name = entry.get("agent_name", entry.get("model", entry.get("name", "")))
                score = float(
                    entry.get("index_score", entry.get("pass_rate", entry.get("score", 0)))
                )
                time_per_task = float(
                    entry.get("time_per_task", entry.get("execution_time", 0))
                )
                cost = float(entry.get("cost_per_task", 0))
                if score > 0:
                    all_rows.append(
                        {
                            "model_raw": model_name,
                            "benchmark": "aa_coding_agents",
                            "source": "artificial_analysis",
                            "score": score,
                            "completion_time_s": time_per_task,
                        }
                    )
    except Exception as exc:
        log.warning("Could not extract AA coding agents page data: %s", exc)

    log.info("  -> Scraped %d entries from Artificial Analysis", len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# Entity Resolution
# ---------------------------------------------------------------------------


def normalize_model_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^\w\s\.\-]", "", name)
    return name


def resolve_model_name(raw_name: str) -> str:
    normalized = normalize_model_name(raw_name)
    key = normalized.lower().strip()

    if key in REVERSE_ALIAS:
        return REVERSE_ALIAS[key]

    best_match, best_score = process.extractOne(
        key, REVERSE_ALIAS.keys(), scorer=fuzz.token_sort_ratio
    )
    if best_score >= 88:
        return REVERSE_ALIAS[best_match]

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
# Mathematical Pipeline
# ---------------------------------------------------------------------------


def reject_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    surviving_indices = []
    for benchmark, group in df.groupby("benchmark"):
        scores = group["score"].dropna()
        if len(scores) < 3:
            surviving_indices.extend(group.index.tolist())
            continue
        mean = scores.mean()
        std = scores.std()
        q1 = scores.quantile(0.25)
        threshold = min(mean - 1.0 * std, q1)
        mask = group["score"] >= threshold
        surviving = group[mask]
        rejected_count = len(group) - len(surviving)
        if rejected_count > 0:
            log.info(
                "Outlier rejection for %s: removed %d models below %.2f (mean=%.2f, std=%.2f, q1=%.2f)",
                benchmark, rejected_count, threshold, mean, std, q1,
            )
        surviving_indices.extend(surviving.index.tolist())

    return df.loc[surviving_indices].reset_index(drop=True)


def calculate_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["score_decimal"] = df["score"] / 100.0

    default_times = {
        "ioi": 600,
        "livecodebench": 120,
        "swe_bench_verified": 900,
        "terminal_bench_2": 300,
        "vibe_code_bench_v1_1": 600,
        "aa_coding_index": 300,
        "aa_livecodebench": 120,
        "aa_coding_agents": 300,
    }

    def _fill_time(row):
        if row["completion_time_s"] > 0:
            return row["completion_time_s"]
        return default_times.get(row["benchmark"], 300)

    df["completion_time_s"] = df.apply(_fill_time, axis=1)

    df["e_bt"] = (df["score_decimal"] ** 2) / df["completion_time_s"]

    return df


def normalize_per_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["normalized_e"] = np.nan
    EPSILON = 0.5

    for benchmark, group in df.groupby("benchmark"):
        e_vals = group["e_bt"]
        e_min = e_vals.min()
        e_max = e_vals.max()
        if e_max - e_min < 1e-12:
            df.loc[group.index, "normalized_e"] = 50.0
        else:
            raw_norm = ((e_vals - e_min) / (e_max - e_min)) * 100.0
            df.loc[group.index, "normalized_e"] = raw_norm.clip(lower=EPSILON)

    return df


def aggregate_composite(df: pd.DataFrame) -> pd.DataFrame:
    MAX_BENCHMARKS = len(VALS_BENCHMARKS) + 3

    records = []
    for model, group in df.groupby("model_canonical"):
        n_benchmarks = group["benchmark"].nunique()
        avg_normalized = group["normalized_e"].mean()

        coverage_weight = min(n_benchmarks / MAX_BENCHMARKS, 1.0)
        penalty = 1.0
        if n_benchmarks == 1:
            penalty = 0.5
        elif n_benchmarks == 2:
            penalty = 0.7
        elif n_benchmarks == 3:
            penalty = 0.85
        elif n_benchmarks == 4:
            penalty = 0.95

        composite = avg_normalized * penalty

        records.append(
            {
                "model": model,
                "composite_efficiency_score": round(composite, 4),
                "avg_normalized_e": round(avg_normalized, 4),
                "coverage_weight": round(coverage_weight, 4),
                "participation_penalty": penalty,
                "n_benchmarks": n_benchmarks,
                "benchmarks_participated": sorted(group["benchmark"].unique().tolist()),
                "benchmark_details": [],
            }
        )

    result = pd.DataFrame(records)
    result = result.sort_values(
        "composite_efficiency_score", ascending=False
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    for idx, row in result.iterrows():
        model = row["model"]
        model_data = df[df["model_canonical"] == model]
        details = []
        for _, bd in model_data.iterrows():
            details.append(
                {
                    "benchmark": bd["benchmark"],
                    "score": round(bd["score"], 4),
                    "completion_time_s": round(bd["completion_time_s"], 2),
                    "e_bt": round(bd["e_bt"], 6),
                    "normalized_e": round(bd["normalized_e"], 4),
                    "source": bd["source"],
                }
            )
        result.at[idx, "benchmark_details"] = details

    return result


# ---------------------------------------------------------------------------
# State / Diff Checking (GitOps)
# ---------------------------------------------------------------------------


def _compute_state_fingerprint(data: list[dict]) -> str:
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def check_should_run(new_data: list[dict]) -> bool:
    state = _load_state()
    new_fingerprint = _compute_state_fingerprint(new_data)
    old_fingerprint = state.get("data_fingerprint", "")

    if new_fingerprint == old_fingerprint:
        log.info("Data fingerprint unchanged (%s). Skipping update.", new_fingerprint[:12])
        return False

    log.info(
        "Data fingerprint changed: %s -> %s. Will update.",
        old_fingerprint[:12] if old_fingerprint else "(none)",
        new_fingerprint[:12],
    )
    return True


def _check_http_freshness(session) -> bool:
    log.info("Checking HTTP freshness headers...")
    try:
        resp = session.head(VALS_BASE + "/benchmarks", timeout=15, allow_redirects=True)
        etag = resp.headers.get("ETag", "")
        last_modified = resp.headers.get("Last-Modified", "")
        state = _load_state()
        old_etag = state.get("vals_etag", "")
        old_lm = state.get("vals_last_modified", "")
        if etag and etag == old_etag:
            log.info("ETag unchanged: %s. May skip.", etag)
            return False
        if last_modified and last_modified == old_lm:
            log.info("Last-Modified unchanged: %s. May skip.", last_modified)
            return False
        new_state = {**state, "vals_etag": etag, "vals_last_modified": last_modified}
        _save_state(new_state)
    except Exception as exc:
        log.warning("HTTP freshness check failed: %s. Proceeding with full scrape.", exc)
    return True


# ---------------------------------------------------------------------------
# Atomic Writes & Validation
# ---------------------------------------------------------------------------


def validate_output(data: dict) -> bool:
    if not isinstance(data, dict):
        log.error("Validation FAILED: output is not a dict")
        return False

    leaderboard = data.get("leaderboard", [])
    if not isinstance(leaderboard, list):
        log.error("Validation FAILED: leaderboard is not a list")
        return False

    if len(leaderboard) == 0:
        log.error("Validation FAILED: leaderboard has 0 records")
        return False

    required_fields = {"model", "composite_efficiency_score", "rank"}
    for i, entry in enumerate(leaderboard):
        if not isinstance(entry, dict):
            log.error("Validation FAILED: entry %d is not a dict", i)
            return False
        missing = required_fields - set(entry.keys())
        if missing:
            log.error("Validation FAILED: entry %d missing fields: %s", i, missing)
            return False
        if not isinstance(entry.get("composite_efficiency_score"), (int, float)):
            log.error(
                "Validation FAILED: entry %d has non-numeric composite_efficiency_score",
                i,
            )
            return False

    log.info(
        "Validation PASSED: %d records, all fields present and valid", len(leaderboard)
    )
    return True


def atomic_write_output(data: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    TEMP_FILE.write_text(json.dumps(data, indent=2, sort_keys=False))

    if not validate_output(data):
        log.error("Validation of temp_data.json FAILED. Aborting write. Old data preserved.")
        sys.exit(1)

    log.info("Writing leaderboard.json and leaderboard.csv")
    LEADERBOARD_JSON.write_text(json.dumps(data, indent=2, sort_keys=False))

    rows = []
    for entry in data["leaderboard"]:
        flat = {
            "rank": entry["rank"],
            "model": entry["model"],
            "composite_efficiency_score": entry["composite_efficiency_score"],
            "avg_normalized_e": entry["avg_normalized_e"],
            "coverage_weight": entry["coverage_weight"],
            "participation_penalty": entry["participation_penalty"],
            "n_benchmarks": entry["n_benchmarks"],
            "benchmarks_participated": "|".join(entry["benchmarks_participated"]),
        }
        rows.append(flat)

    pd.DataFrame(rows).to_csv(LEADERBOARD_CSV, index=False)
    log.info("Wrote %d rows to leaderboard.csv", len(rows))

    try:
        TEMP_FILE.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def run_pipeline():
    log.info("=" * 60)
    log.info("LLM Meta-Leaderboard Pipeline starting")
    log.info("=" * 60)

    session = _session()

    _should_continue = _check_http_freshness(session)
    if not _should_continue:
        log.info("HTTP headers suggest no change. Attempting lightweight check...")
        try:
            resp = session.get(VALS_BASE + "/benchmarks", timeout=REQUEST_TIMEOUT)
            html = resp.text
            soup = bs4.BeautifulSoup(html, "html.parser")
            model_count_text = ""
            for elem in soup.find_all(string=re.compile(r"\d+\s+models?\s+tested")):
                model_count_text = elem.strip()
                break
            if model_count_text:
                state = _load_state()
                if state.get("vals_model_count_text") == model_count_text:
                    log.info("Model count text unchanged. Exiting early.")
                    sys.exit(0)
                new_state = {**state, "vals_model_count_text": model_count_text}
                _save_state(new_state)
        except Exception as exc:
            log.warning("Lightweight check failed: %s. Proceeding.", exc)

    log.info("--- Phase 1: Data Ingestion ---")
    vals_rows = scrape_vals_ai(session)
    aa_rows = scrape_artificial_analysis(session)
    all_rows = vals_rows + aa_rows

    if not all_rows:
        log.error("No data scraped from any source. Failing run to preserve old data.")
        sys.exit(1)

    log.info("Total raw rows scraped: %d", len(all_rows))

    log.info("--- Phase 2: Entity Resolution ---")
    df = pd.DataFrame(all_rows)
    df = deduplicate_models(df)
    log.info("After deduplication: %d rows, %d unique models", len(df), df["model_canonical"].nunique())

    log.info("--- Phase 3: Outlier Rejection ---")
    df = reject_outliers(df)
    log.info("After outlier rejection: %d rows", len(df))

    log.info("--- Phase 4: Efficiency Calculation ---")
    df = calculate_efficiency(df)
    log.info("E_bt calculated for %d rows", len(df))

    log.info("--- Phase 5: Normalization ---")
    df = normalize_per_benchmark(df)

    log.info("--- Phase 6: Composite Aggregation ---")
    result_df = aggregate_composite(df)
    log.info("Composite scores computed for %d models", len(result_df))

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0.0",
        "n_models": len(result_df),
        "n_benchmarks": df["benchmark"].nunique(),
        "leaderboard": result_df.to_dict(orient="records"),
    }

    log.info("--- Phase 7: State Check & Atomic Write ---")
    serialized_rows = [
        {
            "model": r["model"],
            "benchmark": d["benchmark"],
            "score": d["score"],
            "e_bt": d["e_bt"],
        }
        for r in output["leaderboard"]
        for d in r.get("benchmark_details", [])
    ]

    if not check_should_run(serialized_rows):
        log.info("No data changes detected. Exiting without overwriting files.")
        sys.exit(0)

    atomic_write_output(output)

    new_state = _load_state()
    new_state["data_fingerprint"] = _compute_state_fingerprint(serialized_rows)
    new_state["last_run_at"] = output["generated_at"]
    new_state["n_models"] = len(result_df)
    _save_state(new_state)

    log.info("=" * 60)
    log.info("Pipeline completed successfully")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
