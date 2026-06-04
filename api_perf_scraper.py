#!/usr/bin/env python3
"""
API Performance Scraper for LLM Meta-Leaderboard
=================================================
Fetches API-level performance metrics (TPS, TTFT, total response time) from:
  1. Local llm-speedrun benchmark data (all_providers_benchmark_with_estimates.csv)
  2. Web scraping from third-party API benchmark sources (KickLLM, TokenMix, etc.)
  3. OpenRouter public API stats

Integration:
  - Normalizes provider-model names to match main leaderboard canonical names
  - Calculates composite speed_score = (TPS / TTFT_weight) - normalized_penalty
  - Can be merged as an additional "benchmark" column in the main pipeline

Sources:
  - llm-speedrun data/ (local, user-generated)
  - kickllm.com/research/ai-api-latency-comparison.html
  - tokenmix.ai/blog/ai-api-latency-benchmark
  - tokenmix.ai/blog/ai-api-response-time-comparison
  - crazyrouter.com/en/blog/ai-inference-speed-benchmark-2026
  - digitalapplied.com/blog/ai-model-latency-benchmarks-2026-ttft-throughput
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("api-perf")

LLM_SPEEDRUN_DATA = Path(__file__).parent.parent / "llm-speedrun" / "data" / "all_providers_benchmark_with_estimates.csv"
OUTPUT_FILE = Path(__file__).parent / "data" / "api_performance.json"

REQUEST_TIMEOUT = 15
REQUEST_RETRIES = 3
RETRY_BACKOFF = 2

MODEL_ALIAS_MAP = {
    "gpt-5": ["gpt-5", "openai/gpt-5", "gpt5"],
    "gpt-5.4": ["gpt-5.4", "gpt5.4", "gpt-5.4-2026-03-05"],
    "gpt-5.3-codex": ["gpt-5.3-codex", "gpt-5-codex", "gpt5.3-codex", "gpt5.3codex"],
    "gpt-5.2-codex": ["gpt-5.2-codex", "gpt5.2-codex"],
    "gpt-5.2": ["gpt-5.2", "gpt5.2", "gpt-5.2-2025-12-11"],
    "gpt-5.1": ["gpt-5.1", "gpt5.1"],
    "gpt-5-mini": ["gpt-5-mini", "gpt5-mini"],
    "gpt-4.1": ["gpt-4.1", "gpt4.1"],
    "gpt-4.1-mini": ["gpt-4.1-mini", "gpt4.1-mini"],
    "o4-mini": ["o4-mini", "openai/o4-mini"],
    "o3": ["o3", "openai/o3", "openai/o3-mini"],
    "claude-opus-4.7": ["claude-opus-4-7", "claude-opus-4.7", "claude-opus-4-7", "Claude Opus 4.7", "claude-opus-4-6", "claude-opus-4-5", "anthropic-claude-opus-4.6", "anthropic-claude-opus-4-6", "anthropic-claude-opus-4.5", "claude-opus-4-6", "claude-opus-4-5"],
    "claude-sonnet-4.6": ["claude-sonnet-4-6", "claude-sonnet-4.6", "claude-sonnet-4", "Claude Sonnet 4.6", "Claude Sonnet 4", "claude-3-5-sonnet", "claude-3-7-sonnet", "anthropic-claude-sonnet-4.6", "claude-sonnet-4.5"],
    "claude-haiku-3.5": ["claude-haiku-4-5-20251001-thinking", "anthropic-claude-haiku-4.5", "claude-haiku-3.5", "Claude Haiku 3.5"],
    "gemini-3.1-pro-preview-0226": ["gemini-3.1-pro-preview", "gemini-3.1-pro", "gemini-3.1-pro-preview-03-06", "google-gemini-3.1-pro", "gemini-3.1-flash-lite-preview"],
    "gemini-3-pro-preview": ["gemini-3-pro-preview", "gemini-3-pro", "google-gemini-3-pro", "gemini-3-pro-preview-06-18"],
    "gemini-3-flash-preview": ["gemini-3-flash-preview", "gemini-3-flash", "google-gemini-3-flash", "gemini-3-flash-preview-06-18"],
    "gemini-2.5-flash": ["gemini-2.5-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-flash-preview-09-2025", "gemini-flash-2.5"],
    "gemini-2.5-pro": ["gemini-2.5-pro", "gemini-2.5-pro-preview-03-25", "google-gemini-2.5-pro"],
    "gemini-2.5-flash-lite": ["gemini-2.5-flash-lite-preview-09-2025"],
    "grok-4": ["grok-4-0709", "grok-4.3", "grok-4-fast-non-reasoning", "grok-4-fast-reasoning"],
    "grok-3": ["grok-3", "grok3"],
    "grok-4.1": ["grok-4-1-fast-non-reasoning", "grok-4-1-fast-reasoning", "grok-4.1", "grok/grok-4.1", "grok-4.1-expert", "grok-4.20-beta", "grok-4.1-mini"],
    "deepseek-v3": ["deepseek-v3-0324", "deepseek-v3", "deepseekv3", "deepseek-chat"],
    "deepseek-r1": ["deepseek-r1", "deepseekr1"],
    "deepseek-v3.2": ["deepseek-v3.2", "deepseek-v3p1", "deepseek-v3p2"],
    "llama-4-405b": ["llama-4-405b", "llama4-405b"],
    "llama-3.3-70b": ["llama-3.3-70b", "llama3.3-70b", "llama-3.3-70b-instruct"],
    "llama-3.1-70b": ["llama-3.1-70b", "llama3.1-70b"],
    "llama-3.1-8b": ["llama-3.1-8b", "llama3.1-8b"],
    "qwen3-235b": ["qwen3-235b", "qwen3-235b-a22b"],
    "qwen2.5-72b": ["qwen2.5-72b", "qwen2.5-72b-instruct"],
    "qwen3-coder": ["qwen3-coder", "qwen3-coder-flash", "qwen3-coder-next"],
    "mistral-small-3.1": ["mistral-small-3.1", "mistral-small-3.1-24b"],
    "codestral": ["codestral", "codestral-25.01"],
    "minimax-m2.7": ["minimax-m2.7", "minimax-m2.7-thinking", "minimax-m3"],
    "minimax-m2.1": ["minimax-m2.1"],
    "gpt-oss-120b": ["gpt-oss-120b", "nvidia/gpt-oss-120b"],
    "gpt-oss-20b": ["gpt-oss-20b"],
    "gpt-5.5": ["gpt-5.5"],
}


@dataclass
class ApiPerfRecord:
    model_raw: str
    model_canonical: str
    provider: str
    base_url: str
    ttft_s: float
    tps: float
    output_tokens: int
    total_time_s: float
    error: str
    timestamp: str
    source: str
    confidence: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("model_raw", None)
        return d


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "MetaLeaderboard-API-Perf/1.0 (+https://github.com/user/llm-meta-leaderboard)",
        "Accept": "application/json, text/html",
    })
    return s


def _fetch_with_retry(url: str, session: requests.Session) -> requests.Response:
    for attempt in range(REQUEST_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt < REQUEST_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                log.warning("Retry %d/%d for %s: %s (waiting %ds)", attempt+1, REQUEST_RETRIES, url, e, wait)
                time.sleep(wait)
            else:
                raise
    return None


def load_speedrun_data() -> list[ApiPerfRecord]:
    log.info("Loading llm-speedrun benchmark data...")
    records = []
    if not LLM_SPEEDRUN_DATA.exists():
        log.warning("llm-speedrun data not found at %s", LLM_SPEEDRUN_DATA)
        return records

    try:
        df = pd.read_csv(LLM_SPEEDRUN_DATA)
        for _, row in df.iterrows():
            ttft = row.get("ttft_s") or 0
            tps = row.get("tps") or 0
            if pd.isna(ttft) or pd.isna(tps) or ttft <= 0 or tps <= 0:
                continue
            if tps > 10000 or ttft > 30:
                continue

            model_raw = str(row.get("model", "")).strip()
            provider = str(row.get("provider", "unknown")).strip()
            base_url = str(row.get("base_url", "")).strip()
            tokens = int(row.get("tokens", 0)) if not pd.isna(row.get("tokens")) else 0
            total_time = row.get("estimated_total_time_s", 0) or 0
            error = str(row.get("error", "")) or ""
            timestamp = str(row.get("timestamp", "")) or ""

            model_canonical = _normalize_model_name(model_raw)

            records.append(ApiPerfRecord(
                model_raw=model_raw,
                model_canonical=model_canonical,
                provider=provider,
                base_url=base_url,
                ttft_s=float(ttft),
                tps=float(tps),
                output_tokens=tokens,
                total_time_s=float(total_time) if total_time > 0 else (tokens / tps if tps > 0 else 0),
                error=error,
                timestamp=timestamp,
                source="llm-speedrun",
                confidence=0.8 if tokens >= 100 else 0.4,
            ))
    except Exception as e:
        log.error("Error loading speedrun data: %s", e)

    log.info("  -> %d valid API perf records from llm-speedrun", len(records))
    return records


def _normalize_model_name(model_raw: str) -> str:
    model_lower = model_raw.lower().strip().replace("/", "-").replace(" ", "-").replace("_", "-")
    for canonical, aliases in MODEL_ALIAS_MAP.items():
        for alias in aliases:
            al = alias.lower().replace("/", "-").replace(" ", "-").replace("_", "-")
            if al == model_lower or al in model_lower or model_lower in al:
                return canonical
    if "claude" in model_lower and "opus" in model_lower:
        if "sonnet" in model_lower or "-4-" in model_lower and "thinking" not in model_lower:
            return "claude-sonnet-4.6" if "4.6" in model_lower else "claude-sonnet-4"
        if "haiku" in model_lower:
            return "claude-haiku-3.5" if "3.5" in model_lower else "claude-haiku-3"
        return "claude-opus-4.7" if "4.7" in model_lower else "claude-opus-4.6" if "4.6" in model_lower else "claude-opus-4.5"
    if "claude" in model_lower and "sonnet" in model_lower:
        return "claude-sonnet-4.6" if "4.6" in model_lower else "claude-sonnet-4"
    if "claude" in model_lower and "haiku" in model_lower:
        return "claude-haiku-3.5" if "3.5" in model_lower else "claude-haiku-3"
    if "gpt-5" in model_lower:
        if "mini" in model_lower: return "gpt-5-mini"
        if "codex" in model_lower: return "gpt-5.3-codex" if "5.3" in model_lower else "gpt-5.2-codex"
        if "5.4" in model_lower: return "gpt-5.4"
        if "5.2" in model_lower: return "gpt-5.2"
        if "5.1" in model_lower: return "gpt-5.1"
        if "5." in model_lower: return f"gpt-{model_lower[:5]}"
        return "gpt-5"
    if "gemini" in model_lower:
        if "flash-lite" in model_lower: return "gemini-2.5-flash-lite"
        if "flash" in model_lower:
            if "3.1" in model_lower: return "gemini-3.1-pro-preview-0226"
            if "3" in model_lower: return "gemini-3-flash-preview"
            return "gemini-2.5-flash"
        if "3.1" in model_lower: return "gemini-3.1-pro-preview-0226"
        if "3-pro" in model_lower: return "gemini-3-pro-preview"
        if "2.5-pro" in model_lower: return "gemini-2.5-pro"
        return model_lower
    if "grok" in model_lower:
        if "4.3" in model_lower: return "grok-4.3"
        if "4.1" in model_lower: return "grok-4.1"
        if "4-" in model_lower or "4." in model_lower: return "grok-4"
        if "3" in model_lower: return "grok-3"
    if "deepseek" in model_lower:
        if "r1" in model_lower: return "deepseek-r1"
        if "v3.2" in model_lower or "v3_2" in model_lower or "v3p" in model_lower: return "deepseek-v3.2"
        if "v3" in model_lower or "v3-" in model_lower: return "deepseek-v3"
    if "llama" in model_lower:
        if "3.3" in model_lower: return "llama-3.3-70b"
        if "3.1" in model_lower:
            if "8b" in model_lower: return "llama-3.1-8b"
            return "llama-3.1-70b"
        if "4" in model_lower and "405" in model_lower: return "llama-4-405b"
    if "qwen" in model_lower:
        if "3-235" in model_lower or "235b" in model_lower: return "qwen3-235b"
        if "coder" in model_lower: return "qwen3-coder"
        if "2.5" in model_lower and "72b" in model_lower: return "qwen2.5-72b"
    if "mistral" in model_lower:
        if "small" in model_lower and "3" in model_lower: return "mistral-small-3.1"
        if "codestral" in model_lower: return "codestral"
    if "minimax" in model_lower:
        if "m2.7" in model_lower or "m2_7" in model_lower or "m3" in model_lower: return "minimax-m2.7"
        if "m2.1" in model_lower or "m2_1" in model_lower: return "minimax-m2.1"
    if "gpt-oss" in model_lower or ("nvidia" in model_lower and "gpt" in model_lower): return "gpt-oss-120b"
    if model_lower.startswith("o") and any(c.isdigit() for c in model_lower):
        if "4-mini" in model_lower or "4mini" in model_lower: return "o4-mini"
        if "3-mini" in model_lower or "3mini" in model_lower: return "o3"
        if "3" in model_lower: return "o3"
        if "4" in model_lower: return "o4-mini"
    return model_lower


def scrape_third_party(session: requests.Session) -> list[ApiPerfRecord]:
    log.info("Scraping third-party API benchmark sources...")
    all_records = []
    sources = [
        {
            "url": "https://kickllm.com/research/ai-api-latency-comparison.html",
            "parser": _parse_kickllm,
        },
        {
            "url": "https://tokenmix.ai/blog/ai-api-latency-benchmark",
            "parser": _parse_tokenmix_benchmark,
        },
        {
            "url": "https://crazyrouter.com/en/blog/ai-inference-speed-benchmark-2026",
            "parser": _parse_crazyrouter,
        },
        {
            "url": "https://www.digitalapplied.com/blog/ai-model-latency-benchmarks-2026-ttft-throughput",
            "parser": _parse_digital_applied,
        },
    ]

    for src in sources:
        try:
            resp = _fetch_with_retry(src["url"], session)
            records = src["parser"](resp.text, src["url"])
            all_records.extend(records)
            log.info("  %s: %d records", src["url"], len(records))
        except Exception as e:
            log.warning("Failed to scrape %s: %s", src["url"], e)

    return all_records


def _parse_kickllm(html: str, url: str) -> list[ApiPerfRecord]:
    records = []
    import bs4
    soup = bs4.BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return records

    for row in table.find_all("tr")[1:]:
        cols = row.find_all(["td", "th"])
        if len(cols) < 5:
            continue
        try:
            model_text = cols[0].get_text(strip=True)
            provider_text = cols[1].get_text(strip=True) if len(cols) > 1 else "unknown"
            ttft_text = cols[2].get_text(strip=True).replace("ms", "").replace(",", "") if len(cols) > 2 else "0"
            tps_text = cols[3].get_text(strip=True) if len(cols) > 3 else "0"

            ttft_s = float(ttft_text) / 1000.0 if ttft_text else 0
            tps = float(tps_text) if tps_text else 0

            if ttft_s <= 0 or tps <= 0:
                continue

            model_canonical = _normalize_model_name(model_text)
            records.append(ApiPerfRecord(
                model_raw=model_text,
                model_canonical=model_canonical,
                provider=provider_text.lower(),
                base_url="",
                ttft_s=ttft_s,
                tps=tps,
                output_tokens=256,
                total_time_s=(256 / tps) + ttft_s,
                error="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="kickllm",
                confidence=0.7,
            ))
        except (ValueError, IndexError):
            continue

    return records


def _parse_tokenmix_benchmark(html: str, url: str) -> list[ApiPerfRecord]:
    records = []
    import bs4
    soup = bs4.BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    for table in tables[:2]:
        for row in table.find_all("tr")[1:]:
            cols = row.find_all(["td", "th"])
            if len(cols) < 4:
                continue
            try:
                provider_model = cols[0].get_text(strip=True)
                ttft_text = cols[1].get_text(strip=True).replace("ms", "").replace("s", "").strip()
                tps_text = cols[2].get_text(strip=True) if len(cols) > 2 else "0"

                if "·" in provider_model:
                    provider, model = provider_model.split("·", 1)
                    provider = provider.strip().lower()
                    model = model.strip()
                else:
                    provider = "unknown"
                    model = provider_model

                ttft_s = float(ttft_text) if "ms" not in ttft_text else float(ttft_text) / 1000
                tps = float(tps_text) if tps_text else 0

                if ttft_s <= 0 or tps <= 0:
                    continue

                model_canonical = _normalize_model_name(model)
                records.append(ApiPerfRecord(
                    model_raw=model,
                    model_canonical=model_canonical,
                    provider=provider,
                    base_url="",
                    ttft_s=ttft_s,
                    tps=tps,
                    output_tokens=256,
                    total_time_s=(256 / tps) + ttft_s,
                    error="",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="tokenmix",
                    confidence=0.75,
                ))
            except (ValueError, IndexError):
                continue

    return records


def _parse_crazyrouter(html: str, url: str) -> list[ApiPerfRecord]:
    records = []
    import bs4
    soup = bs4.BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    for table in tables:
        for row in table.find_all("tr")[1:]:
            cols = row.find_all(["td", "th"])
            if len(cols) < 4:
                continue
            try:
                model_text = cols[0].get_text(strip=True)
                provider_text = cols[1].get_text(strip=True).lower() if len(cols) > 1 else "unknown"
                ttft_text = cols[2].get_text(strip=True).replace("ms", "").replace("s", "").strip() if len(cols) > 2 else "0"
                tps_text = cols[3].get_text(strip=True) if len(cols) > 3 else "0"

                ttft_s = float(ttft_text) if "ms" not in ttft_text and ttft_text else float(ttft_text) / 1000
                tps = float(tps_text) if tps_text else 0

                if ttft_s <= 0 or tps <= 0:
                    continue

                model_canonical = _normalize_model_name(model_text)
                records.append(ApiPerfRecord(
                    model_raw=model_text,
                    model_canonical=model_canonical,
                    provider=provider_text,
                    base_url="",
                    ttft_s=ttft_s,
                    tps=tps,
                    output_tokens=200,
                    total_time_s=(200 / tps) + ttft_s,
                    error="",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="crazyrouter",
                    confidence=0.65,
                ))
            except (ValueError, IndexError):
                continue

    return records


def _parse_digital_applied(html: str, url: str) -> list[ApiPerfRecord]:
    records = []
    import bs4
    soup = bs4.BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    for table in tables[:2]:
        for row in table.find_all("tr")[1:]:
            cols = row.find_all(["td", "th"])
            if len(cols) < 3:
                continue
            try:
                content = cols[0].get_text(strip=True)
                provider_text = cols[1].get_text(strip=True).lower() if len(cols) > 1 else "unknown"
                tps_text = cols[-1].get_text(strip=True) if len(cols) > 2 else "0"

                if "·" in content:
                    provider_extra, model_text = content.split("·", 1)
                    provider = provider_text if provider_text != "unknown" else provider_extra.strip().lower()
                else:
                    provider = provider_text
                    model_text = content

                tps = float(tps_text) if tps_text else 0
                ttft_s = 0.5

                if tps <= 0:
                    continue

                model_canonical = _normalize_model_name(model_text)
                records.append(ApiPerfRecord(
                    model_raw=model_text,
                    model_canonical=model_canonical,
                    provider=provider,
                    base_url="",
                    ttft_s=ttft_s,
                    tps=tps,
                    output_tokens=256,
                    total_time_s=(256 / tps) + ttft_s,
                    error="",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="digitalapplied",
                    confidence=0.6,
                ))
            except (ValueError, IndexError):
                continue

    return records


def aggregate_api_perf(records: list[ApiPerfRecord]) -> list[dict]:
    log.info("Aggregating %d API perf records by model...", len(records))
    if not records:
        return []

    df = pd.DataFrame([r.to_dict() for r in records])

    agg_records = []
    for model, group in df.groupby("model_canonical"):
        weights = group["confidence"].values
        total_weight = weights.sum()

        if total_weight <= 0:
            continue

        avg_ttft = (group["ttft_s"] * weights).sum() / total_weight
        avg_tps = (group["tps"] * weights).sum() / total_weight
        avg_total_time = (group["total_time_s"] * weights).sum() / total_weight
        n_sources = len(group)

        providers = group["provider"].value_counts().to_dict()
        top_provider = max(providers.keys(), key=lambda p: providers[p]) if providers else "unknown"

        agg_records.append({
            "model_canonical": model,
            "ttft_s": round(avg_ttft, 4),
            "tps": round(avg_tps, 2),
            "total_time_s": round(avg_total_time, 2),
            "n_sources": n_sources,
            "providers": list(providers.keys()),
            "top_provider": top_provider,
            "speed_score": round(calculate_speed_score(avg_tps, avg_ttft), 4),
        })

    agg_records.sort(key=lambda x: x["speed_score"], reverse=True)
    for i, rec in enumerate(agg_records, 1):
        rec["speed_rank"] = i

    log.info("  -> %d unique models with API perf data", len(agg_records))
    return agg_records


def calculate_speed_score(tps: float, ttft_s: float) -> float:
    if ttft_s <= 0 or tps <= 0:
        return 0
    ttft_weight = 1.0 + (ttft_s * 0.5)
    return (tps / ttft_weight) * 10


def merge_with_leaderboard(api_perf: list[dict], leaderboard: list[dict]) -> list[dict]:
    api_map = {r["model_canonical"]: r for r in api_perf}

    for entry in leaderboard:
        model = entry.get("model", "")
        if model in api_map:
            perf = api_map[model]
            entry["api_performance"] = {
                "ttft_s": perf["ttft_s"],
                "tps": perf["tps"],
                "total_time_s": perf["total_time_s"],
                "speed_score": perf["speed_score"],
                "speed_rank": perf["speed_rank"],
                "provider": perf["top_provider"],
            }
        else:
            entry["api_performance"] = None

    return leaderboard


def save_api_perf(api_perf: list[dict], path: Path = None):
    path = path or OUTPUT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "sources": ["llm-speedrun", "kickllm", "tokenmix", "crazyrouter", "digitalapplied"],
        "records": api_perf,
    }, indent=2))
    log.info("Wrote %d API perf records to %s", len(api_perf), path)


def run_api_perf_pipeline(leaderboard: list[dict] = None) -> list[dict]:
    log.info("=" * 60)
    log.info("API Performance Scraper")
    log.info("=" * 60)

    session = _session()

    log.info("--- Phase 1: Load local benchmark data ---")
    local_records = load_speedrun_data()

    log.info("--- Phase 2: Scrape third-party sources ---")
    web_records = scrape_third_party(session)

    all_records = local_records + web_records
    log.info("Total records: %d (local=%d, web=%d)", len(all_records), len(local_records), len(web_records))

    if not all_records:
        log.warning("No API perf data collected")
        return []

    log.info("--- Phase 3: Aggregate by model ---")
    api_perf = aggregate_api_perf(all_records)
    save_api_perf(api_perf)

    if leaderboard:
        log.info("--- Phase 4: Merge with main leaderboard ---")
        merged = merge_with_leaderboard(api_perf, leaderboard)
        log.info("  -> %d models matched with API perf", sum(1 for e in merged if e.get("api_performance")))
        return merged

    return api_perf


if __name__ == "__main__":
    api_perf = run_api_perf_pipeline()
    print(f"\nTop 10 by Speed Score:")
    for rec in api_perf[:10]:
        print(f"  {rec['speed_rank']}. {rec['model_canonical']}: {rec['speed_score']:.2f} (TTFT={rec['ttft_s']:.3f}s, TPS={rec['tps']:.1f})")