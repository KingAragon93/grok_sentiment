#!/usr/bin/env python3
"""
Cost guardrails for the groksentiment service (added 2026-07-07).

Every x.ai call here runs X Live Search, which bills PER SOURCE READ on top
of tokens — and the current xai-sdk x_search tool has no max-results knob.
The levers that exist are the search date window, the handle allowlist, and
simply making fewer calls. All three are driven by ONE Mission-Control-
editable config file in GCS:

    gs://historical_stock_day/grok_sentiment_config.json

Keys (all optional; missing keys fall back to DEFAULTS):
    search_window_hours          cap on the X-search lookback for sentiment/
                                 recommendation calls (requests asking for
                                 more are clamped)
    market_factors_window_hours  separate wider window for the market-factors
                                 mode (it hunts multi-day catalysts)
    max_calls_per_hour           hard budget; requests beyond it get HTTP 429
    allowed_handles_enabled      master switch for the allowlist
    allowed_handles              list of X handles (with or without @) that
                                 live search is restricted to; empty list =
                                 no restriction

Config is cached in-memory for CONFIG_TTL_SECONDS, so MC edits apply within
a minute without a redeploy. The hourly budget counter uses GCS generation
preconditions so concurrent instances can't double-spend. Everything fails
OPEN: a GCS hiccup must never block trading-path callers.
"""
import datetime
import json
import logging
import os
import time
from typing import List, Optional, Tuple

from google.cloud import storage

logger = logging.getLogger(__name__)

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "historical_stock_day")
CONFIG_FILE = "grok_sentiment_config.json"
BUDGET_FILE = "grok_call_budget.json"
CONFIG_TTL_SECONDS = 60

DEFAULTS = {
    "search_window_hours": 24,
    "market_factors_window_hours": 72,
    "max_calls_per_hour": 15,
    "allowed_handles_enabled": True,
    "allowed_handles": [],
}

_config_cache = {"config": None, "loaded_at": 0.0}
_storage_client = None


def _client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def load_config(force: bool = False) -> dict:
    """Load the guardrails config from GCS with a short in-memory cache."""
    now = time.time()
    if not force and _config_cache["config"] is not None and now - _config_cache["loaded_at"] < CONFIG_TTL_SECONDS:
        return _config_cache["config"]
    cfg = dict(DEFAULTS)
    try:
        blob = _client().bucket(GCS_BUCKET_NAME).blob(CONFIG_FILE)
        if blob.exists():
            data = json.loads(blob.download_as_text())
            for key in DEFAULTS:
                if key in data:
                    cfg[key] = data[key]
        else:
            logger.warning(f"guardrails: gs://{GCS_BUCKET_NAME}/{CONFIG_FILE} not found — using defaults")
    except Exception as e:
        logger.warning(f"guardrails: config load failed ({e}) — using defaults")
    _config_cache["config"] = cfg
    _config_cache["loaded_at"] = now
    return cfg


def get_search_window_hours(requested_hours: Optional[float] = None, market_factors: bool = False) -> int:
    """Clamp a requested lookback to the configured window cap."""
    cfg = load_config()
    cap = cfg["market_factors_window_hours"] if market_factors else cfg["search_window_hours"]
    try:
        cap = int(cap)
    except (TypeError, ValueError):
        cap = DEFAULTS["market_factors_window_hours" if market_factors else "search_window_hours"]
    if requested_hours is None:
        return cap
    try:
        return max(1, min(int(requested_hours), cap))
    except (TypeError, ValueError):
        return cap


def get_allowed_handles() -> Optional[List[str]]:
    """Return the curated handle allowlist, or None for no restriction."""
    cfg = load_config()
    if not cfg.get("allowed_handles_enabled", True):
        return None
    handles = [str(h).strip().lstrip("@") for h in (cfg.get("allowed_handles") or [])]
    handles = [h for h in handles if h]
    return handles or None


def check_call_budget() -> Tuple[bool, int, int]:
    """Atomically count this call against the hourly budget.

    Returns (allowed, count_this_hour, limit). Fails OPEN on GCS errors so an
    infra hiccup can never block the trading path.
    """
    cfg = load_config()
    try:
        limit = int(cfg.get("max_calls_per_hour", DEFAULTS["max_calls_per_hour"]))
    except (TypeError, ValueError):
        limit = DEFAULTS["max_calls_per_hour"]
    if limit <= 0:
        return True, 0, limit  # 0/negative = budget disabled

    hour_key = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H")
    try:
        bucket = _client().bucket(GCS_BUCKET_NAME)
        for _ in range(4):
            blob = bucket.get_blob(BUDGET_FILE)
            if blob is None:
                doc, generation = {"hour": hour_key, "count": 0}, 0
            else:
                generation = blob.generation
                try:
                    doc = json.loads(blob.download_as_text(if_generation_match=generation))
                except Exception:
                    doc = {"hour": hour_key, "count": 0}
            if doc.get("hour") != hour_key:
                doc = {"hour": hour_key, "count": 0}
            if int(doc.get("count", 0)) >= limit:
                return False, int(doc["count"]), limit
            doc["count"] = int(doc.get("count", 0)) + 1
            try:
                bucket.blob(BUDGET_FILE).upload_from_string(
                    json.dumps(doc), content_type="application/json",
                    if_generation_match=generation,
                )
                return True, doc["count"], limit
            except Exception:
                time.sleep(0.2)  # concurrent writer — re-read and retry
        logger.warning("guardrails: budget counter contention — allowing call (fail-open)")
        return True, -1, limit
    except Exception as e:
        logger.warning(f"guardrails: budget check failed ({e}) — allowing call (fail-open)")
        return True, -1, limit
