"""
Grok Sentiment Cloud Function

A Cloud Run function that provides real-time sentiment analysis for stock tickers
using xAI's Agent Tools API with x_search to analyze X/Twitter posts.

Usage:
    POST request with JSON body:
    {
        "symbol": "SOFI",
        "hours_back": 24,        # Optional, default 24
        "max_turns": 2,          # Optional, default 2 (cost control)
        "send_to_discord": false # Optional, default false
    }

Returns:
    {
        "status": "success",
        "symbol": "SOFI",
        "sentiment_score": 8.5,
        "summary": "Bullish sentiment with...",
        "citations_count": 45,
        "tool_usage": {"SERVER_SIDE_TOOL_X_SEARCH": 2},
        "api_call_duration": 12.5,
        "model_used": "grok-4-1-fast"
    }
"""

import os
import json
import logging
import time
import datetime
from typing import Any, Dict, List, Optional
import pytz
import functions_framework
from flask import jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import xai_sdk
try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user
    from xai_sdk.tools import x_search
    XAI_SDK_AVAILABLE = True
    logger.info("✅ xai_sdk imported successfully")
except ImportError as e:
    XAI_SDK_AVAILABLE = False
    logger.error(f"❌ xai_sdk not available: {e}")

# Try to import requests for Discord
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Environment variables
XAI_API_KEY = os.environ.get('XAI_API_KEY')
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')

# Validate API key on startup
if XAI_API_KEY:
    logger.info("✅ XAI_API_KEY found in environment")
else:
    logger.warning("⚠️ XAI_API_KEY not found in environment variables")


CREDIBLE_HANDLES_FILE = os.path.join(os.path.dirname(__file__), 'credible_handles.json')


def load_credible_handles() -> Dict[str, List[str]]:
    """Load curated X/Twitter allow/exclude lists from credible_handles.json."""
    try:
        with open(CREDIBLE_HANDLES_FILE, 'r') as f:
            data = json.load(f)
        allowed = [h.strip().lstrip('@') for h in data.get('allowed', []) if h and isinstance(h, str)]
        excluded = [h.strip().lstrip('@') for h in data.get('excluded', []) if h and isinstance(h, str)]
        return {"allowed": allowed, "excluded": excluded}
    except Exception as e:
        logger.warning(f"Could not load credible_handles.json: {e}")
        return {"allowed": [], "excluded": []}


CREDIBILITY_RULES = (
    "Apply CREDIBILITY WEIGHTING to each post when computing sentiment:\n"
    "- 3x weight: verified financial journalists, licensed analysts, company IR/official accounts, SEC/regulator feeds\n"
    "- 2x weight: accounts with >50k followers AND >1 year tenure AND a finance-focused bio/history\n"
    "- 1x weight: ordinary retail accounts with a sustained posting history\n"
    "- 0.25x weight: accounts <90 days old, <500 followers, or dominated by rocket/moon emoji spam\n"
    "- 0x weight: obvious bots, paid promotion, or coordinated/copy-paste spam\n\n"
    "Intake rules:\n"
    "- Dedupe retweets and near-identical copies — count an echoing message ONCE.\n"
    "- Weight the most recent third of the time window slightly higher than the oldest third.\n"
    "- Assess SOCIAL SENTIMENT ONLY. Do NOT factor in current stock price, charts, or market data; downstream code handles price.\n"
)

SENTIMENT_JSON_SCHEMA = (
    "- sentiment_score: number from -10 to +10 (overall, credibility-weighted)\n"
    "- credible_score: number from -10 to +10 using ONLY tier-1 (3x) and tier-2 (2x) authors\n"
    "- retail_score: number from -10 to +10 using ONLY tier-3 (1x) and tier-4 (0.25x) authors\n"
    "- bullish_pct: integer 0-100, share of weighted posts that are bullish\n"
    "- bearish_pct: integer 0-100, share of weighted posts that are bearish\n"
    "- neutral_pct: integer 0-100, share of weighted posts that are neutral\n"
    "- sample_size: integer, distinct posts considered after dedupe\n"
    "- unique_authors: integer, distinct authors considered\n"
    "- echo_ratio: number 0.0-1.0, share of posts that were retweets/near-duplicates\n"
    "- signal_confidence: one of 'low', 'medium', 'high' based on sample size, credible coverage, and echo\n"
    "- top_themes: array of up to 3 short theme strings (e.g. 'earnings beat', 'breakout')\n"
    "- top_sources: array of up to 5 X handles (no '@') that drove the score\n"
    "- contrarian_flags: array of short strings for warning signs (e.g. 'high retail euphoria with low credible coverage', 'likely coordinated pump')\n"
    "- summary: ONE sentence, max 100 words, covering sentiment direction, main catalyst, and any divergence between credible_score and retail_score\n"
)


def _parse_sentiment_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the expanded sentiment JSON schema from Grok's response."""

    def _num(key: str, default: float = 0.0) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _int(key: str, default: int = 0) -> int:
        try:
            return int(float(data.get(key, default)))
        except (TypeError, ValueError):
            return int(default)

    def _str_list(key: str, limit: int = None) -> List[str]:
        val = data.get(key, [])
        if not isinstance(val, list):
            return []
        cleaned = [str(x).strip().lstrip('@') for x in val if x]
        return cleaned[:limit] if limit else cleaned

    confidence = str(data.get("signal_confidence", "medium")).lower().strip()
    if confidence not in ("low", "medium", "high"):
        confidence = "medium"

    return {
        "sentiment_score": _num("sentiment_score"),
        "credible_score": _num("credible_score"),
        "retail_score": _num("retail_score"),
        "bullish_pct": max(0, min(100, _int("bullish_pct"))),
        "bearish_pct": max(0, min(100, _int("bearish_pct"))),
        "neutral_pct": max(0, min(100, _int("neutral_pct"))),
        "sample_size": max(0, _int("sample_size")),
        "unique_authors": max(0, _int("unique_authors")),
        "echo_ratio": max(0.0, min(1.0, _num("echo_ratio"))),
        "signal_confidence": confidence,
        "top_themes": _str_list("top_themes", 3),
        "top_sources": _str_list("top_sources", 5),
        "contrarian_flags": _str_list("contrarian_flags"),
        "summary": str(data.get("summary", "Summary not provided.")),
    }


def _downgrade_confidence(level: str) -> str:
    """Knock signal_confidence down by one notch (used when falling back to open search)."""
    mapping = {"high": "medium", "medium": "low", "low": "low"}
    return mapping.get((level or "").lower(), "low")


def get_et_timezone():
    """Get Eastern timezone for consistent logging."""
    return pytz.timezone('America/New_York')


def now_et():
    """Get current time in Eastern timezone."""
    return datetime.datetime.now(get_et_timezone())


def _extract_json_object(response_content: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from plain text or fenced markdown output."""
    if not response_content:
        raise ValueError("Empty response content")

    json_str = response_content
    if "```json" in json_str:
        json_str = json_str.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```", 1)[1].split("```", 1)[0].strip()

    parsed = json.loads(json_str)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed response is not a JSON object")
    return parsed


def get_grok_market_factors(ticker: str, max_turns: int = 3, catalyst_window_days: int = 60) -> Dict[str, Any]:
    """
    Analyze catalysts and sentiment to produce options-selection market factors.

    Returns keys:
      - recommended_min_days
      - volatility_risk
      - bias
    """
    symbol = (ticker or "").upper().strip()
    if not symbol:
        return {
            "status": "error",
            "reason": "Missing ticker symbol",
            "symbol": symbol
        }

    if not XAI_SDK_AVAILABLE:
        return {
            "status": "error",
            "reason": "xai_sdk not available. Install with: pip install xai-sdk>=1.5.0",
            "symbol": symbol
        }

    if not XAI_API_KEY:
        return {
            "status": "error",
            "reason": "XAI_API_KEY not configured in environment variables",
            "symbol": symbol
        }

    try:
        xai_client = XAIClient(api_key=XAI_API_KEY)
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to initialize xAI client: {e}",
            "symbol": symbol
        }

    now = datetime.datetime.now(pytz.UTC)
    from_date = now - datetime.timedelta(days=7)
    to_date = now
    model_used = "grok-4-1-fast"
    start_time = time.time()

    user_prompt = (
        f"Analyze ${symbol} for an options-entry decision. "
        f"Use recent X/Twitter and reputable news context, and identify upcoming catalysts in the next {catalyst_window_days} days. "
        f"Catalysts to check include: earnings, FOMC/CPI/macro events, product launches, major legal/regulatory decisions. "
        f"Return ONLY JSON with exactly these keys:\n"
        f"1. 'key_dates': array of objects with 'event', 'date' (YYYY-MM-DD), and 'days_until' (int 0-{catalyst_window_days})\n"
        f"2. 'sentiment_bias': one of 'Bullish', 'Bearish', 'Neutral'\n"
        f"3. 'bias': one of 'Call' or 'Put'\n"
        f"4. 'volatility_risk': integer from 1 to 10 (higher means more crowded/expensive options)\n"
        f"5. 'rationale': brief explanation (max 120 words)"
    )

    try:
        chat = xai_client.chat.create(
            model=model_used,
            tools=[x_search(from_date=from_date, to_date=to_date)],
            max_turns=max_turns,
        )
        chat.append(user(user_prompt))
        response = chat.sample()
        api_call_duration = time.time() - start_time

        response_content = response.content
        parsed = _extract_json_object(response_content)

        key_dates_raw = parsed.get("key_dates", [])
        key_dates: List[Dict[str, Any]] = []
        days_until_values: List[int] = []

        if isinstance(key_dates_raw, list):
            for item in key_dates_raw:
                if not isinstance(item, dict):
                    continue

                event_name = str(item.get("event", "")).strip()[:120]
                date_str = str(item.get("date", "")).strip()

                days_until = item.get("days_until")
                try:
                    days_until_int = int(days_until)
                except (TypeError, ValueError):
                    days_until_int = None

                if days_until_int is None and date_str:
                    try:
                        event_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                        days_until_int = (event_date - now.date()).days
                    except ValueError:
                        days_until_int = None

                if days_until_int is None:
                    continue

                if days_until_int < 0 or days_until_int > catalyst_window_days:
                    continue

                days_until_values.append(days_until_int)
                key_dates.append({
                    "event": event_name or "Unknown catalyst",
                    "date": date_str,
                    "days_until": days_until_int
                })

        if days_until_values:
            last_catalyst_days = max(days_until_values)
            recommended_min_days = last_catalyst_days + 7
        else:
            last_catalyst_days = None
            recommended_min_days = 14

        raw_bias = str(parsed.get("bias") or parsed.get("sentiment_bias") or "").strip().lower()
        if raw_bias in ("bearish", "put", "sell", "down"):
            normalized_bias = "Put"
        elif raw_bias in ("bullish", "call", "buy", "up"):
            normalized_bias = "Call"
        else:
            normalized_bias = "Call"

        try:
            volatility_risk = int(parsed.get("volatility_risk", 5))
        except (TypeError, ValueError):
            volatility_risk = 5
        volatility_risk = max(1, min(10, volatility_risk))

        tool_usage = {}
        if hasattr(response, 'server_side_tool_usage'):
            tool_usage = dict(response.server_side_tool_usage) if response.server_side_tool_usage else {}

        citations_count = len(response.citations) if hasattr(response, 'citations') and response.citations else 0

        return {
            "status": "success",
            "symbol": symbol,
            "recommended_min_days": int(recommended_min_days),
            "volatility_risk": volatility_risk,
            "bias": normalized_bias,
            "sentiment_bias": str(parsed.get("sentiment_bias", "")).strip() or ("Bullish" if normalized_bias == "Call" else "Bearish"),
            "last_catalyst_days": last_catalyst_days,
            "key_dates": key_dates,
            "rationale": str(parsed.get("rationale", "")).strip()[:500],
            "raw_response": response_content,
            "citations_count": citations_count,
            "tool_usage": tool_usage,
            "api_call_duration": round(api_call_duration, 2),
            "model_used": model_used,
            "timestamp": now_et().isoformat()
        }

    except Exception as e:
        api_call_duration = time.time() - start_time
        logger.error(f"❌ Error getting market factors for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "reason": str(e),
            "api_call_duration": round(api_call_duration, 2),
            "model_used": model_used,
            "timestamp": now_et().isoformat()
        }


def send_discord_message(webhook_url: str, message: str, embed: dict = None):
    """Send a message to Discord webhook."""
    if not REQUESTS_AVAILABLE:
        logger.warning("requests library not available for Discord")
        return False
    
    if not webhook_url:
        logger.warning("No Discord webhook URL provided")
        return False
    
    try:
        payload = {"content": message[:2000]}  # Discord limit
        if embed:
            payload["embeds"] = [embed]
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("✅ Discord message sent successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send Discord message: {e}")
        return False


def _build_sentiment_prompt(symbol: str, hours_back: int, require_recommendation: bool = False) -> str:
    """Build the prompt for a sentiment (and optionally recommendation) analysis."""
    intro = (
        f"Analyze X/Twitter sentiment for ${symbol} stock over the last {hours_back} hours.\n"
        f"Compare the most recent {max(1, hours_back // 2)}h vs the previous {max(1, hours_back // 2)}h "
        f"to detect sentiment shifts.\n\n"
    )
    extra_schema = ""
    extra_guidance = ""
    if require_recommendation:
        extra_schema = (
            "- recommendation: MUST be exactly 'buy', 'hold', or 'sell', based on social sentiment ONLY "
            "(downstream code will factor in price/market data before acting)\n"
            "- confidence: one of 'low', 'medium', 'high'\n"
        )
        extra_guidance = (
            "\nRecommendation guidance:\n"
            "- Base recommendation on credible_score more than retail_score. If credible_score is flat but "
            "retail_score is very high, lean 'hold' and note it in contrarian_flags.\n"
            "- Be decisive when credible authors agree; use 'hold' when credible coverage is thin.\n"
        )

    return (
        intro
        + CREDIBILITY_RULES
        + extra_guidance
        + "\nReturn ONLY a JSON object with exactly these keys:\n"
        + SENTIMENT_JSON_SCHEMA
        + extra_schema
        + "\nNo prose outside the JSON."
    )


def _extract_parsed_data(response_content: str, symbol: str, context: str) -> Dict[str, Any]:
    """Extract JSON from a Grok response, tolerating code fences."""
    if not response_content:
        return {}
    json_str = response_content
    if "```json" in json_str:
        json_str = json_str.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, IndexError) as parse_error:
        logger.warning(f"JSON parse failed for {symbol} ({context}): {parse_error}")
        return {}


def _run_sentiment_pass(
    xai_client,
    symbol: str,
    hours_back: int,
    max_turns: int,
    prompt: str,
    allowed_handles: Optional[List[str]],
    excluded_handles: Optional[List[str]],
) -> Dict[str, Any]:
    """Single call to Grok with x_search. Returns raw/response metadata + parsed JSON."""
    now = datetime.datetime.now(pytz.UTC)
    from_date = now - datetime.timedelta(hours=hours_back)
    to_date = now

    tool_kwargs = {"from_date": from_date, "to_date": to_date}
    if allowed_handles:
        tool_kwargs["allowed_x_handles"] = allowed_handles
    if excluded_handles:
        tool_kwargs["excluded_x_handles"] = excluded_handles

    chat = xai_client.chat.create(
        model="grok-4-1-fast",
        tools=[x_search(**tool_kwargs)],
        max_turns=max_turns,
    )
    chat.append(user(prompt))
    response = chat.sample()

    response_content = response.content or ""
    parsed = _extract_parsed_data(response_content, symbol, "sentiment")

    tool_usage = {}
    if hasattr(response, 'server_side_tool_usage'):
        tool_usage = dict(response.server_side_tool_usage) if response.server_side_tool_usage else {}

    citations_raw = list(response.citations) if hasattr(response, 'citations') and response.citations else []
    citations_count = len(citations_raw)
    citations_sample = citations_raw[:10]

    return {
        "parsed": parsed,
        "response_content": response_content,
        "citations_count": citations_count,
        "citations_sample": citations_sample,
        "tool_usage": tool_usage,
    }


# Minimum citations required from a curated-handle search before we accept it.
MIN_CURATED_CITATIONS = 1


def analyze_sentiment(symbol: str, hours_back: int = 24, max_turns: int = 2):
    """
    Analyze X/Twitter sentiment for a stock symbol using xAI Agent Tools API.

    Two-pass search:
      1) restrict to handles in credible_handles.json (allow-list)
      2) if no posts matched, fall back to an unrestricted search and mark
         source_mode='open_fallback' + downgrade signal_confidence by one tier.

    Args:
        symbol: Stock ticker symbol (e.g., "SOFI", "AAPL")
        hours_back: How many hours of X posts to analyze (default 24)
        max_turns: Maximum tool call turns for cost control (default 2)

    Returns:
        dict with sentiment_score, summary, credible_score, retail_score,
        sample_size, echo_ratio, signal_confidence, top_sources, etc.
    """
    if not XAI_SDK_AVAILABLE:
        return {
            "status": "error",
            "reason": "xai_sdk not available. Install with: pip install xai-sdk>=1.5.0",
            "symbol": symbol
        }

    if not XAI_API_KEY:
        return {
            "status": "error",
            "reason": "XAI_API_KEY not configured in environment variables",
            "symbol": symbol
        }

    try:
        xai_client = XAIClient(api_key=XAI_API_KEY)
        logger.info(f"✅ xAI client initialized for {symbol}")
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to initialize xAI client: {e}",
            "symbol": symbol
        }

    handles = load_credible_handles()
    allowed = handles["allowed"]
    excluded = handles["excluded"]
    prompt = _build_sentiment_prompt(symbol, hours_back, require_recommendation=False)

    start_time = time.time()
    model_used = "grok-4-1-fast"
    source_mode = "open"
    fallback_used = False

    try:
        pass_result = None
        if allowed:
            logger.info(f"🔎 {symbol} | curated pass ({len(allowed)} handles)")
            pass_result = _run_sentiment_pass(
                xai_client, symbol, hours_back, max_turns, prompt,
                allowed_handles=allowed, excluded_handles=excluded,
            )
            source_mode = "curated"
            if pass_result.get("citations_count", 0) < MIN_CURATED_CITATIONS:
                logger.info(f"↪️ {symbol} | no curated citations, falling back to open search")
                pass_result = None
                fallback_used = True

        if pass_result is None:
            pass_result = _run_sentiment_pass(
                xai_client, symbol, hours_back, max_turns, prompt,
                allowed_handles=None, excluded_handles=excluded,
            )
            source_mode = "open_fallback" if fallback_used else "open"

        api_call_duration = time.time() - start_time

        parsed = pass_result["parsed"]
        response_content = pass_result["response_content"]

        if parsed:
            sentiment_fields = _parse_sentiment_fields(parsed)
        else:
            sentiment_fields = _parse_sentiment_fields({})
            sentiment_fields["summary"] = (response_content or "Unable to parse response")[:500]
            sentiment_fields["signal_confidence"] = "low"

        if fallback_used:
            sentiment_fields["signal_confidence"] = _downgrade_confidence(
                sentiment_fields["signal_confidence"]
            )

        citations_count = pass_result["citations_count"]

        logger.info(
            f"✅ {symbol} | score={sentiment_fields['sentiment_score']:.1f} "
            f"credible={sentiment_fields['credible_score']:.1f} "
            f"retail={sentiment_fields['retail_score']:.1f} "
            f"source={source_mode} citations={citations_count} "
            f"time={api_call_duration:.2f}s"
        )

        result = {
            "status": "success",
            "symbol": symbol,
            **sentiment_fields,
            "source_mode": source_mode,
            "handles_used": allowed if source_mode == "curated" else [],
            "raw_response": response_content,
            "citations_count": citations_count,
            "citations_sample": pass_result["citations_sample"],
            "tool_usage": pass_result["tool_usage"],
            "api_call_duration": round(api_call_duration, 2),
            "model_used": model_used,
            "hours_back": hours_back,
            "timestamp": now_et().isoformat(),
        }
        return result

    except Exception as e:
        api_call_duration = time.time() - start_time
        logger.error(f"❌ Error analyzing {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "reason": str(e),
            "api_call_duration": round(api_call_duration, 2),
            "model_used": model_used,
            "timestamp": now_et().isoformat()
        }


def get_stock_recommendation(symbol: str, max_turns: int = 2):
    """
    Get a buy/hold/sell recommendation for a stock symbol using xAI Agent Tools API.
    
    Args:
        symbol: Stock ticker symbol (e.g., "SOFI", "AAPL")
        max_turns: Maximum tool call turns for cost control (default 2)
    
    Returns:
        dict with recommendation (buy/hold/sell), buy_signal (bool), summary, etc.
    """
    if not XAI_SDK_AVAILABLE:
        return {
            "status": "error",
            "reason": "xai_sdk not available. Install with: pip install xai-sdk>=1.5.0",
            "symbol": symbol
        }
    
    if not XAI_API_KEY:
        return {
            "status": "error",
            "reason": "XAI_API_KEY not configured in environment variables",
            "symbol": symbol
        }
    
    # Initialize client
    try:
        xai_client = XAIClient(api_key=XAI_API_KEY)
        logger.info(f"✅ xAI client initialized for {symbol} recommendation")
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to initialize xAI client: {e}",
            "symbol": symbol
        }
    
    # Get current time in ET
    current_time_et = now_et()
    formatted_time = current_time_et.strftime("%B %d, %Y, %I:%M %p ET")
    
    # Calculate date ranges for x_search
    now_utc = datetime.datetime.now(pytz.UTC)
    from_date = now_utc - datetime.timedelta(hours=72)  # Look back 72 hours for context
    to_date = now_utc
    
    # User prompt for stock recommendation
    user_prompt = (
        f"**For short-term trade: Predict if ${symbol} will go up over the next few hours and recommend buy if yes.** "
        f"Recommend buy/hold/sell for ${symbol} as of {formatted_time}. "
        f"Factor in recent trends, fundamentals, key news, risks, and opportunities. "
        f"Structure: Trend Summary (brief, focus on intraday), Fundamentals (brief), News Impact (balanced positives and negatives), "
        f"Recommendation (buy/hold/sell with why). Keep under 400 words. "
        f"Provide your analysis as JSON with exactly these keys: "
        f"'recommendation' (string: 'buy', 'hold', or 'sell'), "
        f"'summary' (string: your full analysis under 400 words). "
        f"Return ONLY the JSON object, no other text."
    )
    
    start_time = time.time()
    model_used = "grok-4-1-fast"
    
    try:
        # Create chat with x_search tool
        chat = xai_client.chat.create(
            model=model_used,
            tools=[
                x_search(
                    from_date=from_date,
                    to_date=to_date,
                )
            ],
            max_turns=max_turns,
        )
        
        # Add the user message
        chat.append(user(user_prompt))
        
        # Get the response (non-streaming)
        response = chat.sample()
        
        api_call_duration = time.time() - start_time
        response_content = response.content
        
        # Parse JSON from response
        recommendation = "hold"
        summary = "Unable to parse response"
        
        try:
            json_str = response_content
            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            recommendation = str(data.get("recommendation", "hold")).lower().strip()
            summary = str(data.get("summary", "Summary not provided."))
        except (json.JSONDecodeError, IndexError) as parse_error:
            logger.warning(f"JSON parse failed for {symbol} recommendation: {parse_error}")
            summary = response_content[:500] if response_content else "Unable to parse response"
        
        # Normalize recommendation and determine buy_signal
        if recommendation not in ["buy", "hold", "sell"]:
            recommendation = "hold"
        buy_signal = recommendation == "buy"
        
        # Get usage stats
        tool_usage = {}
        if hasattr(response, 'server_side_tool_usage'):
            tool_usage = dict(response.server_side_tool_usage) if response.server_side_tool_usage else {}
        
        citations_count = len(response.citations) if hasattr(response, 'citations') and response.citations else 0
        
        logger.info(f"✅ {symbol} recommendation | {recommendation.upper()} | Buy Signal: {buy_signal} | Time: {api_call_duration:.2f}s")
        
        return {
            "status": "success",
            "symbol": symbol,
            "recommendation": recommendation,
            "buy_signal": buy_signal,
            "summary": summary,
            "raw_response": response_content,
            "citations_count": citations_count,
            "tool_usage": tool_usage,
            "api_call_duration": round(api_call_duration, 2),
            "model_used": model_used,
            "analysis_time": formatted_time,
            "timestamp": current_time_et.isoformat()
        }
        
    except Exception as e:
        api_call_duration = time.time() - start_time
        logger.error(f"❌ Error getting recommendation for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "reason": str(e),
            "api_call_duration": round(api_call_duration, 2),
            "model_used": model_used,
            "timestamp": now_et().isoformat()
        }


def get_aligned_recommendation(symbol: str, sentiment_score: float = None, hours_back: int = 24, max_turns: int = 2):
    """
    Get a sentiment-only buy/hold/sell recommendation by asking Grok directly.

    Uses the same curated-handle + fallback intake as analyze_sentiment, and asks
    Grok to produce a social-sentiment recommendation. Downstream code is expected
    to gate on price/market data before acting on this signal.

    Args:
        symbol: Stock ticker symbol
        sentiment_score: Ignored (retained for API compatibility)
        hours_back: Hours of X posts to analyze (default 24)
        max_turns: Maximum tool call turns (default 2)

    Returns:
        dict with sentiment_score, recommendation, buy_signal, confidence,
        credible_score, retail_score, sample_size, echo_ratio, source_mode, etc.
    """
    if not XAI_SDK_AVAILABLE:
        return {
            "status": "error",
            "reason": "xai_sdk not available. Install with: pip install xai-sdk>=1.5.0",
            "symbol": symbol
        }

    if not XAI_API_KEY:
        return {
            "status": "error",
            "reason": "XAI_API_KEY not configured in environment variables",
            "symbol": symbol
        }

    try:
        xai_client = XAIClient(api_key=XAI_API_KEY)
        logger.info(f"✅ xAI client initialized for aligned recommendation: {symbol}")
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to initialize xAI client: {e}",
            "symbol": symbol
        }

    current_time_et = now_et()
    handles = load_credible_handles()
    allowed = handles["allowed"]
    excluded = handles["excluded"]
    prompt = _build_sentiment_prompt(symbol, hours_back, require_recommendation=True)

    start_time = time.time()
    model_used = "grok-4-1-fast"
    source_mode = "open"
    fallback_used = False

    try:
        pass_result = None
        if allowed:
            logger.info(f"🔎 {symbol} | curated aligned pass ({len(allowed)} handles)")
            pass_result = _run_sentiment_pass(
                xai_client, symbol, hours_back, max_turns, prompt,
                allowed_handles=allowed, excluded_handles=excluded,
            )
            source_mode = "curated"
            if pass_result.get("citations_count", 0) < MIN_CURATED_CITATIONS:
                logger.info(f"↪️ {symbol} | aligned: no curated citations, falling back to open search")
                pass_result = None
                fallback_used = True

        if pass_result is None:
            pass_result = _run_sentiment_pass(
                xai_client, symbol, hours_back, max_turns, prompt,
                allowed_handles=None, excluded_handles=excluded,
            )
            source_mode = "open_fallback" if fallback_used else "open"

        api_call_duration = time.time() - start_time

        parsed = pass_result["parsed"]
        response_content = pass_result["response_content"]

        if parsed:
            sentiment_fields = _parse_sentiment_fields(parsed)
            recommendation = str(parsed.get("recommendation", "hold")).lower().strip()
            confidence = str(parsed.get("confidence", sentiment_fields["signal_confidence"])).lower().strip()
            alignment_reason = (
                f"GROK SENTIMENT: score={sentiment_fields['sentiment_score']:.1f} "
                f"credible={sentiment_fields['credible_score']:.1f} "
                f"retail={sentiment_fields['retail_score']:.1f} "
                f"source={source_mode}; downstream script should gate on price before trading."
            )
        else:
            sentiment_fields = _parse_sentiment_fields({})
            sentiment_fields["summary"] = (response_content or "Unable to parse response")[:500]
            sentiment_fields["signal_confidence"] = "low"
            recommendation = "hold"
            confidence = "low"
            alignment_reason = "PARSE_ERROR: Could not parse Grok response"

        if recommendation not in ("buy", "hold", "sell"):
            recommendation = "hold"
        if confidence not in ("low", "medium", "high"):
            confidence = "medium"

        if fallback_used:
            sentiment_fields["signal_confidence"] = _downgrade_confidence(
                sentiment_fields["signal_confidence"]
            )
            confidence = _downgrade_confidence(confidence)

        buy_signal = recommendation == "buy"
        citations_count = pass_result["citations_count"]

        logger.info(
            f"✅ {symbol} | aligned rec={recommendation.upper()} "
            f"score={sentiment_fields['sentiment_score']:.1f} "
            f"credible={sentiment_fields['credible_score']:.1f} "
            f"retail={sentiment_fields['retail_score']:.1f} "
            f"source={source_mode} citations={citations_count}"
        )

        return {
            "status": "success",
            "symbol": symbol,
            **sentiment_fields,
            "recommendation": recommendation,
            "buy_signal": buy_signal,
            "confidence": confidence,
            "alignment_reason": alignment_reason,
            "source_mode": source_mode,
            "handles_used": allowed if source_mode == "curated" else [],
            "raw_response": response_content,
            "citations_count": citations_count,
            "citations_sample": pass_result["citations_sample"],
            "tool_usage": pass_result["tool_usage"],
            "api_call_duration": round(api_call_duration, 2),
            "model_used": model_used,
            "recommended_hold_hours": 36,
            "timestamp": current_time_et.isoformat()
        }

    except Exception as e:
        api_call_duration = time.time() - start_time
        logger.error(f"❌ Error getting aligned recommendation for {symbol}: {e}")
        return {
            "status": "error",
            "symbol": symbol,
            "reason": str(e),
            "api_call_duration": round(api_call_duration, 2),
            "model_used": model_used,
            "timestamp": now_et().isoformat()
        }


def format_discord_embed(result: dict) -> dict:
    """Format the result as a Discord embed."""
    if result.get("status") != "success":
        return {
            "title": f"❌ Sentiment Analysis Failed: {result.get('symbol', 'Unknown')}",
            "description": result.get("reason", "Unknown error"),
            "color": 15158332  # Red
        }
    
    score = result.get("sentiment_score", 0)
    
    # Determine color based on sentiment
    if score > 3:
        color = 3066993  # Green
        emoji = "🟢"
        direction = "BULLISH"
    elif score < -3:
        color = 15158332  # Red
        emoji = "🔴"
        direction = "BEARISH"
    else:
        color = 16776960  # Yellow
        emoji = "⚪"
        direction = "NEUTRAL"
    
    return {
        "title": f"📊 Sentiment Analysis: {result.get('symbol')}",
        "description": result.get("summary", "No summary available"),
        "color": color,
        "fields": [
            {
                "name": "Sentiment Score",
                "value": f"{emoji} **{score:.1f}/10** ({direction})",
                "inline": True
            },
            {
                "name": "X Posts Analyzed",
                "value": f"📱 {result.get('citations_count', 0)} posts",
                "inline": True
            },
            {
                "name": "Analysis Time",
                "value": f"⏱️ {result.get('api_call_duration', 0):.1f}s",
                "inline": True
            },
            {
                "name": "Tool Usage",
                "value": f"🔧 {result.get('tool_usage', {})}",
                "inline": False
            }
        ],
        "footer": {
            "text": f"Powered by xAI Grok | {result.get('model_used', 'grok-4-1-fast')}"
        },
        "timestamp": result.get("timestamp", now_et().isoformat())
    }


@functions_framework.http
def grok_sentiment(request):
    """
    HTTP Cloud Function for Grok sentiment analysis.
    
    Request JSON body:
    {
        "symbol": "SOFI",           # Required: Stock ticker
        "hours_back": 24,           # Optional: Hours of X posts to analyze (default 24)
        "max_turns": 2,             # Optional: Max tool call turns (default 2)
        "send_to_discord": false,   # Optional: Send result to Discord (default false)
        "discord_webhook_url": ""   # Optional: Override default Discord webhook
    }
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for the main response
    headers = {'Access-Control-Allow-Origin': '*'}
    
    # Parse request
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return jsonify({
                "status": "error",
                "reason": "No JSON body provided"
            }), 400, headers
    except Exception as e:
        return jsonify({
            "status": "error",
            "reason": f"Failed to parse JSON: {e}"
        }), 400, headers
    
    # Extract parameters
    symbol = request_json.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({
            "status": "error",
            "reason": "Missing required parameter: symbol"
        }), 400, headers
    
    hours_back = request_json.get('hours_back', 24)
    max_turns = request_json.get('max_turns', 2)
    send_to_discord = request_json.get('send_to_discord', False)
    discord_webhook = request_json.get('discord_webhook_url', DISCORD_WEBHOOK_URL)
    
    logger.info(f"📊 Analyzing sentiment for {symbol} (hours_back={hours_back}, max_turns={max_turns})")
    
    # Run sentiment analysis
    result = analyze_sentiment(
        symbol=symbol,
        hours_back=hours_back,
        max_turns=max_turns
    )
    
    # Send to Discord if requested
    if send_to_discord and discord_webhook:
        embed = format_discord_embed(result)
        send_discord_message(discord_webhook, "", embed)
    
    return jsonify(result), 200, headers


@functions_framework.http
def grok_recommendation(request):
    """
    HTTP Cloud Function for Grok stock recommendation.
    
    Request JSON body:
    {
        "symbol": "SOFI",           # Required: Stock ticker
        "max_turns": 2,             # Optional: Max tool call turns (default 2)
        "send_to_discord": false,   # Optional: Send result to Discord (default false)
        "discord_webhook_url": ""   # Optional: Override default Discord webhook
    }
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for the main response
    headers = {'Access-Control-Allow-Origin': '*'}
    
    # Parse request
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return jsonify({
                "status": "error",
                "reason": "No JSON body provided"
            }), 400, headers
    except Exception as e:
        return jsonify({
            "status": "error",
            "reason": f"Failed to parse JSON: {e}"
        }), 400, headers
    
    # Extract parameters
    symbol = request_json.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({
            "status": "error",
            "reason": "Missing required parameter: symbol"
        }), 400, headers
    
    max_turns = request_json.get('max_turns', 2)
    send_to_discord = request_json.get('send_to_discord', False)
    discord_webhook = request_json.get('discord_webhook_url', DISCORD_WEBHOOK_URL)
    
    logger.info(f"📈 Getting recommendation for {symbol} (max_turns={max_turns})")
    
    # Run recommendation analysis
    result = get_stock_recommendation(
        symbol=symbol,
        max_turns=max_turns
    )
    
    # Send to Discord if requested
    if send_to_discord and discord_webhook:
        embed = format_recommendation_embed(result)
        send_discord_message(discord_webhook, "", embed)
    
    return jsonify(result), 200, headers


@functions_framework.http
def grok_aligned_recommendation(request):
    """
    HTTP Cloud Function for aligned Grok sentiment + recommendation (ONE API CALL).
    
    This is the cost-effective endpoint that combines sentiment analysis with
    buy/hold/sell recommendation in a single call. High sentiment scores 
    automatically result in buy signals based on calibrated thresholds.
    
    Request JSON body:
    {
        "symbol": "SOFI",           # Required: Stock ticker
        "hours_back": 24,           # Optional: Hours of X posts to analyze (default 24)
        "max_turns": 2,             # Optional: Max tool call turns (default 2)
        "send_to_discord": false,   # Optional: Send result to Discord (default false)
        "discord_webhook_url": ""   # Optional: Override default Discord webhook
    }
    
    Returns:
    {
        "status": "success",
        "symbol": "SOFI",
        "sentiment_score": 9.2,
        "recommendation": "buy",
        "buy_signal": true,
        "alignment_reason": "SENTIMENT OVERRIDE: Score 9.2 >= 9.0 (historically +2.04% avg return)",
        "confidence": "high",
        "summary": "...",
        "citations_count": 5,
        "recommended_hold_hours": 36,
        "timestamp": "2026-01-13T10:30:00-05:00"
    }
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for the main response
    headers = {'Access-Control-Allow-Origin': '*'}
    
    # Parse request
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return jsonify({
                "status": "error",
                "reason": "No JSON body provided"
            }), 400, headers
    except Exception as e:
        return jsonify({
            "status": "error",
            "reason": f"Failed to parse JSON: {e}"
        }), 400, headers
    
    # Extract parameters
    symbol = request_json.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({
            "status": "error",
            "reason": "Missing required parameter: symbol"
        }), 400, headers
    
    hours_back = request_json.get('hours_back', 24)
    max_turns = request_json.get('max_turns', 2)
    send_to_discord = request_json.get('send_to_discord', False)
    discord_webhook = request_json.get('discord_webhook_url', DISCORD_WEBHOOK_URL)
    
    factors_mode = (
        bool(request_json.get('market_factors'))
        or str(request_json.get('analysis_type', '')).strip().lower() == 'market_factors'
        or 'catalyst_window_days' in request_json
    )

    if factors_mode:
        catalyst_window_days = request_json.get('catalyst_window_days', 60)
        logger.info(
            f"🧭 Getting market factors for {symbol} via aligned endpoint "
            f"(window_days={catalyst_window_days}, max_turns={max_turns})"
        )
        result = get_grok_market_factors(
            ticker=symbol,
            max_turns=max_turns,
            catalyst_window_days=catalyst_window_days,
        )
    else:
        logger.info(f"🎯 Getting ALIGNED recommendation for {symbol} (hours_back={hours_back}, max_turns={max_turns})")

        # Run aligned recommendation (ONE API CALL - cost effective!)
        result = get_aligned_recommendation(
            symbol=symbol,
            sentiment_score=None,  # Will analyze sentiment internally
            hours_back=hours_back,
            max_turns=max_turns
        )
    
    # Send to Discord if requested
    if send_to_discord and discord_webhook:
        if factors_mode:
            message = (
                f"🧭 Market factors {result.get('symbol', symbol)} | "
                f"Bias: {result.get('bias', 'N/A')} | "
                f"Min Days: {result.get('recommended_min_days', 'N/A')} | "
                f"Vol Risk: {result.get('volatility_risk', 'N/A')}"
            )
            send_discord_message(discord_webhook, message)
        else:
            embed = format_aligned_embed(result)
            send_discord_message(discord_webhook, "", embed)
    
    return jsonify(result), 200, headers


@functions_framework.http
def grok_market_factors(request):
    """
    HTTP Cloud Function for catalyst-aware market factor analysis.

    Request JSON body:
    {
        "symbol": "AAPL",                  # Required
        "max_turns": 3,                     # Optional
        "catalyst_window_days": 60          # Optional
    }
    """
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return jsonify({"status": "error", "reason": "No JSON body provided"}), 400, headers
    except Exception as e:
        return jsonify({"status": "error", "reason": f"Failed to parse JSON: {e}"}), 400, headers

    symbol = request_json.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({"status": "error", "reason": "Missing required parameter: symbol"}), 400, headers

    max_turns = request_json.get('max_turns', 3)
    catalyst_window_days = request_json.get('catalyst_window_days', 60)

    logger.info(
        f"🧭 Getting market factors for {symbol} "
        f"(max_turns={max_turns}, window_days={catalyst_window_days})"
    )

    result = get_grok_market_factors(
        ticker=symbol,
        max_turns=max_turns,
        catalyst_window_days=catalyst_window_days
    )

    status_code = 200 if result.get("status") == "success" else 500
    return jsonify(result), status_code, headers


def format_aligned_embed(result: dict) -> dict:
    """Format the aligned recommendation result as a Discord embed."""
    if result.get("status") != "success":
        return {
            "title": f"❌ Aligned Recommendation Failed: {result.get('symbol', 'Unknown')}",
            "description": result.get("reason", "Unknown error"),
            "color": 15158332  # Red
        }
    
    recommendation = result.get("recommendation", "hold")
    buy_signal = result.get("buy_signal", False)
    sentiment_score = result.get("sentiment_score", 0)
    confidence = result.get("confidence", "medium")
    
    # Determine color and emoji based on recommendation
    if recommendation == "buy":
        color = 3066993  # Green
        emoji = "🟢"
    elif recommendation == "sell":
        color = 15158332  # Red
        emoji = "🔴"
    else:
        color = 16776960  # Yellow
        emoji = "⚪"
    
    # Confidence emoji
    conf_emoji = "🔥" if confidence == "high" else "📊" if confidence == "medium" else "❓"
    
    credible_score = result.get("credible_score", 0.0) or 0.0
    retail_score = result.get("retail_score", 0.0) or 0.0
    sample_size = result.get("sample_size", 0) or 0
    echo_ratio = result.get("echo_ratio", 0.0) or 0.0
    source_mode = result.get("source_mode", "open")
    flags = result.get("contrarian_flags") or []
    top_sources = result.get("top_sources") or []

    fields = [
        {"name": "Overall Score", "value": f"📈 **{sentiment_score:.1f}** / 10", "inline": True},
        {"name": "Credible vs Retail",
         "value": f"🏦 {credible_score:.1f}  |  📣 {retail_score:.1f}",
         "inline": True},
        {"name": "Recommendation",
         "value": f"{emoji} **{recommendation.upper()}**",
         "inline": True},
        {"name": "Confidence", "value": f"{conf_emoji} {confidence.upper()}", "inline": True},
        {"name": "Sample / Echo",
         "value": f"🧪 {sample_size} posts · 🔁 {echo_ratio:.2f}",
         "inline": True},
        {"name": "Source Mode", "value": f"🎯 {source_mode}", "inline": True},
    ]
    if top_sources:
        fields.append({
            "name": "Top Sources",
            "value": ", ".join(f"@{h}" for h in top_sources[:5]),
            "inline": False,
        })
    if flags:
        fields.append({
            "name": "Contrarian Flags",
            "value": "\n".join(f"• {f}" for f in flags[:5])[:1024],
            "inline": False,
        })
    fields.append({
        "name": "Alignment Reason",
        "value": (result.get("alignment_reason") or "N/A")[:1024],
        "inline": False,
    })

    return {
        "title": f"🎯 Aligned Recommendation: {result.get('symbol')}",
        "description": result.get("summary", "No summary available")[:4000],
        "color": color,
        "fields": fields,
        "footer": {
            "text": "Powered by xAI Grok | Social sentiment only — downstream code applies price gating"
        },
        "timestamp": result.get("timestamp", now_et().isoformat())
    }


def format_recommendation_embed(result: dict) -> dict:
    """Format the recommendation result as a Discord embed."""
    if result.get("status") != "success":
        return {
            "title": f"❌ Recommendation Failed: {result.get('symbol', 'Unknown')}",
            "description": result.get("reason", "Unknown error"),
            "color": 15158332  # Red
        }
    
    recommendation = result.get("recommendation", "hold")
    buy_signal = result.get("buy_signal", False)
    
    # Determine color and emoji based on recommendation
    if recommendation == "buy":
        color = 3066993  # Green
        emoji = "🟢"
    elif recommendation == "sell":
        color = 15158332  # Red
        emoji = "🔴"
    else:
        color = 16776960  # Yellow
        emoji = "⚪"
    
    return {
        "title": f"📈 Stock Recommendation: {result.get('symbol')}",
        "description": result.get("summary", "No summary available")[:4000],  # Discord limit
        "color": color,
        "fields": [
            {
                "name": "Recommendation",
                "value": f"{emoji} **{recommendation.upper()}**",
                "inline": True
            },
            {
                "name": "Buy Signal",
                "value": f"{'✅ YES' if buy_signal else '❌ NO'}",
                "inline": True
            },
            {
                "name": "Analysis Time",
                "value": f"⏱️ {result.get('api_call_duration', 0):.1f}s",
                "inline": True
            },
            {
                "name": "As Of",
                "value": f"🕐 {result.get('analysis_time', 'N/A')}",
                "inline": False
            }
        ],
        "footer": {
            "text": f"Powered by xAI Grok | {result.get('model_used', 'grok-4-1-fast')}"
        },
        "timestamp": result.get("timestamp", now_et().isoformat())
    }


# For local testing with functions-framework
if __name__ == "__main__":
    import sys
    
    # Quick test without HTTP
    # Usage: python main.py SOFI [sentiment|recommendation|aligned]
    test_symbol = sys.argv[1] if len(sys.argv) > 1 else "SOFI"
    test_mode = sys.argv[2] if len(sys.argv) > 2 else "aligned"
    
    if test_mode == "recommendation":
        print(f"\n📈 Testing Grok Recommendation for {test_symbol}...\n")
        result = get_stock_recommendation(test_symbol, max_turns=2)
    elif test_mode == "aligned":
        print(f"\n🎯 Testing Aligned Recommendation for {test_symbol}...\n")
        result = get_aligned_recommendation(test_symbol, hours_back=24, max_turns=2)
    else:
        print(f"\n🧪 Testing Grok Sentiment for {test_symbol}...\n")
        result = analyze_sentiment(test_symbol, hours_back=24, max_turns=2)
    
    print(json.dumps(result, indent=2))
