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
from typing import Any, Dict, List
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


def analyze_sentiment(symbol: str, hours_back: int = 24, max_turns: int = 2):
    """
    Analyze sentiment for a stock symbol using xAI Agent Tools API.
    
    Args:
        symbol: Stock ticker symbol (e.g., "SOFI", "AAPL")
        hours_back: How many hours of X posts to analyze (default 24)
        max_turns: Maximum tool call turns for cost control (default 2)
    
    Returns:
        dict with sentiment_score, summary, citations_count, etc.
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
        logger.info(f"✅ xAI client initialized for {symbol}")
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to initialize xAI client: {e}",
            "symbol": symbol
        }
    
    # Calculate date ranges
    now = datetime.datetime.now(pytz.UTC)
    from_date = now - datetime.timedelta(hours=hours_back)
    to_date = now
    
    # User prompt for sentiment analysis
    user_prompt = (
        f"Analyze sentiment for ${symbol} stock on X/Twitter. "
        f"Search for posts from the last {hours_back} hours about ${symbol}. "
        f"Compare the last {hours_back//2} hours vs previous {hours_back//2} hours to detect any sentiment shifts. "
        f"Provide your analysis as JSON with exactly these keys: "
        f"'sentiment_score' (number from -10 to +10), "
        f"'summary' (ONE sentence max 100 words: sentiment shift direction, main catalyst, price alignment). "
        f"Return ONLY the JSON object, no other text."
    )
    
    start_time = time.time()
    model_used = "grok-4-1-fast"
    
    try:
        # Create chat with x_search tool only (cost-efficient)
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
        sentiment_score = 0.0
        summary = "Unable to parse response"
        
        try:
            json_str = response_content
            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            sentiment_score = float(data.get("sentiment_score", 0.0))
            summary = str(data.get("summary", "Summary not provided."))
        except (json.JSONDecodeError, IndexError) as parse_error:
            logger.warning(f"JSON parse failed for {symbol}: {parse_error}")
            summary = response_content[:500] if response_content else "Unable to parse response"
        
        # Get usage stats
        tool_usage = {}
        if hasattr(response, 'server_side_tool_usage'):
            tool_usage = dict(response.server_side_tool_usage) if response.server_side_tool_usage else {}
        
        citations_count = len(response.citations) if hasattr(response, 'citations') and response.citations else 0
        citations = list(response.citations)[:10] if hasattr(response, 'citations') and response.citations else []
        
        logger.info(f"✅ {symbol} analyzed | Score: {sentiment_score:.1f}/10 | Time: {api_call_duration:.2f}s | Citations: {citations_count}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "sentiment_score": sentiment_score,
            "summary": summary,
            "raw_response": response_content,
            "citations_count": citations_count,
            "citations_sample": citations,
            "tool_usage": tool_usage,
            "api_call_duration": round(api_call_duration, 2),
            "model_used": model_used,
            "hours_back": hours_back,
            "timestamp": now_et().isoformat()
        }
        
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
    Get an aligned buy/hold/sell recommendation by asking Grok directly.
    
    This function combines sentiment analysis with buy/hold/sell recommendation
    in a SINGLE Grok API call, asking Grok to provide both the sentiment score
    AND the trading recommendation based on that sentiment.
    
    Args:
        symbol: Stock ticker symbol
        sentiment_score: Optional pre-calculated sentiment score (ignored - we ask Grok fresh)
        hours_back: Hours of X posts to analyze (default 24)
        max_turns: Maximum tool call turns (default 2)
    
    Returns:
        dict with sentiment_score, recommendation, buy_signal, confidence, etc.
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
        logger.info(f"✅ xAI client initialized for aligned recommendation: {symbol}")
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to initialize xAI client: {e}",
            "symbol": symbol
        }
    
    # Calculate date ranges
    now = datetime.datetime.now(pytz.UTC)
    from_date = now - datetime.timedelta(hours=hours_back)
    to_date = now
    current_time_et = now_et()
    formatted_time = current_time_et.strftime("%B %d, %Y, %I:%M %p ET")
    
    # Combined prompt: Get BOTH sentiment score AND buy/hold/sell recommendation
    user_prompt = (
        f"Analyze ${symbol} stock for a SHORT-TERM TRADE decision as of {formatted_time}. "
        f"Search X/Twitter for posts from the last {hours_back} hours about ${symbol}. "
        f"\n\nProvide your analysis as JSON with these exact keys:\n"
        f"1. 'sentiment_score': number from -10 to +10 based on X/Twitter sentiment\n"
        f"2. 'recommendation': MUST be exactly 'buy', 'hold', or 'sell' - your trading recommendation\n"
        f"3. 'confidence': 'high', 'medium', or 'low' based on how confident you are\n"
        f"4. 'summary': Brief explanation (under 100 words) of sentiment, catalysts, and why you recommend buy/hold/sell\n"
        f"\n**IMPORTANT GUIDELINES**:\n"
        f"- If sentiment_score >= 8.0, you should strongly consider 'buy' unless there are clear risks\n"
        f"- If sentiment_score >= 6.0, lean towards 'buy' if momentum is positive\n"
        f"- Be decisive - avoid defaulting to 'hold' when sentiment is clearly bullish\n"
        f"- Consider: sentiment momentum, catalysts, price action alignment\n"
        f"\nReturn ONLY the JSON object, no other text."
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
        sentiment_score = 0.0
        recommendation = "hold"
        confidence = "medium"
        summary = "Unable to parse response"
        alignment_reason = ""
        
        try:
            json_str = response_content
            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            sentiment_score = float(data.get("sentiment_score", 0.0))
            recommendation = str(data.get("recommendation", "hold")).lower().strip()
            confidence = str(data.get("confidence", "medium")).lower().strip()
            summary = str(data.get("summary", "Summary not provided."))
            
            # Normalize recommendation
            if recommendation not in ["buy", "hold", "sell"]:
                recommendation = "hold"
            
            # Normalize confidence
            if confidence not in ["high", "medium", "low"]:
                confidence = "medium"
            
            # Create alignment reason based on Grok's decision
            alignment_reason = f"GROK ANALYSIS: Score {sentiment_score:.1f}, recommends {recommendation.upper()} with {confidence} confidence"
            
            # Safety override: If sentiment is very high (9+) but Grok said hold, override to buy
            if sentiment_score >= 9.0 and recommendation == "hold":
                recommendation = "buy"
                alignment_reason = f"SENTIMENT OVERRIDE: Score {sentiment_score:.1f} >= 9.0 overrides hold to buy"
                logger.info(f"⚠️ {symbol} | Override: {sentiment_score:.1f} sentiment -> BUY")
            
        except (json.JSONDecodeError, IndexError) as parse_error:
            logger.warning(f"JSON parse failed for {symbol} aligned: {parse_error}")
            summary = response_content[:500] if response_content else "Unable to parse response"
            alignment_reason = "PARSE_ERROR: Could not parse Grok response"
        
        # Determine buy_signal
        buy_signal = recommendation == "buy"
        
        # Get usage stats
        tool_usage = {}
        if hasattr(response, 'server_side_tool_usage'):
            tool_usage = dict(response.server_side_tool_usage) if response.server_side_tool_usage else {}
        
        citations_count = len(response.citations) if hasattr(response, 'citations') and response.citations else 0
        
        logger.info(f"✅ {symbol} | ALIGNED {recommendation.upper()} | Score: {sentiment_score:.1f} | Conf: {confidence} | Buy: {buy_signal}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "sentiment_score": sentiment_score,
            "recommendation": recommendation,
            "buy_signal": buy_signal,
            "confidence": confidence,
            "alignment_reason": alignment_reason,
            "summary": summary,
            "raw_response": response_content,
            "citations_count": citations_count,
            "tool_usage": tool_usage,
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
    
    return {
        "title": f"🎯 Aligned Recommendation: {result.get('symbol')}",
        "description": result.get("summary", "No summary available")[:4000],
        "color": color,
        "fields": [
            {
                "name": "Sentiment Score",
                "value": f"📈 **{sentiment_score:.1f}** / 10",
                "inline": True
            },
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
                "name": "Confidence",
                "value": f"{conf_emoji} {confidence.upper()}",
                "inline": True
            },
            {
                "name": "Hold Period",
                "value": f"⏱️ {result.get('recommended_hold_hours', 36)}h",
                "inline": True
            },
            {
                "name": "Citations",
                "value": f"📰 {result.get('citations_count', 0)}",
                "inline": True
            },
            {
                "name": "Alignment Reason",
                "value": result.get("alignment_reason", "N/A")[:1024],
                "inline": False
            }
        ],
        "footer": {
            "text": "Powered by xAI Grok | Aligned Sentiment + Recommendation"
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
