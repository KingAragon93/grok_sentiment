# Grok Sentiment Cloud Function

<img src="https://avatars2.githubusercontent.com/u/2810941?v=3&s=96" alt="Google Cloud Platform logo" title="Google Cloud Platform" align="right" height="96" width="96"/>

A Cloud Run function that provides real-time sentiment analysis and **aligned buy/hold/sell recommendations** for stock tickers using xAI's Grok API with `x_search` to analyze X/Twitter posts.

## Table of Contents

* [Quick Start](#quick-start)
* [API Endpoints](#api-endpoints)
* [Directory Structure](#directory-structure)
* [Local Development](#local-development)
* [Deployment](#deployment)
* [Environment Variables](#environment-variables)
* [Analysis Tools](#analysis-tools)

---

## Quick Start

**Use the cost-effective aligned endpoint (1 API call instead of 2):**

```bash
curl -X POST https://YOUR_URL/grok_aligned_recommendation \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SOFI", "hours_back": 24}'
```

**Response:**
```json
{
    "status": "success",
    "symbol": "SOFI",
    "sentiment_score": 7.8,
    "credible_score": 4.1,
    "retail_score": 9.3,
    "recommendation": "hold",
    "buy_signal": false,
    "confidence": "medium",
    "signal_confidence": "medium",
    "source_mode": "curated",
    "handles_used": ["WSJmarkets", "charliebilello", "..."],
    "bullish_pct": 68, "bearish_pct": 12, "neutral_pct": 20,
    "sample_size": 74, "unique_authors": 49, "echo_ratio": 0.31,
    "top_themes": ["breakout", "earnings"],
    "top_sources": ["charliebilello", "unusual_whales"],
    "contrarian_flags": ["retail euphoria well above credible coverage"],
    "summary": "Retail chatter bullish on breakout narrative; credible voices more muted.",
    "alignment_reason": "GROK SENTIMENT: score=7.8 credible=4.1 retail=9.3 source=curated; downstream script should gate on price before trading.",
    "recommended_hold_hours": 36,
    "timestamp": "2026-04-20T10:30:00-04:00"
}
```

> **Scope:** This service reports **social sentiment only**. It does NOT
> read stock price. Downstream callers are expected to combine these signals
> with price/market data before trading.

---

## API Endpoints

### 1. Aligned Recommendation (RECOMMENDED)

**Endpoint:** `POST /grok_aligned_recommendation`

Runs the sentiment intake (two-pass curated allow-list → open fallback) and
returns a sentiment-only `recommendation` in a single API call.

| Request Field | Type | Required | Default | Description |
|---------------|------|----------|---------|-------------|
| `symbol` | string | Yes | - | Stock ticker symbol |
| `hours_back` | int | No | 24 | Hours of X posts to analyze |
| `max_turns` | int | No | 2 | Max Grok tool call turns |
| `send_to_discord` | bool | No | false | Send result to Discord |
| `discord_webhook_url` | string | No | env var | Override Discord webhook |

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `sentiment_score` | float | -10 to +10, credibility-weighted overall |
| `credible_score` | float | -10 to +10, tier-1/tier-2 authors only |
| `retail_score` | float | -10 to +10, tier-3/tier-4 authors only |
| `bullish_pct` / `bearish_pct` / `neutral_pct` | int | Distribution of weighted posts |
| `sample_size` | int | Distinct posts after dedupe |
| `unique_authors` | int | Distinct authors considered |
| `echo_ratio` | float | 0-1, share of retweets/near-duplicates |
| `signal_confidence` | string | "low" / "medium" / "high" for the intake itself |
| `source_mode` | string | `curated` (allow-list hit), `open_fallback` (allow-list empty, fell back to open search), or `open` (no allow-list configured) |
| `handles_used` | array | X handles that were queried when source_mode=curated |
| `top_sources` / `top_themes` / `contrarian_flags` | array | Drivers of the score |
| `recommendation` | string | "buy", "hold", or "sell" based on SOCIAL SENTIMENT ONLY |
| `buy_signal` | bool | True if recommendation is buy (not a trade signal by itself) |
| `confidence` | string | Grok's confidence in the recommendation |
| `alignment_reason` | string | Human-readable explanation |
| `recommended_hold_hours` | int | Optimal holding period (36h) |
| `summary` | string | One-sentence summary |

The allow-list lives in [`credible_handles.json`](credible_handles.json).
Add or remove handles there to change what the curated pass searches.

---

### 2. Sentiment Analysis Only

**Endpoint:** `POST /grok_sentiment`

Same intake as the aligned endpoint, without a `recommendation`.

```json
{
    "symbol": "SOFI",
    "hours_back": 24,
    "max_turns": 2
}
```

Returns the full sentiment schema (`sentiment_score`, `credible_score`,
`retail_score`, `source_mode`, `sample_size`, `echo_ratio`,
`signal_confidence`, `top_sources`, `contrarian_flags`, `summary`, …).

---

### 3. Standalone Recommendation

**Endpoint:** `POST /grok_recommendation`

Get a recommendation based on trends, fundamentals, and news (separate API call).

```json
{
    "symbol": "SOFI",
    "max_turns": 2
}
```

---

## Directory Structure

```
grok_sentiment/
├── main.py                      # Cloud Run functions (all 3 endpoints)
├── gcs_logger.py               # GCS logging for aligned recommendations
├── analyze_recommendations.py   # Analysis tool for metrics & reporting
├── sentiment_analyzer.py        # Historical performance analyzer
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── cloudbuild.yaml             # Cloud Build CI/CD
├── MIGRATION_INSTRUCTIONS.txt   # Guide for migrating to aligned endpoint
└── src/
    └── GoogleCloudServiceAccount.json  # GCS credentials (not in git)
```

---

## Local Development

### Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy .env.example to .env and fill in your API keys
```

### Run Locally

```bash
# Test aligned recommendation
python main.py SOFI aligned

# Test sentiment only
python main.py SOFI sentiment

# Run the analysis tool
python analyze_recommendations.py          # Full analysis
python analyze_recommendations.py --today  # Today only
python analyze_recommendations.py --buys   # Buy metrics

# Start local server
functions-framework --target=grok_aligned_recommendation --host=localhost --port=8080
```

---

## Deployment

### Cloud Build (Recommended)

```bash
gcloud builds submit --config cloudbuild.yaml .
```

### Direct Deploy

```bash
# Deploy aligned recommendation endpoint
gcloud functions deploy grok_aligned_recommendation \
    --runtime python311 \
    --trigger-http \
    --entry-point grok_aligned_recommendation \
    --allow-unauthenticated \
    --memory 512MB \
    --timeout 60s \
    --set-env-vars "XAI_API_KEY=your_key"
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `XAI_API_KEY` | Yes | xAI API key for Grok |
| `GOOGLE_APPLICATION_CREDENTIALS` | No | Path to GCS service account JSON |
| `GCS_BUCKET_NAME` | No | GCS bucket for logging (default: historical_stock_day) |
| `DISCORD_WEBHOOK_URL` | No | Discord webhook for notifications |
| `ALPACA_API_KEY` | No | Alpaca API key for price tracking |
| `ALPACA_SECRET_KEY` | No | Alpaca secret key |

---

## Analysis Tools

### Analyze Recommendations

Run analysis on historical and aligned recommendation data:

```bash
# Full analysis with all metrics
python analyze_recommendations.py

# Today's activity only
python analyze_recommendations.py --today

# Buy recommendation metrics
python analyze_recommendations.py --buys
```

**Output includes:**
- Total records and signal distribution
- Sentiment averages by recommendation type
- Performance by score range (9+, 8-9, 7-8, etc.)
- Today's calls and their details
- Buy accuracy rates (when tracked)

### GCS Logging

All aligned recommendations are logged to GCS for future analysis:

```
gs://historical_stock_day/aligned_recommendations_log.json
```

Initialize the log file:
```bash
python gcs_logger.py
```

---

## Cost Optimization

| Feature | Setting | Benefit |
|---------|---------|---------|
| **Aligned endpoint** | 1 call | 50% cost reduction vs 2 calls |
| Model | `grok-4-1-fast` | Faster, cheaper than grok-4 |
| Tools | `x_search()` only | Fewer API calls |
| `max_turns` | 2 (default) | Limits tool iterations |

---

## Migration from Old Endpoints

If you were calling `grok_sentiment` + `grok_recommendation` separately, switch to `grok_aligned_recommendation`.

See [MIGRATION_INSTRUCTIONS.txt](MIGRATION_INSTRUCTIONS.txt) for detailed migration guide with code examples.

---

## Agent Documentation Maintenance

Refresh and validate retrieval docs/indexes:

```bash
python scripts/refresh_agent_docs.py
```

Direct commands:

```bash
python scripts/validate_doc_references.py
python scripts/generate_agent_symbol_index.py
```

---

## Clean Up

```bash
gcloud run services delete groksentiment --region us-central1
```

---

## License

MIT License
