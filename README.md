# Grok Sentiment Cloud Function

<img src="https://avatars2.githubusercontent.com/u/2810941?v=3&s=96" alt="Google Cloud Platform logo" title="Google Cloud Platform" align="right" height="96" width="96"/>

A Cloud Run function that provides real-time sentiment analysis for stock tickers using xAI's Agent Tools API with `x_search` to analyze X/Twitter posts.

## Table of Contents

* [Directory contents](#directory-contents)
* [API Usage](#api-usage)
* [Getting started with VS Code](#getting-started-with-vs-code)
* [Local Development](#local-development)
* [Deployment](#deployment)
* [Environment Variables](#environment-variables)

## Directory contents

* `.vscode/launch.json` - VS Code launch configurations
* `main.py` - The Grok sentiment analysis Cloud Function code
* `requirements.txt` - Python dependencies
* `Dockerfile` - Container configuration for Cloud Run
* `cloudbuild.yaml` - Cloud Build configuration for CI/CD

## API Usage

### Endpoint

```
POST https://groksentiment-1039781888202.us-central1.run.app
```

### Request Body

```json
{
    "symbol": "SOFI",           // Required: Stock ticker symbol
    "hours_back": 24,           // Optional: Hours of X posts to analyze (default: 24)
    "max_turns": 2,             // Optional: Max tool call turns for cost control (default: 2)
    "send_to_discord": false,   // Optional: Send result to Discord (default: false)
    "discord_webhook_url": ""   // Optional: Override default Discord webhook
}
```

### Response

```json
{
    "status": "success",
    "symbol": "SOFI",
    "sentiment_score": 8.5,
    "summary": "Bullish sentiment with heightened chart breakout discussions...",
    "raw_response": "...",
    "citations_count": 45,
    "citations_sample": ["https://x.com/...", "..."],
    "tool_usage": {"SERVER_SIDE_TOOL_X_SEARCH": 2},
    "api_call_duration": 12.5,
    "model_used": "grok-4-1-fast",
    "hours_back": 24,
    "timestamp": "2026-01-05T14:30:00-05:00"
}
```

### Example cURL

```bash
curl -X POST https://groksentiment-1039781888202.us-central1.run.app \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SOFI", "hours_back": 24}'
```

### Example Python

```python
import requests

response = requests.post(
    "https://groksentiment-1039781888202.us-central1.run.app",
    json={"symbol": "SOFI", "hours_back": 24}
)
result = response.json()
print(f"Sentiment: {result['sentiment_score']}/10")
print(f"Summary: {result['summary']}")
```

## Getting started with VS Code

### Before you begin

1. If you're new to Google Cloud, [create an account](https://console.cloud.google.com/freetrial/signup/tos) to evaluate how our products perform in real-world scenarios.

2. Make sure that billing is enabled for your Cloud project.

3. Enable the following APIs:
   - Cloud Functions
   - Cloud Build
   - Artifact Registry
   - Cloud Run
   - Logging

4. Install the [Cloud Code plugin](https://cloud.google.com/code/docs/vscode/install).

## Local Development

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set environment variables

```bash
export XAI_API_KEY="your_xai_api_key_here"
export DISCORD_WEBHOOK_URL="your_discord_webhook_url"  # Optional
```

### Run locally with functions-framework

```bash
functions-framework --target=grok_sentiment --host=localhost --port=8080
```

### Test locally

```bash
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SOFI"}'
```

Or run the test directly:

```bash
python main.py SOFI
```

## Deployment

### Option 1: Cloud Build (Recommended)

```bash
gcloud builds submit --config cloudbuild.yaml .
```

### Option 2: Direct Deploy

```bash
gcloud run deploy groksentiment \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 512Mi \
  --timeout 300s \
  --set-env-vars "XAI_API_KEY=your_key_here"
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `XAI_API_KEY` | Yes | Your xAI API key for Grok access |
| `DISCORD_WEBHOOK_URL` | No | Default Discord webhook for notifications |

### Setting Secrets in Cloud Run

For production, use Secret Manager:

```bash
# Create secret
echo -n "your_xai_api_key" | gcloud secrets create xai-api-key --data-file=-

# Grant access to Cloud Run service account
gcloud secrets add-iam-policy-binding xai-api-key \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Deploy with secret
gcloud run deploy groksentiment \
  --source . \
  --region us-central1 \
  --set-secrets="XAI_API_KEY=xai-api-key:latest"
```

## Cost Optimization

This function is optimized for cost efficiency:

| Feature | Setting | Benefit |
|---------|---------|---------|
| Model | `grok-4-1-fast` | Faster, cheaper than grok-4 |
| Tools | `x_search()` only | No web_search reduces API calls |
| `max_turns` | 2 (default) | Limits tool iterations |
| Date range | Configurable `hours_back` | Reduces search data volume |
| Streaming | Disabled | Simpler, lower overhead |

## Clean up

To delete the Cloud Run service:

```bash
gcloud run services delete groksentiment --region us-central1
```

## License

MIT License
