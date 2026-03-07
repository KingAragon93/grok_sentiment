# AGENT_RETRIEVAL_MAP

Purpose: define retrieval policy and compact read paths for autonomous coding agents.

## Retrieval policy

### Include first (high confidence)
- [main.py](main.py) — runtime request handlers and decision logic.
- [Dockerfile](Dockerfile) — deployed function target (`functions-framework --target=...`).
- [cloudbuild.yaml](cloudbuild.yaml) — deploy path and remote runtime rollout.

### Include second (documentation)
- [README.md](README.md)
- [AGENT_START_HERE.md](AGENT_START_HERE.md)
- [AGENT_SYMBOL_INDEX.md](AGENT_SYMBOL_INDEX.md)
- [MIGRATION_INSTRUCTIONS.txt](MIGRATION_INSTRUCTIONS.txt)

### Include with caution
- [sentiment_analyzer.py](sentiment_analyzer.py) — analysis-focused; avoid treating as runtime source of truth.
- [analyze_recommendations.py](analyze_recommendations.py) — reporting utility, not request-time decision path.
- [stock_data_access_example.py](stock_data_access_example.py) — reference utility.

### Exclude by default
- `.git/`, `__pycache__/`, `.venv/`, `venv/`, `node_modules/`, `.mypy_cache/`, `.pytest_cache/`
- Generated or scratch: `_*.py`, `_*.txt`, `*.log`, local temp exports

## Hotspot tags
- `HOTSPOT:RECOMMENDATION_DECISION`
- `HOTSPOT:PARSING_FALLBACK`
- `HOTSPOT:DEPLOY_ENTRYPOINT`
- `HOTSPOT:REMOTE_LOG_SCHEMA`

## Task-first retrieval routes

### A) Recommendation behavior mismatch
1. [main.py](main.py)
2. [Dockerfile](Dockerfile)
3. [cloudbuild.yaml](cloudbuild.yaml)
4. [README.md](README.md)

### B) Logging/metrics mismatch
1. [gcs_logger.py](gcs_logger.py)
2. [analyze_recommendations.py](analyze_recommendations.py)
3. [sentiment_analyzer.py](sentiment_analyzer.py)

### C) Deployment/runtime drift
1. [Dockerfile](Dockerfile)
2. [cloudbuild.yaml](cloudbuild.yaml)
3. [main.py](main.py)

## Source-of-truth config rule
- Local authoritative config (if present): `.env`
- Runtime remote config source: Cloud Run env/secrets + Cloud Build substitutions
- Required step before behavior analysis: pull latest remote runtime config and deployed revision details
