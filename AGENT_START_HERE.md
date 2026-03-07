# AGENT_START_HERE

Purpose: fast onboarding for long-running code agents working in this repo.

## 1) Primary mission
- Runtime domain: stock sentiment + aligned buy/hold/sell recommendation via Cloud Run functions.
- Primary behavior entrypoint: `grok_aligned_recommendation` in [main.py](main.py).

## 2) Deterministic read order (task-oriented)
Read in this order unless task explicitly requires otherwise:

1. Request/response behavior
   - [main.py](main.py)
2. Logging + historical analysis surfaces
   - [gcs_logger.py](gcs_logger.py)
   - [analyze_recommendations.py](analyze_recommendations.py)
   - [sentiment_analyzer.py](sentiment_analyzer.py)
3. Runtime/deploy configuration
   - [Dockerfile](Dockerfile)
   - [cloudbuild.yaml](cloudbuild.yaml)
   - [requirements.txt](requirements.txt)
4. Operational guidance
   - [README.md](README.md)
   - [AGENT_RETRIEVAL_MAP.md](AGENT_RETRIEVAL_MAP.md)
   - [AGENT_SYMBOL_INDEX.md](AGENT_SYMBOL_INDEX.md)

## 3) High-risk hotspot tags
- `HOTSPOT:RECOMMENDATION_DECISION` — buy/hold/sell determination path in [main.py](main.py)
- `HOTSPOT:PARSING_FALLBACK` — JSON parse fallback/default handling in [main.py](main.py)
- `HOTSPOT:DEPLOY_ENTRYPOINT` — functions-framework target in [Dockerfile](Dockerfile)
- `HOTSPOT:REMOTE_LOG_SCHEMA` — GCS log schema writes/reads in [gcs_logger.py](gcs_logger.py) and [analyze_recommendations.py](analyze_recommendations.py)

## 4) Source-of-truth config rule
- Local authoritative config (dev): `.env` (never commit secrets).
- Runtime remote config (prod): Cloud Run environment + Cloud Build substitutions/secrets.
- Before behavior analysis, always confirm latest remote runtime config and deployed entrypoint.

## 5) Scratch-file rule
- Allowed temporary local artifacts: `_*.py`, `_*.txt`.
- Expectation: delete before commit or add to `.gitignore` patterns (already included).

## 6) Agent maintenance commands
Run from repo root:

```bash
python scripts/validate_doc_references.py
python scripts/generate_agent_symbol_index.py
python scripts/refresh_agent_docs.py
```
