# AGENT_SYMBOL_INDEX

Auto-generated top-level symbol index for retrieval-efficient code navigation.

Generated (UTC): 2026-03-07 04:47:15Z
Symbol cap per file: 20

## primary-runtime

### gcs_logger.py
- `class` `AlignedRecommendationLog` (L31)
- `class` `GCSAlignedLogger` (L55)
- `def` `init_aligned_log_file` (L206)

### main.py
- `def` `get_et_timezone` (L71)
- `def` `now_et` (L76)
- `def` `send_discord_message` (L81)
- `def` `analyze_sentiment` (L105)
- `def` `get_stock_recommendation` (L240)
- `def` `get_aligned_recommendation` (L384)
- `def` `format_discord_embed` (L561)
- `def` `grok_sentiment` (L620)
- `def` `grok_recommendation` (L691)
- `def` `grok_aligned_recommendation` (L759)
- `def` `format_aligned_embed` (L849)
- `def` `format_recommendation_embed` (L925)

## analysis

### analyze_recommendations.py
- `class` `AnalysisMetrics` (L51)
- `class` `RecommendationAnalyzer` (L71)
- `def` `main` (L482)

### sentiment_analyzer.py
- `class` `SentimentRecord` (L57)
- `class` `PricePoint` (L106)
- `class` `PerformanceRecord` (L119)
- `class` `StockPriceTracker` (L144)
- `class` `GrokHistoryReader` (L244)
- `class` `SentimentPerformanceAnalyzer` (L353)
- `def` `print_metrics_report` (L823)

## reference

### stock_data_access_example.py
- `class` `StockDataProvider` (L26)

## tooling

### scripts/generate_agent_symbol_index.py
- `def` `classify_file` (L26)
- `def` `iter_python_files` (L39)
- `def` `top_level_symbols` (L52)
- `def` `build_markdown` (L68)
- `def` `main` (L105)

### scripts/refresh_agent_docs.py
- `def` `run_step` (L13)
- `def` `main` (L19)

### scripts/validate_doc_references.py
- `def` `check_links` (L28)
- `def` `check_secrets` (L46)
- `def` `validate_required_policy_text` (L54)
- `def` `main` (L72)
