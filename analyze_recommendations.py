#!/usr/bin/env python3
"""
Aligned Recommendation Analyzer

Single file to run analysis on:
1. Historical sentiment predictions from GCS
2. Today's aligned recommendation calls
3. Metrics on recommended buys (accuracy, returns, timeframes)

Usage:
    python analyze_recommendations.py              # Full analysis
    python analyze_recommendations.py --today      # Today's calls only
    python analyze_recommendations.py --buys       # Buy metrics only
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
import logging

from dotenv import load_dotenv
from google.cloud import storage

# Try to import price tracking
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "historical_stock_day")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "src/GoogleCloudServiceAccount.json")
HISTORICAL_LOG = "grok_interactions_log.json"
ALIGNED_LOG = "aligned_recommendations_log.json"


@dataclass
class AnalysisMetrics:
    """Container for analysis metrics."""
    total_records: int = 0
    buy_signals: int = 0
    hold_signals: int = 0
    sell_signals: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    avg_sentiment_buy: float = 0.0
    avg_sentiment_hold: float = 0.0
    profitable_buys: int = 0
    loss_buys: int = 0
    pending_buys: int = 0
    avg_return_4h: float = 0.0
    avg_return_12h: float = 0.0
    avg_return_24h: float = 0.0
    avg_return_36h: float = 0.0
    accuracy_rate: float = 0.0


class RecommendationAnalyzer:
    """Analyzes aligned recommendations and historical data."""
    
    def __init__(self):
        self.client = None
        self.alpaca_client = None
        self._init_gcs()
        self._init_alpaca()
    
    def _init_gcs(self):
        """Initialize GCS client."""
        try:
            if os.path.exists(CREDENTIALS_PATH):
                self.client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
            else:
                self.client = storage.Client()
            logger.info("✅ GCS client initialized")
        except Exception as e:
            logger.error(f"❌ GCS init failed: {e}")
    
    def _init_alpaca(self):
        """Initialize Alpaca client for price data."""
        if not ALPACA_AVAILABLE:
            logger.warning("⚠️ Alpaca not available - price tracking disabled")
            return
        
        try:
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            if api_key and secret_key:
                self.alpaca_client = StockHistoricalDataClient(api_key, secret_key)
                logger.info("✅ Alpaca client initialized")
        except Exception as e:
            logger.warning(f"⚠️ Alpaca init failed: {e}")
    
    def load_historical_logs(self) -> List[Dict[str, Any]]:
        """Load historical sentiment logs from GCS and normalize format."""
        try:
            bucket = self.client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(HISTORICAL_LOG)
            if blob.exists():
                content = blob.download_as_text()
                raw_data = json.loads(content)
                logger.info(f"📥 Loaded {len(raw_data)} historical records")
                
                # Normalize to common format
                normalized = []
                for record in raw_data:
                    # Handle nested GCS format: {"ticker": "X", "response": {...}, ...}
                    if 'ticker' in record and 'response' in record:
                        response = record.get('response', {})
                        normalized.append({
                            "symbol": record.get('ticker', ''),
                            "sentiment_score": float(response.get('parsed_sentiment_score', 0) or 0),
                            "recommendation": record.get('recommendation', 'hold'),
                            "buy_signal": record.get('buy_signal', False),
                            "timestamp": record.get('timestamp', ''),
                            "summary": response.get('parsed_summary', ''),
                            "citations_count": record.get('citations_count', 0),
                        })
                    # Handle direct format
                    else:
                        normalized.append({
                            "symbol": record.get('symbol', ''),
                            "sentiment_score": float(record.get('sentiment_score', 0) or 0),
                            "recommendation": record.get('recommendation', 'hold'),
                            "buy_signal": record.get('buy_signal', False),
                            "timestamp": record.get('timestamp', ''),
                            "summary": record.get('summary', ''),
                            "citations_count": record.get('citations_count', 0),
                        })
                
                return normalized
            return []
        except Exception as e:
            logger.error(f"❌ Failed to load historical logs: {e}")
            return []
    
    def load_aligned_logs(self) -> List[Dict[str, Any]]:
        """Load aligned recommendation logs from GCS."""
        try:
            bucket = self.client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(ALIGNED_LOG)
            if blob.exists():
                content = blob.download_as_text()
                data = json.loads(content)
                logger.info(f"📥 Loaded {len(data)} aligned records")
                return data
            logger.info("📄 No aligned logs yet")
            return []
        except Exception as e:
            logger.error(f"❌ Failed to load aligned logs: {e}")
            return []
    
    def get_today_str(self) -> str:
        """Get today's date string."""
        return datetime.now().strftime("%Y-%m-%d")
    
    def filter_by_date(self, logs: List[Dict], date_str: str) -> List[Dict]:
        """Filter logs by date."""
        return [
            log for log in logs
            if log.get("timestamp", "").startswith(date_str)
        ]
    
    def filter_buys(self, logs: List[Dict]) -> List[Dict]:
        """Filter to buy recommendations only."""
        return [
            log for log in logs
            if log.get("recommendation") == "buy" or log.get("buy_signal") == True
        ]
    
    def calculate_metrics(self, logs: List[Dict]) -> AnalysisMetrics:
        """Calculate comprehensive metrics from logs."""
        metrics = AnalysisMetrics()
        metrics.total_records = len(logs)
        
        if not logs:
            return metrics
        
        sentiment_buy = []
        sentiment_hold = []
        returns_4h = []
        returns_12h = []
        returns_24h = []
        returns_36h = []
        
        for log in logs:
            rec = log.get("recommendation", "hold")
            if rec == "buy":
                metrics.buy_signals += 1
                sentiment_buy.append(log.get("sentiment_score", 0))
            elif rec == "sell":
                metrics.sell_signals += 1
            else:
                metrics.hold_signals += 1
                sentiment_hold.append(log.get("sentiment_score", 0))
            
            # Confidence
            conf = log.get("confidence", "medium")
            if conf == "high":
                metrics.high_confidence += 1
            else:
                metrics.medium_confidence += 1
            
            # Outcomes
            outcome = log.get("outcome")
            if log.get("buy_signal"):
                if outcome == "profitable":
                    metrics.profitable_buys += 1
                elif outcome == "loss":
                    metrics.loss_buys += 1
                else:
                    metrics.pending_buys += 1
            
            # Returns
            if log.get("return_4h") is not None:
                returns_4h.append(log["return_4h"])
            if log.get("return_12h") is not None:
                returns_12h.append(log["return_12h"])
            if log.get("return_24h") is not None:
                returns_24h.append(log["return_24h"])
            if log.get("return_36h") is not None:
                returns_36h.append(log["return_36h"])
        
        # Averages
        if sentiment_buy:
            metrics.avg_sentiment_buy = sum(sentiment_buy) / len(sentiment_buy)
        if sentiment_hold:
            metrics.avg_sentiment_hold = sum(sentiment_hold) / len(sentiment_hold)
        if returns_4h:
            metrics.avg_return_4h = sum(returns_4h) / len(returns_4h)
        if returns_12h:
            metrics.avg_return_12h = sum(returns_12h) / len(returns_12h)
        if returns_24h:
            metrics.avg_return_24h = sum(returns_24h) / len(returns_24h)
        if returns_36h:
            metrics.avg_return_36h = sum(returns_36h) / len(returns_36h)
        
        # Accuracy
        if metrics.profitable_buys + metrics.loss_buys > 0:
            metrics.accuracy_rate = metrics.profitable_buys / (metrics.profitable_buys + metrics.loss_buys) * 100
        
        return metrics
    
    def analyze_historical_by_score(self, logs: List[Dict]) -> Dict[str, Dict]:
        """Analyze historical performance by sentiment score ranges."""
        ranges = {
            "9+": {"min": 9.0, "max": 10.0, "records": [], "returns": []},
            "8-9": {"min": 8.0, "max": 9.0, "records": [], "returns": []},
            "7-8": {"min": 7.0, "max": 8.0, "records": [], "returns": []},
            "5-7": {"min": 5.0, "max": 7.0, "records": [], "returns": []},
            "2-5": {"min": 2.0, "max": 5.0, "records": [], "returns": []},
            "<2": {"min": -10.0, "max": 2.0, "records": [], "returns": []},
        }
        
        for log in logs:
            score = log.get("sentiment_score", 0)
            ret = log.get("return_24h")
            
            for range_name, range_data in ranges.items():
                if range_data["min"] <= score < range_data["max"] or (range_name == "9+" and score >= 9.0):
                    range_data["records"].append(log)
                    if ret is not None:
                        range_data["returns"].append(ret)
                    break
        
        # Calculate stats per range
        results = {}
        for range_name, range_data in ranges.items():
            count = len(range_data["records"])
            returns = range_data["returns"]
            results[range_name] = {
                "count": count,
                "avg_return": sum(returns) / len(returns) if returns else None,
                "positive_rate": sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else None,
                "returns_tracked": len(returns)
            }
        
        return results
    
    def print_header(self, title: str):
        """Print section header."""
        print("\n" + "=" * 70)
        print(f" {title}")
        print("=" * 70)
    
    def print_metrics(self, metrics: AnalysisMetrics, title: str = "METRICS"):
        """Print formatted metrics."""
        self.print_header(title)
        print(f"""
  Total Records:        {metrics.total_records}
  
  SIGNALS:
    Buy Signals:        {metrics.buy_signals} ({metrics.buy_signals/max(metrics.total_records,1)*100:.1f}%)
    Hold Signals:       {metrics.hold_signals} ({metrics.hold_signals/max(metrics.total_records,1)*100:.1f}%)
    Sell Signals:       {metrics.sell_signals} ({metrics.sell_signals/max(metrics.total_records,1)*100:.1f}%)
  
  CONFIDENCE:
    High Confidence:    {metrics.high_confidence}
    Medium Confidence:  {metrics.medium_confidence}
  
  SENTIMENT AVERAGES:
    Avg Score (Buys):   {metrics.avg_sentiment_buy:.2f}
    Avg Score (Holds):  {metrics.avg_sentiment_hold:.2f}
  
  BUY OUTCOMES:
    Profitable:         {metrics.profitable_buys}
    Loss:               {metrics.loss_buys}
    Pending:            {metrics.pending_buys}
    Accuracy Rate:      {metrics.accuracy_rate:.1f}%
  
  AVERAGE RETURNS (where tracked):
    4h Return:          {metrics.avg_return_4h:+.2f}%
    12h Return:         {metrics.avg_return_12h:+.2f}%
    24h Return:         {metrics.avg_return_24h:+.2f}%
    36h Return:         {metrics.avg_return_36h:+.2f}%
""")
    
    def print_score_analysis(self, results: Dict[str, Dict]):
        """Print score range analysis."""
        self.print_header("PERFORMANCE BY SENTIMENT SCORE RANGE")
        print(f"\n  {'Range':<10} {'Count':<10} {'Avg Return':<15} {'Win Rate':<12} {'Tracked'}")
        print("  " + "-" * 60)
        for range_name, data in results.items():
            avg_ret = f"{data['avg_return']:+.2f}%" if data['avg_return'] is not None else "N/A"
            win_rate = f"{data['positive_rate']:.1f}%" if data['positive_rate'] is not None else "N/A"
            print(f"  {range_name:<10} {data['count']:<10} {avg_ret:<15} {win_rate:<12} {data['returns_tracked']}")
    
    def run_full_analysis(self):
        """Run complete analysis on all data."""
        self.print_header("ALIGNED RECOMMENDATION ANALYSIS")
        print(f"  Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  GCS Bucket: {GCS_BUCKET_NAME}")
        
        # Load data
        historical = self.load_historical_logs()
        aligned = self.load_aligned_logs()
        
        # Historical metrics
        if historical:
            hist_metrics = self.calculate_metrics(historical)
            self.print_metrics(hist_metrics, "HISTORICAL SENTIMENT DATA (grok_interactions_log.json)")
            
            # Score range analysis
            score_analysis = self.analyze_historical_by_score(historical)
            self.print_score_analysis(score_analysis)
        
        # Aligned metrics
        if aligned:
            aligned_metrics = self.calculate_metrics(aligned)
            self.print_metrics(aligned_metrics, "ALIGNED RECOMMENDATIONS (aligned_recommendations_log.json)")
        else:
            self.print_header("ALIGNED RECOMMENDATIONS")
            print("\n  No aligned recommendations logged yet.")
            print("  Run get_aligned_recommendation() to start logging.\n")
        
        # Today's data
        today = self.get_today_str()
        today_hist = self.filter_by_date(historical, today)
        today_aligned = self.filter_by_date(aligned, today)
        
        self.print_header(f"TODAY'S ACTIVITY ({today})")
        print(f"""
  Historical sentiment calls today:  {len(today_hist)}
  Aligned recommendation calls today: {len(today_aligned)}
""")
        
        if today_hist:
            print("  Today's Historical Calls:")
            for log in today_hist[:10]:  # Show first 10
                score = log.get("sentiment_score", 0)
                symbol = log.get("symbol", "?")
                ts = log.get("timestamp", "")[:19]
                print(f"    {ts} | {symbol:<6} | Score: {score:+.1f}")
        
        if today_aligned:
            print("\n  Today's Aligned Recommendations:")
            for log in today_aligned:
                score = log.get("sentiment_score", 0)
                symbol = log.get("symbol", "?")
                rec = log.get("recommendation", "?")
                conf = log.get("confidence", "?")
                ts = log.get("timestamp", "")[:19]
                print(f"    {ts} | {symbol:<6} | Score: {score:+.1f} | {rec.upper():<5} | {conf}")
    
    def run_buys_analysis(self):
        """Analyze buy recommendations specifically."""
        self.print_header("BUY RECOMMENDATION ANALYSIS")
        
        historical = self.load_historical_logs()
        aligned = self.load_aligned_logs()
        
        # Get all buys from historical
        hist_buys = [
            log for log in historical
            if log.get("sentiment_score", 0) >= 7.5  # Would have been a buy
        ]
        
        aligned_buys = self.filter_buys(aligned)
        
        print(f"""
  Historical High-Sentiment Records (7.5+): {len(hist_buys)}
  Aligned Buy Recommendations:              {len(aligned_buys)}
""")
        
        if hist_buys:
            # Group by sentiment range
            score_9_plus = [l for l in hist_buys if l.get("sentiment_score", 0) >= 9.0]
            score_8_to_9 = [l for l in hist_buys if 8.0 <= l.get("sentiment_score", 0) < 9.0]
            score_7_to_8 = [l for l in hist_buys if 7.5 <= l.get("sentiment_score", 0) < 8.0]
            
            print("  Historical Breakdown:")
            print(f"    Score 9+:    {len(score_9_plus)} records")
            print(f"    Score 8-9:   {len(score_8_to_9)} records")
            print(f"    Score 7.5-8: {len(score_7_to_8)} records")
            
            # Show top performers
            if score_9_plus:
                print("\n  Top 9+ Sentiment Records:")
                sorted_records = sorted(score_9_plus, key=lambda x: x.get("sentiment_score", 0), reverse=True)
                for log in sorted_records[:5]:
                    symbol = log.get("symbol", "?")
                    score = log.get("sentiment_score", 0)
                    ts = log.get("timestamp", "")[:10]
                    ret = log.get("return_24h")
                    ret_str = f"{ret:+.2f}%" if ret is not None else "pending"
                    print(f"      {ts} | {symbol:<6} | {score:.1f} | 24h: {ret_str}")
    
    def run_today_analysis(self):
        """Show only today's activity."""
        today = self.get_today_str()
        self.print_header(f"TODAY'S ANALYSIS ({today})")
        
        historical = self.load_historical_logs()
        aligned = self.load_aligned_logs()
        
        today_hist = self.filter_by_date(historical, today)
        today_aligned = self.filter_by_date(aligned, today)
        
        print(f"\n  Historical calls: {len(today_hist)}")
        print(f"  Aligned calls:    {len(today_aligned)}")
        
        if today_hist:
            print("\n  Historical Sentiment Calls:")
            print(f"  {'Time':<12} {'Symbol':<8} {'Score':<8} {'Summary'}")
            print("  " + "-" * 60)
            for log in today_hist:
                ts = log.get("timestamp", "")[11:19]
                symbol = log.get("symbol", "?")
                score = log.get("sentiment_score", 0)
                summary = (log.get("summary", "") or "")[:40]
                print(f"  {ts:<12} {symbol:<8} {score:+.1f}    {summary}...")
        
        if today_aligned:
            print("\n  Aligned Recommendations:")
            print(f"  {'Time':<12} {'Symbol':<8} {'Score':<8} {'Rec':<6} {'Conf':<8} {'Reason'}")
            print("  " + "-" * 70)
            for log in today_aligned:
                ts = log.get("timestamp", "")[11:19]
                symbol = log.get("symbol", "?")
                score = log.get("sentiment_score", 0)
                rec = log.get("recommendation", "?")
                conf = log.get("confidence", "?")
                reason = (log.get("alignment_reason", "") or "")[:30]
                print(f"  {ts:<12} {symbol:<8} {score:+.1f}    {rec:<6} {conf:<8} {reason}...")
        
        if not today_hist and not today_aligned:
            print("\n  No activity recorded today.")


def main():
    parser = argparse.ArgumentParser(description="Analyze aligned recommendations")
    parser.add_argument("--today", action="store_true", help="Show today's activity only")
    parser.add_argument("--buys", action="store_true", help="Show buy metrics only")
    args = parser.parse_args()
    
    analyzer = RecommendationAnalyzer()
    
    if args.today:
        analyzer.run_today_analysis()
    elif args.buys:
        analyzer.run_buys_analysis()
    else:
        analyzer.run_full_analysis()
    
    print("\n" + "=" * 70)
    print(" Analysis complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
