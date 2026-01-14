"""
Grok Sentiment Performance Analyzer

This module analyzes the historical performance of Grok sentiment scores
by tracking stock price movements at various timeframes after predictions.

Features:
- Reads historical Grok interactions from Google Cloud Storage
- Fetches stock prices at 4, 12, 24, 36 hour intervals after prediction
- Calculates accuracy metrics for sentiment scoring
- Identifies discrepancies between sentiment scores and recommendations
- Provides data for model training and improvement

Usage:
    python sentiment_analyzer.py [--symbol SYMBOL] [--days DAYS] [--analyze-all]
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import requests
from google.cloud import storage
import pytz

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Keys from .env
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
POLYGON_API_KEY = os.getenv('POLYGON')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'your-bucket-name')

# Timeframes to analyze (in hours)
ANALYSIS_TIMEFRAMES = [4, 12, 24, 36]

# Sentiment score thresholds
HIGH_SENTIMENT_THRESHOLD = 7.0  # Score >= 7 should indicate strong bullish
MID_SENTIMENT_THRESHOLD = 4.0   # Score >= 4 is moderately bullish
LOW_SENTIMENT_THRESHOLD = -4.0  # Score <= -4 is bearish


@dataclass
class SentimentRecord:
    """Represents a single Grok sentiment analysis record."""
    symbol: str
    sentiment_score: float
    recommendation: Optional[str]
    buy_signal: Optional[bool]
    timestamp: str
    summary: str
    citations_count: int = 0
    processor: Optional[str] = None  # e.g., 'process_1H', 'process_4H'
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SentimentRecord':
        """
        Parse a record from the GCS log format.
        
        Handles both formats:
        1. Direct format: {"symbol": "X", "sentiment_score": 8.5, ...}
        2. Nested format: {"ticker": "X", "response": {"parsed_sentiment_score": 8.5, ...}, ...}
        """
        # Handle nested GCS log format
        if 'ticker' in data and 'response' in data:
            response = data.get('response', {})
            metadata = data.get('metadata', {})
            return cls(
                symbol=data.get('ticker', ''),
                sentiment_score=float(response.get('parsed_sentiment_score', 0) or 0),
                recommendation=data.get('recommendation'),  # May not exist in old format
                buy_signal=data.get('buy_signal'),  # May not exist in old format
                timestamp=data.get('timestamp', ''),
                summary=response.get('parsed_summary', '') or '',
                citations_count=data.get('citations_count', 0),
                processor=metadata.get('processor')
            )
        
        # Handle direct format (from main.py output)
        return cls(
            symbol=data.get('symbol', ''),
            sentiment_score=float(data.get('sentiment_score', 0)),
            recommendation=data.get('recommendation'),
            buy_signal=data.get('buy_signal'),
            timestamp=data.get('timestamp', ''),
            summary=data.get('summary', ''),
            citations_count=data.get('citations_count', 0),
            processor=data.get('processor')
        )


@dataclass
class PricePoint:
    """Represents a stock price at a specific time."""
    symbol: str
    timestamp: str
    price: float
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class PerformanceRecord:
    """Represents the performance analysis of a sentiment prediction."""
    symbol: str
    sentiment_score: float
    recommendation: Optional[str]
    buy_signal: Optional[bool]
    prediction_time: str
    initial_price: float
    price_4h: Optional[float]
    price_12h: Optional[float]
    price_24h: Optional[float]
    price_36h: Optional[float]
    return_4h: Optional[float]
    return_12h: Optional[float]
    return_24h: Optional[float]
    return_36h: Optional[float]
    score_accuracy_4h: Optional[str]  # 'correct', 'incorrect', 'neutral'
    score_accuracy_12h: Optional[str]
    score_accuracy_24h: Optional[str]
    score_accuracy_36h: Optional[str]
    recommendation_match: Optional[str]  # 'aligned', 'misaligned', 'n/a'
    processor: Optional[str] = None  # e.g., 'process_1H', 'process_4H'
    summary: Optional[str] = None


class StockPriceTracker:
    """Tracks stock prices at various timeframes using Alpaca API."""
    
    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://data.alpaca.markets/v2/stocks"
        self.headers = {
            'APCA-API-KEY-ID': ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
        }
    
    def get_price_at_time(self, symbol: str, target_time: datetime) -> Optional[PricePoint]:
        """
        Get the stock price at or near a specific time.
        Uses 1-hour bars to find the closest price.
        """
        try:
            # Convert to UTC if not already
            if target_time.tzinfo is None:
                target_time = pytz.UTC.localize(target_time)
            else:
                target_time = target_time.astimezone(pytz.UTC)
            
            # Get bars around the target time
            start = target_time - timedelta(hours=2)
            end = target_time + timedelta(hours=2)
            
            url = f"{self.base_url}/{symbol}/bars"
            params = {
                'timeframe': '1Hour',
                'start': start.isoformat(),
                'end': end.isoformat(),
                'adjustment': 'raw'
            }
            
            response = self.session.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('bars'):
                logger.warning(f"No bars found for {symbol} around {target_time}")
                return None
            
            # Find the bar closest to target time
            bars = data['bars']
            closest_bar = min(bars, key=lambda b: abs(
                datetime.fromisoformat(b['t'].replace('Z', '+00:00')) - target_time
            ))
            
            return PricePoint(
                symbol=symbol,
                timestamp=closest_bar['t'],
                price=closest_bar['c'],  # Use close price
                open=closest_bar['o'],
                high=closest_bar['h'],
                low=closest_bar['l'],
                close=closest_bar['c'],
                volume=closest_bar['v']
            )
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol} at {target_time}: {e}")
            return None
    
    def get_prices_at_timeframes(
        self, 
        symbol: str, 
        base_time: datetime, 
        timeframes_hours: List[int] = ANALYSIS_TIMEFRAMES
    ) -> Dict[int, Optional[PricePoint]]:
        """
        Get prices at multiple timeframes after the base time.
        
        Args:
            symbol: Stock ticker
            base_time: The time of the sentiment prediction
            timeframes_hours: List of hours after base_time to check
            
        Returns:
            Dict mapping hours -> PricePoint
        """
        prices = {}
        
        # Get initial price (at prediction time)
        prices[0] = self.get_price_at_time(symbol, base_time)
        
        # Get prices at each timeframe
        for hours in timeframes_hours:
            target_time = base_time + timedelta(hours=hours)
            
            # Don't fetch future prices
            if target_time > datetime.now(pytz.UTC):
                prices[hours] = None
                logger.info(f"Skipping {symbol} at +{hours}h (future)")
            else:
                prices[hours] = self.get_price_at_time(symbol, target_time)
        
        return prices


class GrokHistoryReader:
    """Reads historical Grok interactions from Google Cloud Storage."""
    
    def __init__(self, bucket_name: str = GCS_BUCKET_NAME):
        self.bucket_name = bucket_name
        try:
            # Use service account credentials from environment variable
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path:
                self.client = storage.Client.from_service_account_json(credentials_path)
                logger.info(f"✅ Using service account credentials from {credentials_path}")
            else:
                self.client = storage.Client()
                logger.info("✅ Using default GCP credentials")
            
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"✅ Connected to GCS bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to GCS: {e}")
            self.client = None
            self.bucket = None
    
    def read_interactions_log(
        self, 
        blob_path: str = "grok_interactions_log.json"
    ) -> List[SentimentRecord]:
        """
        Read the Grok interactions log from GCS.
        
        Args:
            blob_path: Path to the log file in the bucket (default: grok_interactions_log.json)
            
        Returns:
            List of SentimentRecord objects
        """
        if not self.bucket:
            logger.error("GCS bucket not available")
            return []
        
        try:
            blob = self.bucket.blob(blob_path)
            content = blob.download_as_text()
            
            # Handle both JSON array and newline-delimited JSON
            records = []
            try:
                # Try parsing as JSON array first
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        try:
                            records.append(SentimentRecord.from_dict(item))
                        except Exception as e:
                            logger.warning(f"Failed to parse record: {e}")
                else:
                    records.append(SentimentRecord.from_dict(data))
            except json.JSONDecodeError:
                # Try newline-delimited JSON
                for line in content.strip().split('\n'):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            records.append(SentimentRecord.from_dict(item))
                        except Exception as e:
                            logger.warning(f"Failed to parse line: {e}")
            
            logger.info(f"✅ Loaded {len(records)} sentiment records from GCS")
            return records
            
        except Exception as e:
            logger.error(f"❌ Failed to read interactions log: {e}")
            return []
    
    def read_local_log(self, file_path: str) -> List[SentimentRecord]:
        """
        Read a local interactions log file (for testing).
        
        Args:
            file_path: Path to local JSON file
            
        Returns:
            List of SentimentRecord objects
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            records = []
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        records.append(SentimentRecord.from_dict(item))
                else:
                    records.append(SentimentRecord.from_dict(data))
            except json.JSONDecodeError:
                for line in content.strip().split('\n'):
                    if line.strip():
                        item = json.loads(line)
                        records.append(SentimentRecord.from_dict(item))
            
            logger.info(f"✅ Loaded {len(records)} records from local file")
            return records
            
        except Exception as e:
            logger.error(f"❌ Failed to read local file: {e}")
            return []


class SentimentPerformanceAnalyzer:
    """
    Analyzes the performance of Grok sentiment predictions.
    
    Tracks price movements after predictions and calculates accuracy metrics.
    """
    
    def __init__(self):
        self.price_tracker = StockPriceTracker()
        self.history_reader = GrokHistoryReader()
        self.performance_records: List[PerformanceRecord] = []
    
    def calculate_accuracy(
        self, 
        sentiment_score: float, 
        price_return: float
    ) -> str:
        """
        Determine if the sentiment score correctly predicted price direction.
        
        Args:
            sentiment_score: The Grok sentiment score (-10 to +10)
            price_return: The percentage return of the stock
            
        Returns:
            'correct', 'incorrect', or 'neutral'
        """
        # High positive sentiment should predict positive returns
        if sentiment_score >= HIGH_SENTIMENT_THRESHOLD:
            if price_return > 0.5:  # >0.5% gain
                return 'correct'
            elif price_return < -0.5:  # >0.5% loss
                return 'incorrect'
            else:
                return 'neutral'
        
        # Moderate positive sentiment
        elif sentiment_score >= MID_SENTIMENT_THRESHOLD:
            if price_return > 0:
                return 'correct'
            elif price_return < -0.5:
                return 'incorrect'
            else:
                return 'neutral'
        
        # Negative sentiment should predict negative returns
        elif sentiment_score <= LOW_SENTIMENT_THRESHOLD:
            if price_return < -0.5:
                return 'correct'
            elif price_return > 0.5:
                return 'incorrect'
            else:
                return 'neutral'
        
        # Neutral sentiment
        else:
            return 'neutral'
    
    def check_recommendation_alignment(
        self, 
        sentiment_score: float, 
        recommendation: Optional[str],
        buy_signal: Optional[bool]
    ) -> str:
        """
        Check if sentiment score aligns with the recommendation.
        
        This identifies the discrepancy issue you mentioned where high
        sentiment scores (8.5, 9.2) don't result in 'buy' recommendations.
        
        Args:
            sentiment_score: The Grok sentiment score
            recommendation: 'buy', 'hold', or 'sell'
            buy_signal: Boolean buy signal
            
        Returns:
            'aligned', 'misaligned', or 'n/a'
        """
        if recommendation is None:
            return 'n/a'
        
        recommendation = recommendation.lower()
        
        # High sentiment (>=7) should be 'buy'
        if sentiment_score >= HIGH_SENTIMENT_THRESHOLD:
            if recommendation == 'buy':
                return 'aligned'
            else:
                return 'misaligned'
        
        # Moderate positive sentiment (4-7) should be 'buy' or 'hold'
        elif sentiment_score >= MID_SENTIMENT_THRESHOLD:
            if recommendation in ['buy', 'hold']:
                return 'aligned'
            else:
                return 'misaligned'
        
        # Low sentiment (<=-4) should be 'sell' or 'hold'
        elif sentiment_score <= LOW_SENTIMENT_THRESHOLD:
            if recommendation in ['sell', 'hold']:
                return 'aligned'
            else:
                return 'misaligned'
        
        # Neutral sentiment (between -4 and 4) should be 'hold'
        else:
            if recommendation == 'hold':
                return 'aligned'
            else:
                return 'aligned'  # More lenient for neutral scores
    
    def analyze_record(self, record: SentimentRecord) -> Optional[PerformanceRecord]:
        """
        Analyze a single sentiment record by tracking subsequent price movements.
        
        Args:
            record: A SentimentRecord to analyze
            
        Returns:
            PerformanceRecord with price tracking and accuracy metrics
        """
        try:
            # Parse the prediction timestamp
            prediction_time = datetime.fromisoformat(
                record.timestamp.replace('Z', '+00:00')
            )
            
            # Get prices at all timeframes
            prices = self.price_tracker.get_prices_at_timeframes(
                record.symbol, 
                prediction_time
            )
            
            # Get initial price
            initial_price_point = prices.get(0)
            if not initial_price_point:
                logger.warning(f"No initial price for {record.symbol} at {prediction_time}")
                return None
            
            initial_price = initial_price_point.price
            
            # Calculate returns and accuracy for each timeframe
            def get_return(hours: int) -> Optional[float]:
                price_point = prices.get(hours)
                if price_point and initial_price > 0:
                    return ((price_point.price - initial_price) / initial_price) * 100
                return None
            
            return_4h = get_return(4)
            return_12h = get_return(12)
            return_24h = get_return(24)
            return_36h = get_return(36)
            
            # Calculate accuracy
            def get_accuracy(ret: Optional[float]) -> Optional[str]:
                if ret is not None:
                    return self.calculate_accuracy(record.sentiment_score, ret)
                return None
            
            perf_record = PerformanceRecord(
                symbol=record.symbol,
                sentiment_score=record.sentiment_score,
                recommendation=record.recommendation,
                buy_signal=record.buy_signal,
                prediction_time=record.timestamp,
                initial_price=initial_price,
                price_4h=prices.get(4).price if prices.get(4) else None,
                price_12h=prices.get(12).price if prices.get(12) else None,
                price_24h=prices.get(24).price if prices.get(24) else None,
                price_36h=prices.get(36).price if prices.get(36) else None,
                return_4h=return_4h,
                return_12h=return_12h,
                return_24h=return_24h,
                return_36h=return_36h,
                score_accuracy_4h=get_accuracy(return_4h),
                score_accuracy_12h=get_accuracy(return_12h),
                score_accuracy_24h=get_accuracy(return_24h),
                score_accuracy_36h=get_accuracy(return_36h),
                recommendation_match=self.check_recommendation_alignment(
                    record.sentiment_score,
                    record.recommendation,
                    record.buy_signal
                ),
                processor=record.processor,
                summary=record.summary
            )
            
            return perf_record
            
        except Exception as e:
            logger.error(f"Error analyzing record for {record.symbol}: {e}")
            return None
    
    def analyze_all_records(
        self, 
        records: Optional[List[SentimentRecord]] = None,
        from_gcs: bool = True,
        local_file: Optional[str] = None
    ) -> List[PerformanceRecord]:
        """
        Analyze all sentiment records.
        
        Args:
            records: Optional list of records (if not loading from GCS)
            from_gcs: Whether to load records from GCS
            local_file: Path to local file (for testing)
            
        Returns:
            List of PerformanceRecords
        """
        if records is None:
            if local_file:
                records = self.history_reader.read_local_log(local_file)
            elif from_gcs:
                records = self.history_reader.read_interactions_log()
            else:
                logger.error("No records source specified")
                return []
        
        logger.info(f"Analyzing {len(records)} sentiment records...")
        
        self.performance_records = []
        for i, record in enumerate(records):
            logger.info(f"Processing {i+1}/{len(records)}: {record.symbol}")
            perf = self.analyze_record(record)
            if perf:
                self.performance_records.append(perf)
        
        logger.info(f"✅ Analyzed {len(self.performance_records)} records successfully")
        return self.performance_records
    
    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive accuracy metrics from analyzed records.
        
        Returns:
            Dict with accuracy statistics and insights
        """
        if not self.performance_records:
            return {"error": "No performance records available"}
        
        df = pd.DataFrame([asdict(r) for r in self.performance_records])
        
        metrics = {
            "total_records": len(df),
            "symbols_analyzed": df['symbol'].nunique(),
            "analysis_timestamp": datetime.now().isoformat(),
            "timeframe_accuracy": {},
            "sentiment_buckets": {},
            "recommendation_alignment": {},
            "high_sentiment_mismatches": [],
            "best_performing_scores": {},
            "suggestions": []
        }
        
        # Calculate accuracy by timeframe
        for hours in ANALYSIS_TIMEFRAMES:
            acc_col = f'score_accuracy_{hours}h'
            if acc_col in df.columns:
                valid = df[df[acc_col].notna()]
                if len(valid) > 0:
                    correct = len(valid[valid[acc_col] == 'correct'])
                    incorrect = len(valid[valid[acc_col] == 'incorrect'])
                    neutral = len(valid[valid[acc_col] == 'neutral'])
                    total = len(valid)
                    
                    metrics["timeframe_accuracy"][f"{hours}h"] = {
                        "correct": correct,
                        "incorrect": incorrect,
                        "neutral": neutral,
                        "total": total,
                        "accuracy_rate": round(correct / total * 100, 2) if total > 0 else 0,
                        "error_rate": round(incorrect / total * 100, 2) if total > 0 else 0
                    }
        
        # Analyze by sentiment score buckets
        def bucket_score(score):
            if score >= 8:
                return "very_high_8+"
            elif score >= 6:
                return "high_6-8"
            elif score >= 4:
                return "moderate_4-6"
            elif score >= 0:
                return "low_positive_0-4"
            elif score >= -4:
                return "low_negative_-4-0"
            else:
                return "negative_below_-4"
        
        df['score_bucket'] = df['sentiment_score'].apply(bucket_score)
        
        for bucket in df['score_bucket'].unique():
            bucket_df = df[df['score_bucket'] == bucket]
            bucket_metrics = {
                "count": len(bucket_df),
                "avg_sentiment_score": round(bucket_df['sentiment_score'].mean(), 2)
            }
            
            # Average returns by timeframe
            for hours in ANALYSIS_TIMEFRAMES:
                ret_col = f'return_{hours}h'
                if ret_col in bucket_df.columns:
                    valid_returns = bucket_df[ret_col].dropna()
                    if len(valid_returns) > 0:
                        bucket_metrics[f"avg_return_{hours}h"] = round(valid_returns.mean(), 3)
                        bucket_metrics[f"median_return_{hours}h"] = round(valid_returns.median(), 3)
            
            metrics["sentiment_buckets"][bucket] = bucket_metrics
        
        # Recommendation alignment analysis
        if 'recommendation_match' in df.columns:
            alignment = df['recommendation_match'].value_counts().to_dict()
            metrics["recommendation_alignment"] = alignment
            
            # Find high sentiment mismatches (your specific concern)
            high_sentiment = df[
                (df['sentiment_score'] >= HIGH_SENTIMENT_THRESHOLD) & 
                (df['recommendation_match'] == 'misaligned')
            ]
            
            for _, row in high_sentiment.iterrows():
                metrics["high_sentiment_mismatches"].append({
                    "symbol": row['symbol'],
                    "sentiment_score": row['sentiment_score'],
                    "recommendation": row['recommendation'],
                    "prediction_time": row['prediction_time'],
                    "return_24h": row.get('return_24h'),
                    "comment": "High sentiment but no buy recommendation"
                })
        
        # Find which sentiment score ranges are most predictive
        for bucket, data in metrics["sentiment_buckets"].items():
            if "avg_return_24h" in data:
                avg_ret = data["avg_return_24h"]
                avg_score = data["avg_sentiment_score"]
                
                # Check if direction matches expectation
                if avg_score >= 4 and avg_ret > 0:
                    metrics["best_performing_scores"][bucket] = "correctly_bullish"
                elif avg_score >= 4 and avg_ret < 0:
                    metrics["best_performing_scores"][bucket] = "incorrectly_bullish"
                elif avg_score < 0 and avg_ret < 0:
                    metrics["best_performing_scores"][bucket] = "correctly_bearish"
                elif avg_score < 0 and avg_ret > 0:
                    metrics["best_performing_scores"][bucket] = "incorrectly_bearish"
        
        # Generate suggestions for model improvement
        self._generate_suggestions(metrics, df)
        
        return metrics
    
    def _generate_suggestions(self, metrics: Dict, df: pd.DataFrame):
        """Generate actionable suggestions based on the analysis."""
        suggestions = []
        
        # Check for high sentiment misalignment
        if len(metrics.get("high_sentiment_mismatches", [])) > 0:
            mismatch_count = len(metrics["high_sentiment_mismatches"])
            suggestions.append({
                "issue": "Sentiment-Recommendation Misalignment",
                "description": f"Found {mismatch_count} cases where sentiment score >= {HIGH_SENTIMENT_THRESHOLD} but recommendation was not 'buy'",
                "action": "Consider adjusting recommendation logic to give more weight to high sentiment scores, or investigate why the recommendation differs"
            })
        
        # Check accuracy by timeframe
        best_timeframe = None
        best_accuracy = 0
        for tf, data in metrics.get("timeframe_accuracy", {}).items():
            if data["accuracy_rate"] > best_accuracy:
                best_accuracy = data["accuracy_rate"]
                best_timeframe = tf
        
        if best_timeframe:
            suggestions.append({
                "insight": f"Best Prediction Timeframe: {best_timeframe}",
                "description": f"Sentiment scores are most accurate at the {best_timeframe} timeframe ({best_accuracy}% accuracy)",
                "action": f"Consider optimizing trading strategy for {best_timeframe} holding periods"
            })
        
        # Check which sentiment buckets perform best
        best_bucket = None
        best_return = -float('inf')
        for bucket, data in metrics.get("sentiment_buckets", {}).items():
            ret = data.get("avg_return_24h", -float('inf'))
            if ret > best_return:
                best_return = ret
                best_bucket = bucket
        
        if best_bucket and best_return > 0:
            suggestions.append({
                "insight": f"Best Performing Sentiment Range: {best_bucket}",
                "description": f"Stocks with sentiment in the '{best_bucket}' range had average 24h returns of {best_return}%",
                "action": "Focus buy signals on stocks falling within this sentiment range"
            })
        
        # Threshold calibration suggestion
        high_bucket = metrics.get("sentiment_buckets", {}).get("very_high_8+", {})
        if high_bucket.get("avg_return_24h", 0) < 0:
            suggestions.append({
                "issue": "High Sentiment Underperformance",
                "description": "Stocks with very high sentiment (8+) are underperforming",
                "action": "The sentiment model may be overreacting to hype. Consider raising the buy threshold or adding contrarian signals for extremely high scores"
            })
        
        metrics["suggestions"] = suggestions
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export performance records to a pandas DataFrame for further analysis."""
        if not self.performance_records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self.performance_records])
    
    def export_to_json(self, file_path: str):
        """Export performance records to a JSON file."""
        if not self.performance_records:
            logger.warning("No records to export")
            return
        
        data = [asdict(r) for r in self.performance_records]
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"✅ Exported {len(data)} records to {file_path}")
    
    def export_to_csv(self, file_path: str):
        """Export performance records to a CSV file."""
        df = self.export_to_dataframe()
        if not df.empty:
            df.to_csv(file_path, index=False)
            logger.info(f"✅ Exported {len(df)} records to {file_path}")
    
    def get_training_data(self) -> pd.DataFrame:
        """
        Get data formatted for model training/fine-tuning.
        
        Returns DataFrame with:
        - Input features (sentiment score, recommendation, etc.)
        - Target labels (actual price movements)
        """
        df = self.export_to_dataframe()
        if df.empty:
            return df
        
        # Create training features
        training_df = df[[
            'symbol', 
            'sentiment_score', 
            'prediction_time',
            'initial_price'
        ]].copy()
        
        # Add binary outcome columns
        for hours in ANALYSIS_TIMEFRAMES:
            ret_col = f'return_{hours}h'
            if ret_col in df.columns:
                # Binary: 1 if positive return, 0 if negative
                training_df[f'positive_{hours}h'] = (df[ret_col] > 0).astype(int)
                # Target return
                training_df[f'target_return_{hours}h'] = df[ret_col]
        
        # Add recommendation encoding
        if 'recommendation' in df.columns:
            training_df['recommendation'] = df['recommendation']
            training_df['recommendation_encoded'] = df['recommendation'].map({
                'buy': 1, 'hold': 0, 'sell': -1
            }).fillna(0)
        
        return training_df


def print_metrics_report(metrics: Dict):
    """Print a formatted metrics report to console."""
    print("\n" + "="*60)
    print("📊 GROK SENTIMENT PERFORMANCE ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📈 Total Records Analyzed: {metrics.get('total_records', 0)}")
    print(f"📊 Unique Symbols: {metrics.get('symbols_analyzed', 0)}")
    print(f"🕐 Analysis Time: {metrics.get('analysis_timestamp', 'N/A')}")
    
    # Timeframe Accuracy
    print("\n" + "-"*40)
    print("⏱️ ACCURACY BY TIMEFRAME")
    print("-"*40)
    for tf, data in metrics.get("timeframe_accuracy", {}).items():
        print(f"\n  {tf}:")
        print(f"    ✅ Correct: {data['correct']} ({data['accuracy_rate']}%)")
        print(f"    ❌ Incorrect: {data['incorrect']} ({data['error_rate']}%)")
        print(f"    ⚪ Neutral: {data['neutral']}")
    
    # Sentiment Buckets
    print("\n" + "-"*40)
    print("📊 PERFORMANCE BY SENTIMENT SCORE RANGE")
    print("-"*40)
    for bucket, data in metrics.get("sentiment_buckets", {}).items():
        print(f"\n  {bucket}:")
        print(f"    Count: {data['count']}")
        print(f"    Avg Score: {data.get('avg_sentiment_score', 'N/A')}")
        if 'avg_return_24h' in data:
            ret = data['avg_return_24h']
            emoji = "🟢" if ret > 0 else "🔴"
            print(f"    Avg 24h Return: {emoji} {ret:.3f}%")
    
    # Recommendation Alignment
    print("\n" + "-"*40)
    print("🎯 RECOMMENDATION ALIGNMENT")
    print("-"*40)
    alignment = metrics.get("recommendation_alignment", {})
    for status, count in alignment.items():
        emoji = "✅" if status == "aligned" else "⚠️" if status == "misaligned" else "❔"
        print(f"  {emoji} {status}: {count}")
    
    # High Sentiment Mismatches
    mismatches = metrics.get("high_sentiment_mismatches", [])
    if mismatches:
        print("\n" + "-"*40)
        print("⚠️ HIGH SENTIMENT MISMATCHES (Score >= 7, No Buy)")
        print("-"*40)
        for m in mismatches[:10]:  # Show first 10
            print(f"  • {m['symbol']}: Score {m['sentiment_score']}, Rec: {m['recommendation']}")
            if m.get('return_24h') is not None:
                ret = m['return_24h']
                emoji = "🟢" if ret > 0 else "🔴"
                print(f"    24h Return: {emoji} {ret:.2f}%")
    
    # Suggestions
    print("\n" + "-"*40)
    print("💡 SUGGESTIONS FOR IMPROVEMENT")
    print("-"*40)
    for i, suggestion in enumerate(metrics.get("suggestions", []), 1):
        print(f"\n  {i}. {suggestion.get('issue', suggestion.get('insight', 'Suggestion'))}")
        print(f"     {suggestion.get('description', '')}")
        print(f"     ➡️ {suggestion.get('action', '')}")
    
    print("\n" + "="*60)


# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Grok sentiment performance")
    parser.add_argument("--symbol", type=str, help="Filter by specific symbol")
    parser.add_argument("--local-file", type=str, help="Path to local JSON file instead of GCS")
    parser.add_argument("--export-csv", type=str, help="Export results to CSV file")
    parser.add_argument("--export-json", type=str, help="Export results to JSON file")
    parser.add_argument("--training-data", type=str, help="Export training data to CSV")
    
    args = parser.parse_args()
    
    analyzer = SentimentPerformanceAnalyzer()
    
    # Load and analyze records
    if args.local_file:
        records = analyzer.history_reader.read_local_log(args.local_file)
    else:
        records = analyzer.history_reader.read_interactions_log()
    
    # Filter by symbol if specified
    if args.symbol:
        records = [r for r in records if r.symbol.upper() == args.symbol.upper()]
        print(f"Filtered to {len(records)} records for {args.symbol.upper()}")
    
    # Analyze
    analyzer.analyze_all_records(records=records)
    
    # Get and print metrics
    metrics = analyzer.get_metrics()
    print_metrics_report(metrics)
    
    # Export if requested
    if args.export_csv:
        analyzer.export_to_csv(args.export_csv)
    
    if args.export_json:
        analyzer.export_to_json(args.export_json)
    
    if args.training_data:
        training_df = analyzer.get_training_data()
        training_df.to_csv(args.training_data, index=False)
        print(f"✅ Exported training data to {args.training_data}")
