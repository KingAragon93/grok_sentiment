#!/usr/bin/env python3
"""
GCS Logger for Aligned Recommendations

Logs all aligned recommendation calls to Google Cloud Storage for future
analysis and model training. Creates a new log file for tracking predictions.

Log file: gs://historical_stock_day/aligned_recommendations_log.json
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# GCS Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "historical_stock_day")
ALIGNED_LOG_FILE = "aligned_recommendations_log.json"
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "src/GoogleCloudServiceAccount.json")


@dataclass
class AlignedRecommendationLog:
    """Structure for logging aligned recommendations."""
    timestamp: str
    symbol: str
    sentiment_score: float
    recommendation: str  # buy, hold, sell
    buy_signal: bool
    confidence: str  # high, medium, low
    alignment_reason: str
    citations_count: int
    recommended_hold_hours: int
    # Price tracking fields (filled in later by analysis)
    price_at_recommendation: Optional[float] = None
    price_4h: Optional[float] = None
    price_12h: Optional[float] = None
    price_24h: Optional[float] = None
    price_36h: Optional[float] = None
    return_4h: Optional[float] = None
    return_12h: Optional[float] = None
    return_24h: Optional[float] = None
    return_36h: Optional[float] = None
    outcome: Optional[str] = None  # profitable, loss, pending


class GCSAlignedLogger:
    """Handles logging aligned recommendations to GCS."""
    
    def __init__(self):
        self.bucket_name = GCS_BUCKET_NAME
        self.log_file = ALIGNED_LOG_FILE
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize GCS client with service account."""
        try:
            if os.path.exists(CREDENTIALS_PATH):
                self.client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
                logger.info(f"✅ GCS client initialized with service account")
            else:
                self.client = storage.Client()
                logger.info(f"✅ GCS client initialized with default credentials")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GCS client: {e}")
            self.client = None
    
    def _get_bucket(self):
        """Get the GCS bucket."""
        if not self.client:
            return None
        try:
            return self.client.bucket(self.bucket_name)
        except Exception as e:
            logger.error(f"❌ Failed to get bucket {self.bucket_name}: {e}")
            return None
    
    def load_logs(self) -> List[Dict[str, Any]]:
        """Load existing logs from GCS."""
        bucket = self._get_bucket()
        if not bucket:
            return []
        
        try:
            blob = bucket.blob(self.log_file)
            if blob.exists():
                content = blob.download_as_text()
                data = json.loads(content)
                logger.info(f"📥 Loaded {len(data)} existing log entries")
                return data
            else:
                logger.info(f"📄 No existing log file, starting fresh")
                return []
        except Exception as e:
            logger.error(f"❌ Failed to load logs: {e}")
            return []
    
    def save_logs(self, logs: List[Dict[str, Any]]) -> bool:
        """Save logs to GCS."""
        bucket = self._get_bucket()
        if not bucket:
            return False
        
        try:
            blob = bucket.blob(self.log_file)
            content = json.dumps(logs, indent=2, default=str)
            blob.upload_from_string(content, content_type='application/json')
            logger.info(f"📤 Saved {len(logs)} log entries to GCS")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save logs: {e}")
            return False
    
    def log_recommendation(self, result: Dict[str, Any]) -> bool:
        """
        Log an aligned recommendation result to GCS.
        
        Args:
            result: The result dict from get_aligned_recommendation()
        
        Returns:
            True if logged successfully
        """
        if result.get("status") != "success":
            logger.warning(f"Skipping failed recommendation log")
            return False
        
        # Create log entry
        log_entry = AlignedRecommendationLog(
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            symbol=result.get("symbol", ""),
            sentiment_score=result.get("sentiment_score", 0),
            recommendation=result.get("recommendation", "hold"),
            buy_signal=result.get("buy_signal", False),
            confidence=result.get("confidence", "medium"),
            alignment_reason=result.get("alignment_reason", ""),
            citations_count=result.get("citations_count", 0),
            recommended_hold_hours=result.get("recommended_hold_hours", 36)
        )
        
        # Load existing logs
        logs = self.load_logs()
        
        # Append new entry
        logs.append(asdict(log_entry))
        
        # Save back to GCS
        return self.save_logs(logs)
    
    def get_logs_for_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Get all logs for a specific date (YYYY-MM-DD format)."""
        logs = self.load_logs()
        return [
            log for log in logs 
            if log.get("timestamp", "").startswith(date_str)
        ]
    
    def get_buy_recommendations(self) -> List[Dict[str, Any]]:
        """Get all buy recommendations."""
        logs = self.load_logs()
        return [log for log in logs if log.get("recommendation") == "buy"]
    
    def get_pending_analysis(self) -> List[Dict[str, Any]]:
        """Get recommendations that need price tracking (no outcome yet)."""
        logs = self.load_logs()
        return [log for log in logs if log.get("outcome") is None]
    
    def update_log_with_prices(self, timestamp: str, symbol: str, 
                                price_data: Dict[str, float]) -> bool:
        """
        Update a log entry with price tracking data.
        
        Args:
            timestamp: Original recommendation timestamp
            symbol: Stock symbol
            price_data: Dict with price_at_recommendation, price_4h, etc.
        """
        logs = self.load_logs()
        
        for log in logs:
            if log.get("timestamp") == timestamp and log.get("symbol") == symbol:
                log.update(price_data)
                
                # Calculate outcome if we have 36h data
                if log.get("return_36h") is not None:
                    if log.get("buy_signal"):
                        log["outcome"] = "profitable" if log["return_36h"] > 0 else "loss"
                    else:
                        log["outcome"] = "correct_hold" if log["return_36h"] <= 0 else "missed_opportunity"
                
                return self.save_logs(logs)
        
        logger.warning(f"Log entry not found: {symbol} @ {timestamp}")
        return False


def init_aligned_log_file():
    """Initialize the aligned recommendations log file in GCS."""
    gcs_logger = GCSAlignedLogger()
    
    # Check if file exists
    logs = gcs_logger.load_logs()
    
    if not logs:
        # Create empty log structure
        initial_data = []
        if gcs_logger.save_logs(initial_data):
            print(f"✅ Created new log file: gs://{GCS_BUCKET_NAME}/{ALIGNED_LOG_FILE}")
            return True
        else:
            print(f"❌ Failed to create log file")
            return False
    else:
        print(f"📄 Log file already exists with {len(logs)} entries")
        return True


if __name__ == "__main__":
    init_aligned_log_file()
