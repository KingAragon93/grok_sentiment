"""
Stock Data Access Examples for Grok Sentiment Optimization

This script demonstrates how to access stock data from various APIs
to provide context for Grok sentiment analysis.

Assumes .env file is present in the same directory with API keys.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# API Keys from .env
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
POLYGON_API_KEY = os.getenv('POLYGON')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')

class StockDataProvider:
    """Provides stock data from multiple APIs for Grok context."""

    def __init__(self):
        self.session = requests.Session()

    def get_alpaca_quote(self, symbol):
        """Get real-time quote from Alpaca."""
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
        headers = {
            'APCA-API-KEY-ID': ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
        }

        try:
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            return {
                'symbol': symbol,
                'price': data['quote']['ap'],  # Ask price
                'bid': data['quote']['bp'],    # Bid price
                'ask': data['quote']['ap'],    # Ask price
                'volume': data['quote']['as'], # Ask size
                'timestamp': data['quote']['t']
            }
        except Exception as e:
            print(f"Alpaca quote error for {symbol}: {e}")
            return None

    def get_alpaca_bars(self, symbol, timeframe='1D', limit=30):
        """Get historical bars from Alpaca."""
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
        headers = {
            'APCA-API-KEY-ID': ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
        }

        params = {
            'timeframe': timeframe,
            'limit': limit,
            'adjustment': 'raw'
        }

        try:
            response = self.session.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            bars = []
            for bar in data['bars']:
                bars.append({
                    'timestamp': bar['t'],
                    'open': bar['o'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar['v']
                })

            return bars
        except Exception as e:
            print(f"Alpaca bars error for {symbol}: {e}")
            return []

    def get_polygon_quote(self, symbol):
        """Get real-time quote from Polygon."""
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
        params = {'apiKey': POLYGON_API_KEY}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data['results']:
                result = data['results'][0]
                return {
                    'symbol': symbol,
                    'open': result['o'],
                    'high': result['h'],
                    'low': result['l'],
                    'close': result['c'],
                    'volume': result['v'],
                    'timestamp': result['t']
                }
        except Exception as e:
            print(f"Polygon quote error for {symbol}: {e}")
            return None

    def get_alpha_vantage_quote(self, symbol):
        """Get quote from Alpha Vantage."""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_KEY
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': symbol,
                    'price': float(quote['05. price']),
                    'change': float(quote['09. change']),
                    'change_percent': quote['10. change percent'],
                    'volume': int(quote['06. volume']),
                    'latest_trading_day': quote['07. latest trading day']
                }
        except Exception as e:
            print(f"Alpha Vantage quote error for {symbol}: {e}")
            return None

    def get_comprehensive_stock_data(self, symbol):
        """
        Get comprehensive stock data from multiple sources.
        Useful for providing rich context to Grok.
        """
        data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {}
        }

        # Try Alpaca first (most reliable for real-time)
        alpaca_quote = self.get_alpaca_quote(symbol)
        if alpaca_quote:
            data['sources']['alpaca'] = alpaca_quote

        # Try Polygon for additional data
        polygon_quote = self.get_polygon_quote(symbol)
        if polygon_quote:
            data['sources']['polygon'] = polygon_quote

        # Try Alpha Vantage as backup
        av_quote = self.get_alpha_vantage_quote(symbol)
        if av_quote:
            data['sources']['alpha_vantage'] = av_quote

        # Get recent bars for trend analysis
        bars = self.get_alpaca_bars(symbol, timeframe='1D', limit=5)
        if bars:
            data['recent_bars'] = bars

        return data

    def format_for_grok(self, stock_data):
        """
        Format stock data into a readable string for Grok context.
        """
        if not stock_data or 'sources' not in stock_data:
            return f"No stock data available for {stock_data.get('symbol', 'unknown')}"

        symbol = stock_data['symbol']
        lines = [f"Stock Data for {symbol}:"]
        lines.append("=" * 40)

        # Use the most recent/reliable price
        price = None
        if 'alpaca' in stock_data['sources']:
            price = stock_data['sources']['alpaca']['price']
            lines.append(f"Current Price: ${price:.2f}")
        elif 'polygon' in stock_data['sources']:
            price = stock_data['sources']['polygon']['close']
            lines.append(f"Last Close: ${price:.2f}")

        # Show recent trend
        if 'recent_bars' in stock_data and len(stock_data['recent_bars']) >= 2:
            bars = stock_data['recent_bars']
            recent = bars[-1]
            previous = bars[-2]

            change = recent['close'] - previous['close']
            change_pct = (change / previous['close']) * 100

            lines.append(f"Recent Change: ${change:.2f} ({change_pct:+.2f}%)")
            lines.append(f"Volume: {recent['volume']:,}")

        # Add volatility info
        if 'recent_bars' in stock_data and len(stock_data['recent_bars']) >= 5:
            closes = [bar['close'] for bar in stock_data['recent_bars']]
            volatility = pd.Series(closes).std() / pd.Series(closes).mean() * 100
            lines.append(f"5-Day Volatility: {volatility:.2f}%")

        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    provider = StockDataProvider()

    # Test with a few symbols
    symbols = ['AAPL', 'TSLA', 'NVDA']

    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Getting data for {symbol}")
        print('='*50)

        # Get comprehensive data
        data = provider.get_comprehensive_stock_data(symbol)

        # Format for Grok
        grok_context = provider.format_for_grok(data)

        print(grok_context)

        # Show raw data structure
        print(f"\nRaw data keys: {list(data.keys())}")
        if 'sources' in data:
            print(f"Available sources: {list(data['sources'].keys())}")

        # Rate limiting
        time.sleep(1)