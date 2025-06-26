"""Historical options data collection from Polygon.io."""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time
import asyncio
import concurrent.futures
from functools import partial

import pandas as pd
import numpy as np
from polygon import RESTClient
from tqdm import tqdm

from .config import config
from .providers.polygon import get_underlying_price, get_dividend_yield

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """Collect and store historical options data from Polygon.io with performance optimizations."""
    
    def __init__(self, data_dir: str = "data", min_dte: int = 7, max_dte: int = 180, max_workers: int = 10, 
                 min_strike_pct: float = 0.8, max_strike_pct: float = 1.2):
        """
        Initialize the historical data collector with optimized settings.
        
        Parameters
        ----------
        data_dir : str
            Directory to store data files
        min_dte : int
            Minimum days to expiration
        max_dte : int
            Maximum days to expiration
        max_workers : int
            Maximum concurrent workers (optimized for API limits)
        min_strike_pct : float
            Minimum strike as percentage of underlying price
        max_strike_pct : float
            Maximum strike as percentage of underlying price
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.min_dte = min_dte
        self.max_dte = max_dte
        # Optimize worker count to prevent connection pool saturation
        # The bottleneck is API rate limits, not CPU, so fewer workers is better
        self.max_workers = min(max_workers, 5)  # Cap at 5 to prevent HTTP pool issues
        self.min_strike_pct = min_strike_pct
        self.max_strike_pct = max_strike_pct
        
        # Initialize Polygon client and metadata
        self.client = RESTClient(config.POLYGON_API_KEY)
        self.metadata_file = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized HistoricalDataCollector with {self.max_workers} workers (optimized for API limits)")
    
    def _load_metadata(self) -> Dict:
        """Load or create metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "symbols": {},
                "last_updated": None,
                "total_files": 0,
                "total_size_mb": 0
            }
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _get_symbol_dir(self, symbol: str) -> Path:
        """Get directory for a specific symbol."""
        symbol_dir = self.data_dir / symbol.upper()
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir
    
    def _get_historical_underlying_price(self, symbol: str, date: datetime) -> float:
        """
        Get historical underlying price for a specific date.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        date : datetime
            Date to get price for
            
        Returns
        -------
        float
            Historical closing price
        """
        try:
            # Get historical stock price for the date
            date_str = date.strftime("%Y-%m-%d")
            
            # Use Polygon's aggregates endpoint for historical stock price
            aggs = list(self.client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=date_str,
                to=date_str
            ))
            
            if aggs and len(aggs) > 0:
                return float(aggs[0].close)
            else:
                logger.warning(f"No stock price data for {symbol} on {date_str}")
                # Fallback to current price if historical not available
                return get_underlying_price(symbol)
                
        except Exception as e:
            logger.error(f"Error getting historical price for {symbol} on {date}: {e}")
            # Fallback to current price
            return get_underlying_price(symbol)
    
    def _get_available_option_tickers(self, symbol: str, date: datetime) -> List[str]:
        """
        Get available option tickers for a symbol on a specific date.
        
        This constructs option tickers that would have been available on the historical date
        based on standard monthly and weekly expiration patterns.
        
        Returns both tickers and pre-computed parsing data for performance.
        """
        try:
            option_tickers = []
            
            # Generate expiration dates that would have been available on the target date
            expiration_dates = self._get_historical_expiration_dates(symbol, date)
            
            if not expiration_dates:
                logger.warning(f"No expiration dates found for {symbol} on {date.strftime('%Y-%m-%d')}")
                return []
            
            # Get current underlying price to estimate strike range
            underlying_price = self._get_historical_underlying_price(symbol, date)
            
            # Generate strike prices around the underlying price
            strikes = self._generate_strike_prices(underlying_price)
            
            # Pre-allocate list for better performance (Python lists auto-resize efficiently)
            estimated_size = len(expiration_dates) * len(strikes) * 2  # calls + puts
            option_tickers = []
            
            # Construct option tickers for each expiration and strike
            for exp_date in expiration_dates:
                dte = (exp_date - date).days
                
                # Filter by DTE range
                if not (self.min_dte <= dte <= self.max_dte):
                    continue
                
                for strike in strikes:
                    # Create call and put tickers
                    for option_type in ['C', 'P']:
                        ticker = self._construct_option_ticker(symbol, exp_date, option_type, strike)
                        if ticker:
                            option_tickers.append(ticker)
            
            logger.info(f"Constructed {len(option_tickers)} option tickers for {symbol} on {date.strftime('%Y-%m-%d')}")
            return option_tickers
            
        except Exception as e:
            logger.error(f"Error constructing option tickers for {symbol} on {date}: {e}")
            return []
    
    def _get_historical_expiration_dates(self, symbol: str, date: datetime) -> List[datetime]:
        """
        Get expiration dates that would have been available on a historical date.
        
        This generates monthly and weekly expirations based on standard patterns.
        """
        expiration_dates = []
        
        # Start from the month of the target date
        current_date = date.replace(day=1)  # First of the month
        
        # Generate expirations for the next 12 months
        for months_ahead in range(12):
            # Calculate the target month
            year = current_date.year
            month = current_date.month + months_ahead
            
            # Handle year rollover
            while month > 12:
                month -= 12
                year += 1
            
            # Monthly expiration: Third Friday of the month
            third_friday = self._get_third_friday(year, month)
            
            # Only include if it's after our target date
            if third_friday > date:
                expiration_dates.append(third_friday)
            
            # Add weekly expirations for near-term months (first 2 months)
            if months_ahead < 2:
                weekly_expirations = self._get_weekly_expirations(year, month, date)
                expiration_dates.extend(weekly_expirations)
        
        return sorted(expiration_dates)
    
    def _get_third_friday(self, year: int, month: int) -> datetime:
        """Get the third Friday of a given month."""
        from calendar import monthrange
        
        # Find the first day of the month
        first_day = datetime(year, month, 1)
        
        # Find the first Friday
        days_until_friday = (4 - first_day.weekday()) % 7  # Friday is weekday 4
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Third Friday is 14 days later
        third_friday = first_friday + timedelta(days=14)
        
        return third_friday
    
    def _get_weekly_expirations(self, year: int, month: int, after_date: datetime) -> List[datetime]:
        """Get weekly expiration dates for a month using optimized algorithm."""
        weekly_exps = []
        
        # Pre-calculate third Friday to avoid repeated computation
        third_friday = self._get_third_friday(year, month)
        
        # Find first Friday of the month using modular arithmetic
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7  # Friday is weekday 4
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Jump directly to each Friday (every 7 days) instead of checking every day
        current_friday = first_friday
        while current_friday.month == month:
            # Skip the third Friday (monthly expiration) and dates before target
            if current_friday != third_friday and current_friday > after_date:
                weekly_exps.append(current_friday)
            current_friday += timedelta(days=7)  # Jump to next Friday
        
        return weekly_exps
    
    def _generate_strike_prices(self, underlying_price: float) -> List[float]:
        """
        Generate focused strike prices around the underlying price for better performance.
        Uses a tighter range and fewer strikes to reduce API calls.
        """
        strikes = []
        
        # Determine strike spacing based on underlying price
        if underlying_price < 50:
            spacing = 2.5
        elif underlying_price < 100:
            spacing = 5.0
        elif underlying_price < 200:
            spacing = 5.0
        else:
            spacing = 10.0
        
        # Use a tighter range around underlying price (85% to 115% instead of 70% to 130%)
        # This focuses on more liquid, tradeable options
        min_strike = underlying_price * self.min_strike_pct
        max_strike = underlying_price * self.max_strike_pct
        
        current_strike = int(min_strike / spacing) * spacing
        
        while current_strike <= max_strike:
            strikes.append(current_strike)
            current_strike += spacing
        
        logger.debug(f"Generated {len(strikes)} strikes from ${min_strike:.0f} to ${max_strike:.0f} (spacing: ${spacing})")
        return strikes
    
    def _construct_option_ticker(self, symbol: str, expiration: datetime, option_type: str, strike: float) -> str:
        """
        Construct an option ticker in Polygon format.
        
        Format: O:SYMBOL[YY]MMDDCPPPPPPPPP
        Example: O:QQQ240119C00400000
        """
        try:
            # Format expiration date as YYMMDD
            exp_str = expiration.strftime("%y%m%d")
            
            # Format strike as 8-digit integer (multiply by 1000)
            strike_int = int(strike * 1000)
            strike_str = f"{strike_int:08d}"
            
            # Construct ticker
            ticker = f"O:{symbol}{exp_str}{option_type}{strike_str}"
            
            return ticker
            
        except Exception as e:
            logger.error(f"Error constructing ticker for {symbol} {expiration} {option_type} {strike}: {e}")
            return None
    
    def _get_historical_option_quotes(self, option_ticker: str, date: datetime) -> Optional[Dict]:
        """
        Get historical quotes for a specific option contract on a specific date.
        
        Parameters
        ----------
        option_ticker : str
            Option ticker (e.g., 'O:QQQ240119C00400000')
        date : datetime
            Date to get quotes for
            
        Returns
        -------
        Dict or None
            Option quote data if available
        """
        try:
            date_str = date.strftime("%Y-%m-%d")
            
            # Get historical quotes for the option ticker
            quotes_iter = self.client.list_quotes(
                ticker=option_ticker,
                timestamp=date_str,
                limit=1000,
                order="desc"
            )
            
            quotes = list(quotes_iter)
            
            if not quotes:
                return None
            
            # Get the last quote of the day (most recent)
            last_quote = quotes[0]
            
            # Extract quote data
            quote_data = {
                'bid': getattr(last_quote, 'bid_price', 0),
                'ask': getattr(last_quote, 'ask_price', 0),
                'bid_size': getattr(last_quote, 'bid_size', 0),
                'ask_size': getattr(last_quote, 'ask_size', 0),
                'timestamp': getattr(last_quote, 'sip_timestamp', 0)
            }
            
            # Validate quote data
            if quote_data['bid'] > 0 and quote_data['ask'] > 0 and quote_data['ask'] > quote_data['bid']:
                return quote_data
            else:
                return None
                
        except Exception as e:
            logger.debug(f"No quotes found for {option_ticker} on {date_str}: {e}")
            return None
    
    def _parse_option_ticker(self, option_ticker: str) -> Dict:
        """
        Parse option ticker to extract contract details.
        
        Parameters
        ----------
        option_ticker : str
            Option ticker (e.g., 'O:QQQ240119C00400000')
            
        Returns
        -------
        Dict
            Parsed contract details
        """
        try:
            # Format: O:SYMBOL[YY]MMDDCPPPPPPPPP
            # Example: O:QQQ240119C00400000
            
            if not option_ticker.startswith('O:'):
                return {}
            
            ticker_part = option_ticker[2:]  # Remove 'O:'
            
            # Find where the date starts (after the symbol)
            # Look for pattern YYMMDD
            import re
            match = re.search(r'(\d{6})([CP])(\d{8})', ticker_part)
            
            if not match:
                return {}
            
            symbol = ticker_part[:match.start()]
            date_part = match.group(1)
            option_type = match.group(2)
            strike_part = match.group(3)
            
            # Parse date (YYMMDD format)
            year = 2000 + int(date_part[:2])
            month = int(date_part[2:4])
            day = int(date_part[4:6])
            expiration_date = f"{year:04d}-{month:02d}-{day:02d}"
            
            # Parse strike (8 digits, divide by 1000)
            strike = int(strike_part) / 1000.0
            
            return {
                'underlying_symbol': symbol,
                'expiration_date': expiration_date,
                'option_type': 'call' if option_type == 'C' else 'put',
                'strike': strike
            }
            
        except Exception as e:
            logger.error(f"Error parsing option ticker {option_ticker}: {e}")
            return {}
    
    def fetch_daily_option_chain(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch historical option chain for a specific date using optimized concurrent processing.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        date : datetime
            Date to fetch historical data for
            
        Returns
        -------
        pd.DataFrame or None
            Historical option chain data if successful
        """
        try:
            logger.info(f"Fetching historical option chain for {symbol} on {date.strftime('%Y-%m-%d')}")
            
            # Get historical underlying price
            underlying_price = self._get_historical_underlying_price(symbol, date)
            logger.info(f"Historical underlying price for {symbol}: ${underlying_price:.2f}")
            
            # Get available option tickers for this date
            option_tickers = self._get_available_option_tickers(symbol, date)
            
            if not option_tickers:
                logger.warning(f"No option tickers found for {symbol} on {date.strftime('%Y-%m-%d')}")
                return None
            
            logger.info(f"Fetching quotes for {len(option_tickers)} option contracts using {self.max_workers} workers")
            
            # Pre-compute ticker parsing for all tickers to avoid O(N) parsing in hot loop
            ticker_cache = {}
            for ticker in option_tickers:
                parsed = self._parse_option_ticker(ticker)
                if parsed:
                    ticker_cache[ticker] = parsed
            
            logger.debug(f"Pre-parsed {len(ticker_cache)} tickers")
            
            # Pre-compute date string to avoid repeated strftime calls
            date_str = date.strftime("%Y-%m-%d")
            
            # Fetch all quotes concurrently
            quote_results = self._get_historical_option_quotes_batch(option_tickers, date)
            
            # Pre-allocate list with estimated size for better performance
            estimated_successful = len(option_tickers) // 2  # Assume ~50% success rate
            options_data = []
            successful_quotes = 0
            
            # Process results with optimized loop
            for ticker, quote_data in quote_results:
                if quote_data is None:
                    continue
                
                # Use pre-computed ticker parsing (O(1) lookup vs O(N) parsing)
                contract_details = ticker_cache.get(ticker)
                if not contract_details:
                    continue
                
                # Calculate DTE (could be pre-computed too, but datetime parsing is fast)
                exp_date = datetime.strptime(contract_details['expiration_date'], "%Y-%m-%d")
                dte = (exp_date - date).days
                
                # Create option data record with pre-computed values
                option_data = {
                    'ticker': ticker,
                    'underlying_symbol': contract_details['underlying_symbol'],
                    'expiration_date': contract_details['expiration_date'],
                    'strike': contract_details['strike'],
                    'option_type': contract_details['option_type'],
                    'bid': quote_data['bid'],
                    'ask': quote_data['ask'],
                    'mid_price': (quote_data['bid'] + quote_data['ask']) / 2,
                    'bid_size': quote_data['bid_size'],
                    'ask_size': quote_data['ask_size'],
                    'dte': dte,
                    'underlying_price': underlying_price,
                    'date': date_str,  # Use pre-computed string
                    'timestamp': quote_data['timestamp'],
                    'data_source': 'historical'
                }
                
                options_data.append(option_data)
                successful_quotes += 1
            
            logger.info(f"Successfully retrieved {successful_quotes} option quotes out of {len(option_tickers)} contracts ({successful_quotes/len(option_tickers)*100:.1f}% success rate)")
            
            if not options_data:
                logger.warning(f"No valid option quotes found for {symbol} on {date_str}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(options_data)
            
            # Add spread percentage
            df['spread_pct'] = (df['ask'] - df['bid']) / df['bid']
            
            # Validate data quality
            if self._validate_option_chain(df, symbol, date_str):
                logger.info(f"Successfully collected {len(df)} historical options for {symbol} on {date_str}")
                return df
            else:
                logger.warning(f"Data validation failed for {symbol} on {date_str}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical option chain for {symbol} on {date.strftime('%Y-%m-%d')}: {e}")
            return None
    
    def _validate_option_chain(self, df: pd.DataFrame, symbol: str, date: str) -> bool:
        """
        Validate historical option chain data quality.
        
        Parameters
        ----------
        df : pd.DataFrame
            Option chain data
        symbol : str
            Stock symbol
        date : str
            Date string
            
        Returns
        -------
        bool
            True if data is valid
        """
        if df.empty:
            logger.warning(f"No data for {symbol} on {date}")
            return False
        
        # Check for required columns
        required_cols = ['strike', 'expiration_date', 'option_type', 'bid', 'ask', 'dte']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for {symbol} on {date}: {missing_cols}")
            return False
        
        # Check for minimum number of options
        if len(df) < 5:  # Reduced threshold for historical data
            logger.warning(f"Too few options for {symbol} on {date}: {len(df)}")
            return False
        
        # Check for reasonable bid/ask spreads
        valid_spreads = df[df['spread_pct'] < 1.0]  # Less than 100% spread
        if len(valid_spreads) < len(df) * 0.5:  # At least 50% should have reasonable spreads
            logger.warning(f"Too many wide spreads for {symbol} on {date}")
            return False
        
        logger.info(f"Data validation passed for {symbol} on {date}: {len(df)} options, avg spread: {df['spread_pct'].mean():.1%}")
        return True
    
    def collect_historical_data(self, 
                              symbol: str, 
                              start_date: datetime, 
                              end_date: datetime,
                              force_refresh: bool = False) -> Dict:
        """
        Collect historical option chain data for a symbol using Polygon.io's historical API.
        
        Parameters
        ----------
        symbol : str
            Stock symbol
        start_date : datetime
            Start date for data collection
        end_date : datetime
            End date for data collection
        force_refresh : bool
            Force refresh existing data
            
        Returns
        -------
        Dict
            Collection summary
        """
        symbol = symbol.upper()
        symbol_dir = self._get_symbol_dir(symbol)
        
        # Initialize symbol metadata
        if symbol not in self.metadata["symbols"]:
            self.metadata["symbols"][symbol] = {
                "first_date": None,
                "last_date": None,
                "total_files": 0,
                "total_size_mb": 0,
                "last_collection": None
            }
        
        # Generate date range (skip weekends)
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                date_range.append(current_date)
            current_date += timedelta(days=1)
        
        # Collection statistics
        collected = 0
        skipped = 0
        failed = 0
        
        logger.info(f"Collecting historical data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"DTE range: {self.min_dte} to {self.max_dte} days")
        
        for date in tqdm(date_range, desc=f"Collecting {symbol}"):
            date_str = date.strftime("%Y-%m-%d")
            file_path = symbol_dir / f"{date_str}.parquet"
            
            # Skip if file exists and not forcing refresh
            if file_path.exists() and not force_refresh:
                skipped += 1
                continue
            
            # Fetch historical data
            df = self.fetch_daily_option_chain(symbol, date)
            
            if df is not None and not df.empty:
                # Save to Parquet
                df.to_parquet(file_path, index=False)
                
                # Update metadata
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                self.metadata["symbols"][symbol]["total_files"] += 1
                self.metadata["symbols"][symbol]["total_size_mb"] += file_size
                self.metadata["symbols"][symbol]["last_collection"] = datetime.now().isoformat()
                
                if self.metadata["symbols"][symbol]["first_date"] is None:
                    self.metadata["symbols"][symbol]["first_date"] = date_str
                self.metadata["symbols"][symbol]["last_date"] = date_str
                
                collected += 1
                logger.info(f"Collected {len(df)} historical options for {symbol} on {date_str}")
            else:
                failed += 1
                logger.warning(f"Failed to collect historical data for {symbol} on {date_str}")
        
        # Update global metadata
        self.metadata["last_updated"] = datetime.now().isoformat()
        self._save_metadata()
        
        summary = {
            "symbol": symbol,
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "collected": collected,
            "skipped": skipped,
            "failed": failed,
            "total_files": self.metadata["symbols"][symbol]["total_files"],
            "total_size_mb": round(self.metadata["symbols"][symbol]["total_size_mb"], 2)
        }
        
        logger.info(f"Historical collection complete for {symbol}: {collected} new files, {skipped} skipped, {failed} failed")
        return summary
    
    def get_available_dates(self, symbol: str) -> List[str]:
        """Get list of available dates for a symbol."""
        symbol_dir = self._get_symbol_dir(symbol)
        if not symbol_dir.exists():
            return []
        
        dates = []
        for file_path in symbol_dir.glob("*.parquet"):
            date_str = file_path.stem
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date_str)
            except ValueError:
                continue
        
        return sorted(dates)
    
    def load_daily_data(self, symbol: str, date: str) -> Optional[pd.DataFrame]:
        """Load historical option chain data for a specific date."""
        symbol_dir = self._get_symbol_dir(symbol)
        file_path = symbol_dir / f"{date}.parquet"
        
        if file_path.exists():
            return pd.read_parquet(file_path)
        else:
            return None
    
    def get_collection_status(self) -> Dict:
        """Get status of all collected data."""
        return self.metadata
    
    def _get_historical_option_quotes_batch(self, option_tickers: List[str], date: datetime) -> List[Tuple[str, Optional[Dict]]]:
        """
        Get historical quotes for multiple option contracts concurrently.
        
        Parameters
        ----------
        option_tickers : List[str]
            List of option tickers to fetch
        date : datetime
            Date to get quotes for
            
        Returns
        -------
        List[Tuple[str, Optional[Dict]]]
            List of (ticker, quote_data) tuples
        """
        def fetch_single_quote(ticker: str) -> Tuple[str, Optional[Dict]]:
            """Helper function to fetch a single quote."""
            return (ticker, self._get_historical_option_quotes(ticker, date))
        
        results = []
        
        # Use ThreadPoolExecutor for concurrent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(fetch_single_quote, ticker): ticker 
                for ticker in option_tickers
            }
            
            # Collect results with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_ticker), 
                             total=len(option_tickers), 
                             desc="Fetching quotes", 
                             leave=False):
                try:
                    ticker, quote_data = future.result(timeout=30)  # 30 second timeout
                    results.append((ticker, quote_data))
                except Exception as e:
                    ticker = future_to_ticker[future]
                    logger.debug(f"Failed to fetch quote for {ticker}: {e}")
                    results.append((ticker, None))
        
        return results


def collect_data_for_backtesting(symbols: List[str], 
                                start_date: str, 
                                end_date: str,
                                data_dir: str = "data",
                                min_dte: int = 7,
                                max_dte: int = 180,
                                max_workers: int = 10,
                                min_strike_pct: float = 0.8,
                                max_strike_pct: float = 1.2) -> Dict:
    """
    Collect historical data for multiple symbols for backtesting using optimized concurrent processing.
    
    Parameters
    ----------
    symbols : List[str]
        List of stock symbols
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    data_dir : str
        Directory to store data
    min_dte : int
        Minimum days to expiration
    max_dte : int
        Maximum days to expiration
    max_workers : int
        Maximum number of concurrent workers for API calls
    min_strike_pct : float
        Minimum strike as percentage of underlying price
    max_strike_pct : float
        Maximum strike as percentage of underlying price
        
    Returns
    -------
    Dict
        Collection summary for all symbols
    """
    collector = HistoricalDataCollector(
        data_dir, 
        min_dte=min_dte, 
        max_dte=max_dte, 
        max_workers=max_workers,
        min_strike_pct=min_strike_pct,
        max_strike_pct=max_strike_pct
    )
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    results = {}
    
    for symbol in symbols:
        logger.info(f"Starting historical collection for {symbol}")
        try:
            result = collector.collect_historical_data(symbol, start_dt, end_dt)
            results[symbol] = result
        except Exception as e:
            logger.error(f"Failed to collect historical data for {symbol}: {e}")
            results[symbol] = {"error": str(e)}
    
    return results 