"""Configuration and environment variable management for the options engine."""

import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the options engine."""
    
    # API Configuration
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    
    # Risk-free rate configuration
    RISK_FREE_RATE_SOURCE: str = os.getenv("RISK_FREE_RATE_SOURCE", "CONSTANT")  # "FRED" or "CONSTANT"
    CONSTANT_RF_RATE: float = float(os.getenv("CONSTANT_RF_RATE", "0.05"))  # 5% default
    
    # Default analysis parameters
    DEFAULT_SYMBOL: str = os.getenv("DEFAULT_SYMBOL", "SPY")
    DEFAULT_MIN_DTE: int = int(os.getenv("DEFAULT_MIN_DTE", "7"))
    DEFAULT_MAX_DTE: int = int(os.getenv("DEFAULT_MAX_DTE", "21"))
    
    # Liquidity filters
    MIN_OPEN_INTEREST: int = int(os.getenv("MIN_OPEN_INTEREST", "10"))
    MIN_VOLUME: int = int(os.getenv("MIN_VOLUME", "10"))
    MAX_BID_ASK_SPREAD: float = float(os.getenv("MAX_BID_ASK_SPREAD", "3.0"))
    MAX_SPREAD_PERCENTAGE: float = float(os.getenv("MAX_SPREAD_PERCENTAGE", "0.05"))  # 5%
    
    # Mispricing detection
    MISPRICING_THRESHOLD: float = float(os.getenv("MISPRICING_THRESHOLD", "3.0"))  # 3%
    
    # Dividend yield fallbacks for common symbols
    DIVIDEND_YIELD_FALLBACKS = {
        'SPY': 0.013,   # ~1.3% for S&P 500 ETF
        'QQQ': 0.005,   # ~0.5% for Nasdaq ETF  
        'IWM': 0.011,   # ~1.1% for Russell 2000 ETF
        'DIA': 0.015,   # ~1.5% for Dow ETF
        'VTI': 0.013,   # ~1.3% for Total Stock Market ETF
        'VOO': 0.013,   # ~1.3% for S&P 500 ETF
        'AAPL': 0.005,  # ~0.5% for Apple
        'MSFT': 0.007,  # ~0.7% for Microsoft
        'GOOGL': 0.000, # ~0% for Google (no dividend)
        'AMZN': 0.000,  # ~0% for Amazon (no dividend)
        'TSLA': 0.000,  # ~0% for Tesla (no dividend)
        'META': 0.000,  # ~0% for Meta (no dividend)
        'NVDA': 0.002,  # ~0.2% for Nvidia
        'HOOD': 0.000,  # ~0% for Robinhood (no dividend)
    }
    
    # Default dividend yield for unknown symbols
    DEFAULT_DIVIDEND_YIELD: float = 0.02  # 2% for individual stocks
    
    @classmethod
    def get_dividend_yield_fallback(cls, symbol: str) -> float:
        """Get fallback dividend yield for a symbol."""
        return cls.DIVIDEND_YIELD_FALLBACKS.get(symbol.upper(), cls.DEFAULT_DIVIDEND_YIELD)
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        if not cls.POLYGON_API_KEY:
            print("❌ POLYGON_API_KEY environment variable not set")
            return False
        
        if cls.CONSTANT_RF_RATE < 0 or cls.CONSTANT_RF_RATE > 0.2:
            print(f"⚠️ Warning: Risk-free rate {cls.CONSTANT_RF_RATE:.1%} seems unusual")
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("📋 Current Configuration:")
        print(f"   • Default Symbol: {cls.DEFAULT_SYMBOL}")
        print(f"   • Risk-free Rate: {cls.CONSTANT_RF_RATE:.2%} ({cls.RISK_FREE_RATE_SOURCE})")
        print(f"   • DTE Range: {cls.DEFAULT_MIN_DTE}-{cls.DEFAULT_MAX_DTE} days")
        print(f"   • Min Open Interest: {cls.MIN_OPEN_INTEREST}")
        print(f"   • Min Volume: {cls.MIN_VOLUME}")
        print(f"   • Max Spread: ${cls.MAX_BID_ASK_SPREAD:.2f} or {cls.MAX_SPREAD_PERCENTAGE:.1%}")
        print(f"   • Mispricing Threshold: {cls.MISPRICING_THRESHOLD:.1%}")

# Global config instance
config = Config()
config.validate() 