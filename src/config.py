import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    DEFAULT_TICKER = "SPY"
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    FRED_RISK_FREE_SERIES = "DGS3MO"  # 3-Month Treasury Rate
    DATA_DIR = "data"  # Directory for storing historical data
    DAYS_PER_YEAR = 365.25
    MIN_VOLUME = 1
    MIN_OPEN_INTEREST = 2
    MAX_SPREAD_PCT = 0.20
    MAX_SPREAD_ABS = 2.0
    MIN_DTE = 1
    MAX_DTE = 365
    MONEYNESS_MIN = 0.5
    MONEYNESS_MAX = 2.0
    IV_MIN = 0.03
    IV_MAX = 8.0
    IQR_MULTIPLIER = 1.5
    MIN_OPTIONS_PER_DTE = 3
    DBSCAN_EPS = 0.5
    DBSCAN_MIN_SAMPLES = 3
    NORMALIZE_DTE_MAX = 365

config = Config() 