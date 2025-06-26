import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta

from ..config import config

def get_risk_free_rate() -> float:
    """Get current risk-free rate from FRED API as a decimal."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    rates = web.DataReader(
        config.FRED_RISK_FREE_SERIES,
        'fred',
        start_date,
        end_date,
        api_key=config.FRED_API_KEY
    )
    latest_rate = rates.iloc[-1, 0]
    return latest_rate / 100.0 