"""Tests for the BSM pricing module."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.bsm import calculate_time_to_expiry, calculate_bsm_price_vectorized
from src.config import config
from src.providers.fred import get_risk_free_rate

def test_calculate_time_to_expiry():
    """Test time to expiry calculation."""
    today = date.today()
    expiry = today + timedelta(days=30)
    
    tte = calculate_time_to_expiry(expiry)
    assert abs(tte - 30/config.DAYS_PER_YEAR) < 1e-6
    
    # Test with pandas timestamp
    tte = calculate_time_to_expiry(pd.Timestamp(expiry))
    assert abs(tte - 30/config.DAYS_PER_YEAR) < 1e-6

def test_calculate_bsm_price():
    """Test BSM price calculation."""
    # Create sample data
    data = {
        "expiration_date": [date.today() + timedelta(days=30)],
        "strike": [100.0],
        "option_type": ["call"],
        "mid_price": [5.0],
        "implied_volatility": [0.20],  # Add required IV column
    }
    df = pd.DataFrame(data)
    
    # Calculate BSM prices using dynamic risk-free rate
    result = calculate_bsm_price_vectorized(
        df,
        underlying_price=100.0,
        risk_free_rate=get_risk_free_rate(),  # Use the actual function
    )
    
    # Check required columns
    required_cols = [
        "time_to_expiry",
        "bsm_price",
    ]
    for col in required_cols:
        assert col in result.columns
    
    # Check values
    assert not result["time_to_expiry"].isna().any()
    assert not result["bsm_price"].isna().any()
    
    # Check time to expiry
    assert abs(result["time_to_expiry"].iloc[0] - 30/config.DAYS_PER_YEAR) < 1e-6 