"""Tests for the BSM pricing module."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.bsm_pricing import calculate_time_to_expiry, calculate_bsm_price

def test_calculate_time_to_expiry():
    """Test time to expiry calculation."""
    # Test with date object
    expiry = date.today() + timedelta(days=30)
    tte = calculate_time_to_expiry(expiry)
    assert abs(tte - 30/365) < 1e-6
    
    # Test with pd.Timestamp
    expiry = pd.Timestamp.now() + pd.Timedelta(days=30)
    tte = calculate_time_to_expiry(expiry)
    assert abs(tte - 30/365) < 1e-6

def test_calculate_bsm_price():
    """Test BSM price calculation."""
    # Create sample data
    data = {
        "expiration_date": [date.today() + timedelta(days=30)],
        "strike": [100.0],
        "option_type": ["call"],
        "mid_price": [5.0],
    }
    df = pd.DataFrame(data)
    
    # Calculate BSM prices
    result = calculate_bsm_price(
        df,
        underlying_price=100.0,
        risk_free_rate=0.05,
    )
    
    # Check required columns
    required_cols = [
        "time_to_expiry",
        "bsm_price",
        "implied_volatility",
    ]
    for col in required_cols:
        assert col in result.columns
    
    # Check values
    assert not result["time_to_expiry"].isna().any()
    assert not result["bsm_price"].isna().any()
    assert not result["implied_volatility"].isna().any()
    
    # Check time to expiry
    assert abs(result["time_to_expiry"].iloc[0] - 30/365) < 1e-6 