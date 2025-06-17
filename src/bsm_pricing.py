"""Black-Scholes-Merton option pricing implementation."""

from datetime import date
from typing import Optional, Union

import numpy as np
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes_merton import black_scholes_merton as bsm

try:
    from .polygon_api import get_risk_free_rate, get_underlying_price, get_dividend_yield
except ImportError:
    # Handle case when module is imported directly
    from polygon_api import get_risk_free_rate, get_underlying_price, get_dividend_yield

def calculate_time_to_expiry(expiry_date: Union[date, pd.Timestamp]) -> float:
    """Calculate time to expiry in years."""
    today = pd.Timestamp.now().date()
    days_to_expiry = (expiry_date - today).days
    return days_to_expiry / 365.0

def calculate_bsm_price_scalar(
    S: float,
    K: float, 
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: Optional[float] = None,  # Now optional - will fetch dynamically if None
    symbol: Optional[str] = None  # Symbol for dynamic dividend yield lookup
) -> float:
    """
    Calculate BSM price for a single option.
    
    Parameters
    ----------
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to expiry in years
    r : float
        Risk-free rate
    sigma : float
        Volatility (as decimal, e.g., 0.20 for 20%)
    option_type : str
        'call' or 'put'
    q : float, optional
        Dividend yield (if None, will fetch dynamically using symbol)
    symbol : str, optional
        Symbol for dynamic dividend yield lookup (required if q is None)
        
    Returns
    -------
    float
        BSM option price
    """
    # Handle dividend yield
    if q is None:
        if symbol is None:
            print("Warning: No dividend yield or symbol provided. Using 0% dividend yield.")
            q = 0.0
        else:
            try:
                q = get_dividend_yield(symbol)
            except Exception as e:
                print(f"Warning: Could not fetch dividend yield for {symbol}: {e}. Using 0%.")
                q = 0.0
    
    try:
        return bsm(
            option_type.lower()[0],  # 'c' or 'p'
            S,
            K, 
            T,
            r,
            sigma,
            q  # dividend yield
        )
    except Exception as e:
        raise ValueError(f"BSM calculation failed: {str(e)}")

# Alias for backward compatibility with notebook
calculate_bsm_price = calculate_bsm_price_scalar

def calculate_bsm_price_dataframe(
    df: pd.DataFrame,
    underlying_price: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    dividend_yield: Optional[float] = None,
    symbol: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate BSM prices for a DataFrame of options.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing options data with columns:
        - expiration_date
        - strike
        - option_type
        - mid_price
    underlying_price : float, optional
        Current price of the underlying. If None, fetched from API.
    risk_free_rate : float, optional
        Risk-free rate. If None, fetched from FRED.
    dividend_yield : float, optional
        Dividend yield. If None, fetched dynamically using symbol.
    symbol : str, optional
        Symbol for dynamic data fetching (required if other params are None)
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - time_to_expiry
        - bsm_price
        - implied_volatility
    """
    # Get market data if not provided
    if underlying_price is None:
        if symbol is None:
            raise ValueError("Either underlying_price or symbol must be provided")
        underlying_price = get_underlying_price(symbol)
        
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate()
        
    if dividend_yield is None:
        if symbol is None:
            print("Warning: No dividend yield or symbol provided. Using 0% dividend yield.")
            dividend_yield = 0.0
        else:
            dividend_yield = get_dividend_yield(symbol)
    
    # Calculate time to expiry
    df["time_to_expiry"] = df["expiration_date"].apply(calculate_time_to_expiry)
    
    # Calculate implied volatility
    def calc_iv(row):
        try:
            return implied_volatility(
                row["mid_price"],
                underlying_price,
                row["strike"],
                row["time_to_expiry"],
                risk_free_rate,
                row["option_type"].lower(),
            )
        except:
            return np.nan
    
    df["implied_volatility"] = df.apply(calc_iv, axis=1)
    
    # Calculate BSM price
    def calc_bsm(row):
        try:
            return bsm(
                row["option_type"].lower()[0],  # 'c' or 'p'
                underlying_price,
                row["strike"],
                row["time_to_expiry"],
                risk_free_rate,
                row["implied_volatility"],
                dividend_yield  # Now uses dynamic dividend yield
            )
        except:
            return np.nan
    
    df["bsm_price"] = df.apply(calc_bsm, axis=1)
    
    return df

def calculate_greeks(
    df: pd.DataFrame,
    underlying_price: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate option Greeks using BSM model.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing options data with BSM prices
    underlying_price : float, optional
        Current price of the underlying
    risk_free_rate : float, optional
        Risk-free rate
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - delta
        - gamma
        - theta
        - vega
        - rho
    """
    # TODO: Implement Greeks calculation
    # This will be implemented in a future version
    return df 