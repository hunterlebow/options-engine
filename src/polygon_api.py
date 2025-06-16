"""Polygon.io API wrapper for options data retrieval."""

from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
from polygon import RESTClient
from tqdm import tqdm

from .config import config

def get_risk_free_rate() -> float:
    """Get the current risk-free rate from FRED or use constant value."""
    if config.RISK_FREE_RATE_SOURCE == "FRED":
        import pandas_datareader as pdr
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        try:
            # Get 3-month T-bill rate
            rf_rate = pdr.data.get_data_fred("TB3MS", start_date, end_date)
            return float(rf_rate.iloc[-1] / 100)  # Convert to decimal
        except Exception as e:
            print(f"Warning: Could not fetch FRED data: {e}. Using constant rate.")
            return config.CONSTANT_RF_RATE
    return config.CONSTANT_RF_RATE

def get_option_chain(
    symbol: str,
    min_dte: Optional[int] = None,
    max_dte: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retrieve options chain data from Polygon.io using the Option Chain Snapshot endpoint.
    
    Parameters
    ----------
    symbol : str
        The underlying symbol (e.g., 'SPY')
    min_dte : int, optional
        Minimum days to expiry
    max_dte : int, optional
        Maximum days to expiry
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing options chain data with columns:
        - expiration_date
        - strike
        - option_type
        - bid
        - ask
        - last_price
        - volume
        - open_interest
        - implied_volatility
        - delta, gamma, theta, vega (greeks)
    """
    min_dte = min_dte or config.DEFAULT_MIN_DAYS_TO_EXPIRY
    max_dte = max_dte or config.DEFAULT_MAX_DAYS_TO_EXPIRY
    
    client = RESTClient(config.POLYGON_API_KEY)
    
    # Get current date for DTE calculations
    today = datetime.now().date()
    
    try:
        # Use the Option Chain Snapshot endpoint - much more efficient!
        # Set up parameters for the API call
        params = {
            "order": "asc",
            "limit": 250,  # Maximum allowed
            "sort": "ticker",
        }
        
        response_iterator = client.list_snapshot_options_chain(symbol, params=params)
        
        # Convert iterator to list to check if we have results
        contracts = list(response_iterator)
        
        if not contracts:
            raise ValueError(f"No options data found for {symbol}")
        
        options_data = []
        skipped_contracts = 0
        
        for contract in tqdm(contracts, desc="Processing options contracts"):
            try:
                # Extract contract details
                if not hasattr(contract, 'details') or not contract.details:
                    print(f"Warning: No contract details found - skipping")
                    skipped_contracts += 1
                    continue
                
                details = contract.details
                
                # Parse expiration date and calculate DTE
                expiry_str = details.expiration_date
                expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                dte = (expiry_date - today).days
                
                # Filter by DTE
                if dte < min_dte or dte > max_dte:
                    continue
                
                # Extract pricing data from last_quote
                if not hasattr(contract, 'last_quote') or not contract.last_quote:
                    print(f"Warning: No quote data for {details.ticker} - skipping")
                    skipped_contracts += 1
                    continue
                
                quote = contract.last_quote
                
                # Handle the actual response structure - bid/ask are direct attributes
                bid_price = getattr(quote, 'bid', None)
                ask_price = getattr(quote, 'ask', None)
                
                # Validate pricing data
                if not bid_price or not ask_price or bid_price <= 0 or ask_price <= 0:
                    print(f"Warning: Invalid pricing for {details.ticker} (bid: {bid_price}, ask: {ask_price}) - skipping")
                    skipped_contracts += 1
                    continue
                
                # Extract other quote data - use midpoint if available, otherwise calculate
                last_price = getattr(quote, 'midpoint', None)
                if not last_price:
                    last_price = (bid_price + ask_price) / 2
                
                # Extract trade data if available
                volume = 0
                if hasattr(contract, 'last_trade') and contract.last_trade:
                    volume = getattr(contract.last_trade, 'size', 0)
                
                # Extract additional data
                open_interest = getattr(contract, 'open_interest', 0)
                implied_vol = getattr(contract, 'implied_volatility', None)
                
                # Extract Greeks if available
                greeks_data = {}
                if hasattr(contract, 'greeks') and contract.greeks:
                    greeks = contract.greeks
                    greeks_data = {
                        'delta': getattr(greeks, 'delta', None),
                        'gamma': getattr(greeks, 'gamma', None),
                        'theta': getattr(greeks, 'theta', None),
                        'vega': getattr(greeks, 'vega', None),
                    }
                else:
                    greeks_data = {
                        'delta': None,
                        'gamma': None,
                        'theta': None,
                        'vega': None,
                    }
                
                # Build the data record
                option_record = {
                    "expiration_date": expiry_date,
                    "strike": details.strike_price,
                    "option_type": details.contract_type.lower(),  # 'call' or 'put'
                    "bid": float(bid_price),
                    "ask": float(ask_price),
                    "last_price": float(last_price),
                    "volume": volume,
                    "open_interest": open_interest,
                    "implied_volatility": implied_vol,
                    "dte": dte,
                    **greeks_data
                }
                
                options_data.append(option_record)
                
            except Exception as e:
                print(f"Error processing contract: {e}")
                skipped_contracts += 1
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(options_data)
        
        if len(df) == 0:
            raise ValueError(f"No valid options data found for {symbol} within DTE range {min_dte}-{max_dte}")
        
        if skipped_contracts > 0:
            print(f"Successfully processed {len(df)} contracts, skipped {skipped_contracts} due to missing/invalid data")
        
        # Calculate mid price
        df["mid_price"] = (df["bid"] + df["ask"]) / 2
        
        # Sort by expiration and strike
        df = df.sort_values(['expiration_date', 'strike']).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to fetch options chain for {symbol}: {e}")

def get_underlying_price(symbol: str) -> float:
    """
    Get the current price of the underlying asset using Polygon.io.
    
    Parameters
    ----------
    symbol : str
        The underlying symbol (e.g., 'SPY')
        
    Returns
    -------
    float
        Current price of the underlying asset
    """
    client = RESTClient(config.POLYGON_API_KEY)
    
    try:
        # Get the last quote for the underlying asset
        quote = client.get_last_quote(symbol)
        
        if not quote:
            raise ValueError(f"No quote data found for {symbol}")
        
        # Calculate mid price from bid/ask
        if hasattr(quote, 'bid') and hasattr(quote, 'ask'):
            bid = quote.bid
            ask = quote.ask
            if bid and ask and bid > 0 and ask > 0:
                return (bid + ask) / 2
        
        # Fallback to bid_price/ask_price attributes if available
        if hasattr(quote, 'bid_price') and hasattr(quote, 'ask_price'):
            bid = quote.bid_price
            ask = quote.ask_price
            if bid and ask and bid > 0 and ask > 0:
                return (bid + ask) / 2
        
        # If bid/ask not available, try to get last trade price
        trade = client.get_last_trade(symbol)
        if trade and hasattr(trade, 'price') and trade.price > 0:
            return float(trade.price)
        
        raise ValueError(f"Unable to determine price for {symbol} - no valid bid/ask or trade data")
        
    except Exception as e:
        raise ValueError(f"Failed to fetch price for {symbol}: {e}") 