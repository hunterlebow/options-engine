"""Polygon.io API wrapper for options data."""
from datetime import datetime, timedelta
import pandas as pd
from polygon import RESTClient
from ..config import config

def get_dividend_yield(symbol: str) -> float:
    """Get dividend yield from Polygon API."""
    client = RESTClient(config.POLYGON_API_KEY)
    
    try:
        # Get last 12 months of dividends
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        
        dividends_response = client.list_dividends(
            ticker=symbol,
            ex_dividend_date_gte=start_date.strftime("%Y-%m-%d"),
            ex_dividend_date_lte=end_date.strftime("%Y-%m-%d"),
            limit=50
        )
        
        dividends = list(dividends_response)
        
        if not dividends:
            return 0.0  # No dividends = 0% yield
        
        total_dividends = sum(div.cash_amount for div in dividends if hasattr(div, 'cash_amount'))
        current_price = get_underlying_price(symbol)
        
        return total_dividends / current_price if current_price > 0 else 0.0
        
    except Exception:
        return 0.0  # Simple fallback

def get_underlying_price(symbol: str) -> float:
    """Get current stock price from Polygon."""
    client = RESTClient(config.POLYGON_API_KEY)
    
    try:
        ticker = client.get_last_trade(symbol)
        return float(ticker.price)
    except Exception:
        return 100.0  # Simple fallback for demos

def get_option_chain(symbol: str, min_dte: int = None, max_dte: int = None) -> pd.DataFrame:
    """Get options chain from Polygon."""
    client = RESTClient(config.POLYGON_API_KEY)
    
    min_dte = min_dte or config.MIN_DTE
    max_dte = max_dte or config.MAX_DTE
    
    today = datetime.now().date()
    
    try:
        response_iterator = client.list_snapshot_options_chain(
            symbol, 
            params={"order": "asc", "limit": 250, "sort": "ticker"}
        )
        
        contracts = list(response_iterator)
        options_data = []
        
        for contract in contracts:
            if not (hasattr(contract, 'details') and contract.details):
                continue
                
            details = contract.details
            
            # Calculate DTE
            expiry_date = datetime.strptime(details.expiration_date, "%Y-%m-%d").date()
            dte = (expiry_date - today).days
            
            if not (min_dte <= dte <= max_dte):
                continue
                
            if not (hasattr(contract, 'last_quote') and contract.last_quote):
                continue
                
            quote = contract.last_quote
            bid_price = getattr(quote, 'bid', 0)
            ask_price = getattr(quote, 'ask', 0)
            
            if bid_price <= 0 or ask_price <= 0:
                continue
                
            # Extract basic data
            option_data = {
                'ticker': details.ticker,
                'expiration_date': details.expiration_date,
                'strike': details.strike_price,
                'option_type': details.contract_type.lower(),
                'bid': bid_price,
                'ask': ask_price,
                'mid_price': (bid_price + ask_price) / 2,
                'dte': dte,
                'volume': getattr(contract.last_trade, 'size', 0) if hasattr(contract, 'last_trade') else 0,
                'open_interest': getattr(contract, 'open_interest', 0),
                'implied_volatility': getattr(contract, 'implied_volatility', 0.20),  # 20% default
            }
            
            # Add Greeks if available
            if hasattr(contract, 'greeks') and contract.greeks:
                greeks = contract.greeks
                option_data.update({
                    'delta': getattr(greeks, 'delta', None),
                    'gamma': getattr(greeks, 'gamma', None),
                    'theta': getattr(greeks, 'theta', None),
                    'vega': getattr(greeks, 'vega', None),
                })
            
            options_data.append(option_data)
        
        return pd.DataFrame(options_data)
        
    except Exception as e:
        print(f"Error fetching options data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error 