"""Option mispricing detection and analysis."""

from typing import Optional, Tuple

import pandas as pd
import numpy as np

from .config import config

def compute_mispricing(
    df: pd.DataFrame,
    underlying_price: float,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Identify mispriced options based on BSM model.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing options data with columns:
        - mid_price (or market_price)
        - bsm_price
        - strike
        - expiration_date
        - option_type
    underlying_price : float
        Current underlying price
    threshold : float, optional
        Threshold for mispricing detection (default from config)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with mispricing metrics added
    """
    threshold = threshold or config.MISPRICING_THRESHOLD
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Ensure we have the right price column name
    if 'market_price' in result_df.columns and 'mid_price' not in result_df.columns:
        result_df['mid_price'] = result_df['market_price']
    
    # Calculate mispricing metrics
    result_df["price_diff"] = result_df["bsm_price"] - result_df["mid_price"]
    result_df["price_diff_pct"] = (result_df["price_diff"] / result_df["mid_price"]) * 100
    result_df["mispricing_pct"] = abs(result_df["price_diff_pct"])
    
    # Add additional metrics
    result_df["mispricing_direction"] = result_df["price_diff"].apply(
        lambda x: "overpriced" if x < 0 else "underpriced"
    )
    
    # Calculate moneyness for analysis
    result_df["moneyness"] = underlying_price / result_df["strike"]
    
    # Add confidence score based on liquidity and spread
    if 'bid' in result_df.columns and 'ask' in result_df.columns:
        result_df['spread'] = result_df['ask'] - result_df['bid']
        result_df['spread_pct'] = (result_df['spread'] / result_df['mid_price']) * 100
        
        # Higher confidence for tighter spreads and higher volume
        result_df['confidence_score'] = 100 / (1 + result_df['spread_pct'])
        
        if 'volume' in result_df.columns:
            result_df['confidence_score'] *= np.log1p(result_df['volume'].fillna(0))
    else:
        result_df['confidence_score'] = 50  # Default confidence
    
    return result_df

def get_top_mispriced(
    df: pd.DataFrame,
    n: int = 10,
    direction: Optional[str] = None,
    min_confidence: float = 20.0
) -> pd.DataFrame:
    """
    Get top N most mispriced options.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing options with mispricing metrics
    n : int
        Number of options to return
    direction : str, optional
        Filter by direction ('overpriced' or 'underpriced')
    min_confidence : float
        Minimum confidence score to include
        
    Returns
    -------
    pd.DataFrame
        Top N mispriced options
    """
    # Filter by confidence if available
    if 'confidence_score' in df.columns:
        filtered_df = df[df['confidence_score'] >= min_confidence].copy()
    else:
        filtered_df = df.copy()
    
    # Filter by direction if specified
    if direction:
        filtered_df = filtered_df[filtered_df["mispricing_direction"] == direction]
    
    # Filter for significant mispricing (>3% difference)
    filtered_df = filtered_df[filtered_df["mispricing_pct"] > 3.0]
    
    # Sort by mispricing percentage (descending)
    filtered_df = filtered_df.sort_values("mispricing_pct", ascending=False)
    
    return filtered_df.head(n)

def analyze_mispricing_patterns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze patterns in mispriced options.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing options with mispricing metrics
        
    Returns
    -------
    pd.DataFrame
        Summary statistics of mispricing patterns
    """
    # Group by expiry and option type
    summary = df.groupby(["expiration_date", "option_type"]).agg({
        "mispricing_pct": ["mean", "std", "count"],
        "strike": ["min", "max"],
        "price_diff_pct": ["mean", "std"]
    })
    
    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    
    return summary

def identify_arbitrage_opportunities(
    df: pd.DataFrame,
    min_profit_pct: float = 5.0
) -> pd.DataFrame:
    """
    Identify potential arbitrage opportunities.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing options with mispricing metrics
    min_profit_pct : float
        Minimum profit percentage to consider
        
    Returns
    -------
    pd.DataFrame
        Potential arbitrage opportunities
    """
    # Look for significant mispricings with high confidence
    arbitrage_candidates = df[
        (df['mispricing_pct'] > min_profit_pct) &
        (df.get('confidence_score', 0) > 50)
    ].copy()
    
    if len(arbitrage_candidates) == 0:
        return pd.DataFrame()
    
    # Add strategy recommendations
    arbitrage_candidates['strategy'] = arbitrage_candidates.apply(
        lambda row: f"{'SELL' if row['price_diff'] < 0 else 'BUY'} "
                   f"{row['option_type'].upper()} "
                   f"${row['strike']:.0f} "
                   f"{row['expiration_date'].strftime('%m/%d')}",
        axis=1
    )
    
    # Calculate potential profit
    arbitrage_candidates['potential_profit_per_contract'] = abs(arbitrage_candidates['price_diff']) * 100
    
    return arbitrage_candidates.sort_values('mispricing_pct', ascending=False)

def calculate_portfolio_metrics(
    opportunities: pd.DataFrame,
    max_position_size: float = 10000  # Maximum $ per position
) -> dict:
    """
    Calculate portfolio-level metrics for mispricing opportunities.
    
    Parameters
    ----------
    opportunities : pd.DataFrame
        DataFrame containing mispricing opportunities
    max_position_size : float
        Maximum dollar amount per position
        
    Returns
    -------
    dict
        Portfolio metrics
    """
    if len(opportunities) == 0:
        return {
            'total_opportunities': 0,
            'total_potential_profit': 0,
            'avg_profit_per_trade': 0,
            'risk_score': 0
        }
    
    # Calculate position sizes (limited by max_position_size)
    opportunities = opportunities.copy()
    opportunities['position_value'] = opportunities['mid_price'] * 100  # Per contract
    opportunities['max_contracts'] = np.floor(max_position_size / opportunities['position_value'])
    opportunities['actual_contracts'] = np.minimum(opportunities['max_contracts'], 10)  # Max 10 contracts
    
    # Calculate total potential profit
    opportunities['total_profit'] = (
        opportunities['actual_contracts'] * 
        opportunities['potential_profit_per_contract']
    )
    
    metrics = {
        'total_opportunities': len(opportunities),
        'total_potential_profit': opportunities['total_profit'].sum(),
        'avg_profit_per_trade': opportunities['total_profit'].mean(),
        'avg_mispricing_pct': opportunities['mispricing_pct'].mean(),
        'risk_score': calculate_risk_score(opportunities)
    }
    
    return metrics

def calculate_risk_score(df: pd.DataFrame) -> float:
    """
    Calculate a risk score for the mispricing opportunities.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing opportunities
        
    Returns
    -------
    float
        Risk score (0-100, lower is better)
    """
    if len(df) == 0:
        return 100
    
    # Factors that increase risk:
    # 1. Short time to expiration
    # 2. Wide spreads
    # 3. Low volume
    # 4. Extreme moneyness
    
    risk_factors = []
    
    # Time risk (higher risk for shorter DTE)
    if 'dte' in df.columns:
        avg_dte = df['dte'].mean()
        time_risk = max(0, (30 - avg_dte) / 30 * 30)  # 0-30 points
        risk_factors.append(time_risk)
    
    # Spread risk
    if 'spread_pct' in df.columns:
        avg_spread = df['spread_pct'].mean()
        spread_risk = min(30, avg_spread * 3)  # 0-30 points
        risk_factors.append(spread_risk)
    
    # Liquidity risk
    if 'volume' in df.columns:
        avg_volume = df['volume'].fillna(0).mean()
        liquidity_risk = max(0, 20 - np.log1p(avg_volume) * 2)  # 0-20 points
        risk_factors.append(liquidity_risk)
    
    # Moneyness risk (higher risk for extreme OTM)
    if 'moneyness' in df.columns:
        avg_moneyness = df['moneyness'].mean()
        moneyness_risk = abs(1 - avg_moneyness) * 20  # 0-20 points
        risk_factors.append(moneyness_risk)
    
    total_risk = sum(risk_factors) if risk_factors else 50
    return min(100, max(0, total_risk)) 