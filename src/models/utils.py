"""
Utility functions for volatility model implementations.
"""
import pandas as pd
import numpy as np
from py_vollib.black_scholes.implied_volatility import implied_volatility as py_vollib_iv
import logging

from ..providers.fred import get_risk_free_rate

logger = logging.getLogger(__name__)

def calculate_implied_volatility(options_data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Calculate implied volatility for each option in the DataFrame.

    This function uses the py_vollib library to compute the implied volatility
    based on the option's market price.

    Parameters
    ----------
    options_data : pd.DataFrame
        DataFrame with option data. Must contain 'mid_price', 'underlying_price',
        'strike', 'dte', and 'option_type'.
    inplace : bool, optional
        If True, modifies the DataFrame in place, by default False.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an added 'implied_vol' column.
    """
    df = options_data if inplace else options_data.copy()
    
    risk_free_rate = get_risk_free_rate()
    
    implied_vols = []
    for _, row in df.iterrows():
        try:
            # py_vollib expects 'c' or 'p' for the flag
            flag = row['option_type'].lower()[0]
            
            iv = py_vollib_iv(
                price=row['mid_price'],
                S=row['underlying_price'],
                K=row['strike'],
                t=row['dte'] / 365.0,
                r=risk_free_rate,
                flag=flag
            )
            implied_vols.append(iv)
        except Exception:
            implied_vols.append(np.nan)
            
    df['implied_vol'] = implied_vols
    
    # Drop rows where IV calculation failed
    initial_count = len(df)
    df.dropna(subset=['implied_vol'], inplace=True)
    final_count = len(df)
    
    if initial_count > final_count:
        logger.debug(f"Dropped {initial_count - final_count} rows due to failed IV calculation.")
        
    return df 