"""Volatility surface construction and visualization utilities."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

def build_surface(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a volatility surface dataset from options data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing options data with columns:
        - strike
        - dte (days to expiry)
        - implied_volatility
        - option_type
        
    Returns
    -------
    pd.DataFrame
        Surface data with moneyness calculations
    """
    # Check if DataFrame is empty
    if df is None or len(df) == 0:
        print("⚠️ Warning: Empty DataFrame provided to build_surface")
        return pd.DataFrame()
    
    # Create a copy to avoid modifying original
    surface_data = df.copy()
    
    # Filter out invalid data first
    surface_data = surface_data.dropna(subset=['strike', 'dte', 'implied_volatility'])
    
    if len(surface_data) == 0:
        print("⚠️ Warning: No valid data after filtering NaN values")
        return pd.DataFrame()
    
    # Calculate moneyness - need underlying price
    if 'underlying_price' in surface_data.columns:
        # Use actual underlying price
        surface_data['moneyness'] = surface_data['underlying_price'] / surface_data['strike']
    else:
        # Try to infer ATM strike as proxy for underlying price
        # Find the strike closest to the middle of the range
        min_strike = surface_data['strike'].min()
        max_strike = surface_data['strike'].max()
        estimated_underlying = (min_strike + max_strike) / 2
        
        print(f"ℹ️ No underlying_price found, estimating from strike range: ${estimated_underlying:.2f}")
        surface_data['moneyness'] = estimated_underlying / surface_data['strike']
    
    # Remove extreme outliers in IV
    q1 = surface_data['implied_volatility'].quantile(0.05)
    q3 = surface_data['implied_volatility'].quantile(0.95)
    surface_data = surface_data[
        (surface_data['implied_volatility'] >= q1) & 
        (surface_data['implied_volatility'] <= q3)
    ]
    
    if len(surface_data) == 0:
        print("⚠️ Warning: No data remaining after outlier removal")
        return pd.DataFrame()
    
    return surface_data

def plot_surface_3d(
    surface_data: pd.DataFrame,
    underlying_price: float,
    title: str = "SPY Options Implied Volatility Surface"
) -> go.Figure:
    """
    Create a 3D surface plot of implied volatilities.
    
    Parameters
    ----------
    surface_data : pd.DataFrame
        Surface data from build_surface()
    underlying_price : float
        Current underlying price for moneyness calculation
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Calculate proper moneyness
    surface_data = surface_data.copy()
    surface_data['moneyness'] = underlying_price / surface_data['strike']
    
    # Create grid for interpolation
    moneyness_range = np.linspace(
        surface_data['moneyness'].min(), 
        surface_data['moneyness'].max(), 
        50
    )
    dte_range = np.linspace(
        surface_data['dte'].min(), 
        surface_data['dte'].max(), 
        30
    )
    
    X, Y = np.meshgrid(moneyness_range, dte_range)
    
    # Interpolate implied volatilities
    points = np.column_stack((surface_data['moneyness'], surface_data['dte']))
    Z = griddata(
        points, 
        surface_data['implied_volatility'], 
        (X, Y), 
        method='cubic',
        fill_value=np.nan
    )
    
    # Create the surface plot
    fig = go.Figure()
    
    # Add surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        name='IV Surface',
        showscale=True,
        colorbar=dict(title="Implied Volatility (%)")
    ))
    
    # Add scatter points for actual data
    fig.add_trace(go.Scatter3d(
        x=surface_data['moneyness'],
        y=surface_data['dte'],
        z=surface_data['implied_volatility'],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            opacity=0.6
        ),
        name='Market Data',
        showlegend=True
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title="Moneyness (S/K)",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility (%)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def plot_smile(
    df: pd.DataFrame,
    expiry_date: pd.Timestamp,
    underlying_price: float,
    option_type: str = "call",
) -> go.Figure:
    """
    Plot the volatility smile for a specific expiry.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing options data
    expiry_date : pd.Timestamp
        Expiry date to plot
    underlying_price : float
        Current underlying price
    option_type : str
        Type of options to plot ('call' or 'put')
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Filter data
    mask = (
        (df["expiration_date"] == expiry_date)
        & (df["option_type"].str.lower() == option_type.lower())
    )
    plot_df = df[mask].sort_values("strike")
    
    if len(plot_df) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title=f"No data for {option_type} options on {expiry_date.date()}")
        return fig
    
    # Calculate moneyness
    plot_df = plot_df.copy()
    plot_df['moneyness'] = underlying_price / plot_df['strike']
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=plot_df["moneyness"],
            y=plot_df["implied_volatility"],
            mode="lines+markers",
            name=f"{option_type.title()} IV",
            line=dict(width=2),
            marker=dict(size=6)
        )
    )
    
    # Add ATM line
    fig.add_vline(
        x=1.0, 
        line_dash="dash", 
        line_color="red",
        annotation_text="ATM"
    )
    
    fig.update_layout(
        title=f"Volatility Smile - {option_type.title()} Options - {expiry_date.date()}",
        xaxis_title="Moneyness (S/K)",
        yaxis_title="Implied Volatility (%)",
        showlegend=True,
        width=800,
        height=500
    )
    
    return fig

def analyze_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the volatility term structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Options data
        
    Returns
    -------
    pd.DataFrame
        Term structure analysis
    """
    # Group by DTE and calculate statistics
    term_structure = df.groupby('dte')['implied_volatility'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    
    term_structure.columns = ['Count', 'Mean_IV', 'Std_IV', 'Min_IV', 'Max_IV']
    
    return term_structure

def calculate_skew_metrics(df: pd.DataFrame, underlying_price: float) -> dict:
    """
    Calculate volatility skew metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Options data
    underlying_price : float
        Current underlying price
        
    Returns
    -------
    dict
        Skew metrics
    """
    df = df.copy()
    df['moneyness'] = underlying_price / df['strike']
    
    # Define moneyness buckets
    otm_puts = df[(df['option_type'] == 'put') & (df['moneyness'] > 1.05)]
    atm_options = df[abs(df['moneyness'] - 1.0) < 0.05]
    otm_calls = df[(df['option_type'] == 'call') & (df['moneyness'] < 0.95)]
    
    metrics = {
        'otm_put_iv': otm_puts['implied_volatility'].mean() if len(otm_puts) > 0 else np.nan,
        'atm_iv': atm_options['implied_volatility'].mean() if len(atm_options) > 0 else np.nan,
        'otm_call_iv': otm_calls['implied_volatility'].mean() if len(otm_calls) > 0 else np.nan,
    }
    
    # Calculate skew (put premium over calls)
    if not np.isnan(metrics['otm_put_iv']) and not np.isnan(metrics['otm_call_iv']):
        metrics['put_call_skew'] = metrics['otm_put_iv'] - metrics['otm_call_iv']
    else:
        metrics['put_call_skew'] = np.nan
        
    return metrics 