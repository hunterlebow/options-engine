"""Volatility surface construction and visualization utilities."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import json
from datetime import datetime

class SurfaceMetadata:
    """
    Comprehensive metadata tracking for volatility surface construction.
    Ensures full auditability and reproducibility for institutional use.
    """
    
    def __init__(self, symbol: str, underlying_price: float):
        self.metadata = {
            "construction_timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "underlying_price": underlying_price,
            "data_pipeline": {
                "input_data": {},
                "filtering_steps": [],
                "outlier_detection": [],
                "interpolation": {},
                "surface_construction": {}
            },
            "quality_metrics": {},
            "configuration": {},
            "warnings": [],
            "version_info": {
                "surface_utils_version": "2.0.0",
                "methodology": "institutional_grade"
            }
        }
    
    def log_input_data(self, df: pd.DataFrame):
        """Log initial dataset characteristics"""
        self.metadata["data_pipeline"]["input_data"] = {
            "total_contracts": len(df),
            "call_options": len(df[df['option_type'] == 'call']) if 'option_type' in df.columns else 0,
            "put_options": len(df[df['option_type'] == 'put']) if 'option_type' in df.columns else 0,
            "strike_range": [float(df['strike'].min()), float(df['strike'].max())] if 'strike' in df.columns else None,
            "dte_range": [int(df['dte'].min()), int(df['dte'].max())] if 'dte' in df.columns else None,
            "iv_coverage": float(df['implied_volatility'].notna().mean()) if 'implied_volatility' in df.columns else 0,
            "volume_coverage": float(df['volume'].notna().mean()) if 'volume' in df.columns else 0,
            "oi_coverage": float(df['open_interest'].notna().mean()) if 'open_interest' in df.columns else 0
        }
    
    def log_filter_step(self, step_name: str, filter_applied: bool, points_before: int, points_after: int, **kwargs):
        """Log each filtering step with detailed parameters"""
        filter_info = {
            "step": step_name,
            "applied": filter_applied,
            "points_before": points_before,
            "points_after": points_after,
            "retention_rate": points_after / points_before if points_before > 0 else 0,
            "parameters": kwargs,
            "timestamp": datetime.now().isoformat()
        }
        self.metadata["data_pipeline"]["filtering_steps"].append(filter_info)
    
    def log_outlier_detection(self, method: str, points_before: int, points_after: int, **kwargs):
        """Log outlier detection methods and results"""
        outlier_info = {
            "method": method,
            "points_before": points_before,
            "points_after": points_after,
            "outliers_removed": points_before - points_after,
            "outlier_rate": (points_before - points_after) / points_before if points_before > 0 else 0,
            "parameters": kwargs
        }
        self.metadata["data_pipeline"]["outlier_detection"].append(outlier_info)
    
    def log_interpolation(self, method: str, grid_size: tuple, smoothing_applied: bool, **kwargs):
        """Log interpolation method and parameters"""
        self.metadata["data_pipeline"]["interpolation"] = {
            "primary_method": method,
            "grid_dimensions": grid_size,
            "smoothing_applied": smoothing_applied,
            "parameters": kwargs
        }
    
    def log_surface_construction(self, final_points: int, moneyness_type: str, **kwargs):
        """Log final surface construction details"""
        self.metadata["data_pipeline"]["surface_construction"] = {
            "final_data_points": final_points,
            "moneyness_calculation": moneyness_type,
            "construction_successful": final_points > 0,
            "parameters": kwargs
        }
    
    def log_quality_metrics(self, **metrics):
        """Log surface quality metrics"""
        self.metadata["quality_metrics"].update(metrics)
    
    def log_configuration(self, config: dict):
        """Log configuration parameters"""
        self.metadata["configuration"].update(config)
    
    def add_warning(self, warning: str):
        """Add warning message"""
        self.metadata["warnings"].append({
            "message": warning,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_summary(self) -> dict:
        """Get concise summary for display"""
        pipeline = self.metadata["data_pipeline"]
        return {
            "symbol": self.metadata["symbol"],
            "total_filters_applied": len(pipeline["filtering_steps"]),
            "outlier_methods": [od["method"] for od in pipeline["outlier_detection"]],
            "final_data_points": pipeline["surface_construction"].get("final_data_points", 0),
            "interpolation_method": pipeline["interpolation"].get("primary_method", "none"),
            "moneyness_type": pipeline["surface_construction"].get("moneyness_calculation", "unknown"),
            "construction_timestamp": self.metadata["construction_timestamp"]
        }
    
    def export_metadata(self) -> str:
        """Export full metadata as JSON string"""
        return json.dumps(self.metadata, indent=2, default=str)

def build_surface(df: pd.DataFrame, symbol: str = "UNKNOWN", underlying_price: float = 100.0) -> tuple[pd.DataFrame, SurfaceMetadata]:
    """
    Build volatility surface data with enhanced noise reduction and comprehensive metadata tracking.
    
    Parameters
    ----------
    df : pd.DataFrame
        Options data with columns: strike, dte, implied_volatility, option_type
    symbol : str
        Symbol for metadata tracking
    underlying_price : float
        Underlying price for metadata tracking
        
    Returns
    -------
    tuple[pd.DataFrame, SurfaceMetadata]
        Surface data and comprehensive metadata
    """
    # Initialize metadata tracking
    metadata = SurfaceMetadata(symbol, underlying_price)
    
    if len(df) == 0:
        metadata.add_warning("Empty DataFrame provided")
        return pd.DataFrame(), metadata
    
    # Log input data characteristics
    metadata.log_input_data(df)
    
    # Create a copy for processing
    surface_df = df.copy()
    initial_count = len(surface_df)
    
    # Ensure we have required columns
    required_cols = ['strike', 'dte', 'implied_volatility']
    if not all(col in surface_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in surface_df.columns]
        metadata.add_warning(f"Missing required columns: {missing_cols}")
        return pd.DataFrame(), metadata
    
    # Calculate moneyness - use forward price if available, otherwise use underlying price
    moneyness_type = "unknown"
    if 'forward_price' in surface_df.columns:
        surface_df['moneyness'] = surface_df['strike'] / surface_df['forward_price']
        moneyness_type = "K/F (forward-based)"
        print("Using forward-based moneyness calculation")
    elif 'underlying_price' in surface_df.columns:
        surface_df['moneyness'] = surface_df['underlying_price'] / surface_df['strike']
        moneyness_type = "S/K (spot-based)"
        print("Using spot-based moneyness calculation")
    else:
        # Try to infer underlying price from strike distribution
        mid_strike = surface_df['strike'].median()
        surface_df['moneyness'] = mid_strike / surface_df['strike']
        moneyness_type = f"estimated/K (median strike: ${mid_strike:.2f})"
        metadata.add_warning(f"No underlying price found, using median strike ${mid_strike:.2f} for moneyness")
        print(f"Warning: No underlying price found, using median strike ${mid_strike:.2f} for moneyness")
    
    # Statistical outlier detection using IQR method
    print("Applying statistical outlier detection...")
    pre_iqr_count = len(surface_df)
    
    Q1 = surface_df['implied_volatility'].quantile(0.25)
    Q3 = surface_df['implied_volatility'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_filter = (
        (surface_df['implied_volatility'] >= lower_bound) & 
        (surface_df['implied_volatility'] <= upper_bound)
    )
    
    surface_df = surface_df[outlier_filter].copy()
    post_iqr_count = len(surface_df)
    
    metadata.log_outlier_detection(
        "IQR", pre_iqr_count, post_iqr_count,
        Q1=float(Q1), Q3=float(Q3), IQR=float(IQR),
        lower_bound=float(lower_bound), upper_bound=float(upper_bound)
    )
    
    print(f"   IQR outlier filter: {post_iqr_count:,} / {pre_iqr_count:,} options retained")
    
    if len(surface_df) == 0:
        metadata.add_warning("No data remaining after IQR outlier filtering")
        return pd.DataFrame(), metadata
    
    # Spatial outlier detection using DBSCAN clustering
    if len(surface_df) >= 10:  # Need minimum points for clustering
        print("Applying spatial outlier detection...")
        pre_dbscan_count = len(surface_df)
        
        # Prepare features for clustering (moneyness, time, IV)
        features = surface_df[['moneyness', 'dte', 'implied_volatility']].copy()
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply DBSCAN to identify outliers
        eps = 0.5
        min_samples = 3
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(features_scaled)
        
        # Keep only points in clusters (not outliers marked as -1)
        cluster_filter = clusters != -1
        surface_df = surface_df[cluster_filter].copy()
        post_dbscan_count = len(surface_df)
        
        metadata.log_outlier_detection(
            "DBSCAN", pre_dbscan_count, post_dbscan_count,
            eps=eps, min_samples=min_samples,
            n_clusters=len(set(clusters)) - (1 if -1 in clusters else 0),
            outliers_identified=np.sum(clusters == -1)
        )
        
        print(f"   Spatial outlier filter: {post_dbscan_count:,} / {pre_dbscan_count:,} options retained")
    else:
        metadata.add_warning("Insufficient data points for DBSCAN spatial outlier detection")
    
    # Sort by strike and DTE for better interpolation
    surface_df = surface_df.sort_values(['dte', 'strike']).reset_index(drop=True)
    final_count = len(surface_df)
    
    # Log surface construction details
    metadata.log_surface_construction(
        final_count, moneyness_type,
        data_reduction_ratio=final_count / initial_count if initial_count > 0 else 0,
        moneyness_range=[float(surface_df['moneyness'].min()), float(surface_df['moneyness'].max())] if final_count > 0 else None,
        iv_range=[float(surface_df['implied_volatility'].min()), float(surface_df['implied_volatility'].max())] if final_count > 0 else None
    )
    
    # Log quality metrics
    if final_count > 0:
        metadata.log_quality_metrics(
            moneyness_std=float(surface_df['moneyness'].std()),
            iv_std=float(surface_df['implied_volatility'].std()),
            dte_coverage=int(surface_df['dte'].max() - surface_df['dte'].min()),
            data_density=final_count / (surface_df['dte'].nunique() * surface_df['strike'].nunique()) if surface_df['dte'].nunique() > 0 and surface_df['strike'].nunique() > 0 else 0
        )
    
    print(f"Surface construction completed with {final_count} clean data points")
    
    return surface_df, metadata

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

def plot_surface_3d_enhanced(
    surface_data: pd.DataFrame,
    underlying_price: float,
    title: str = "Enhanced Implied Volatility Surface",
    metadata: SurfaceMetadata = None
) -> go.Figure:
    """
    Create a 3D surface plot with enhanced noise reduction, smoothing, and comprehensive metadata tracking.
    
    Parameters
    ----------
    surface_data : pd.DataFrame
        Surface data from build_surface()
    underlying_price : float
        Current underlying price for moneyness calculation
    title : str
        Plot title
    metadata : SurfaceMetadata
        Metadata object for tracking interpolation and surface construction
        
    Returns
    -------
    go.Figure
        Plotly figure object with enhanced surface and embedded metadata
    """
    # Initialize metadata if not provided
    if metadata is None:
        metadata = SurfaceMetadata("UNKNOWN", underlying_price)
    
    # Calculate proper moneyness using forward price if available
    surface_data = surface_data.copy()
    moneyness_type = "S/K (spot-based)"
    
    if 'forward_price' in surface_data.columns:
        surface_data['moneyness'] = surface_data['strike'] / surface_data['forward_price']
        moneyness_type = "K/F (forward-based)"
        print("Using forward-based moneyness for cleaner surface")
    else:
        surface_data['moneyness'] = underlying_price / surface_data['strike']
    
    # Check if we have enough data points for surface interpolation
    num_points = len(surface_data)
    
    fig = go.Figure()
    
    if num_points < 4:
        # Insufficient data for surface interpolation - create enhanced scatter plot
        print(f"Warning: Only {num_points} data points available. Creating enhanced scatter plot.")
        metadata.add_warning(f"Insufficient data for surface interpolation: {num_points} points")
        
        # Log interpolation method
        metadata.log_interpolation(
            "scatter_plot", (0, 0), False,
            reason="insufficient_data_points",
            data_points=num_points
        )
        
        # Add scatter points with enhanced styling
        fig.add_trace(go.Scatter3d(
            x=surface_data['moneyness'],
            y=surface_data['dte'],
            z=surface_data['implied_volatility'],
            mode='markers+text',
            marker=dict(
                size=10,
                color=surface_data['implied_volatility'],
                colorscale='Viridis',
                colorbar=dict(title="Implied Volatility"),
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=[f"${row['strike']:.0f}<br>{row['implied_volatility']:.1%}" 
                  for _, row in surface_data.iterrows()],
            textposition="top center",
            name='Market Data',
            showlegend=True
        ))
        
        # Add connecting lines if we have exactly 2 points
        if num_points == 2:
            fig.add_trace(go.Scatter3d(
                x=surface_data['moneyness'],
                y=surface_data['dte'],
                z=surface_data['implied_volatility'],
                mode='lines',
                line=dict(width=6, color='red'),
                name='Data Connection',
                showlegend=True
            ))
    
    else:
        # Sufficient data for enhanced surface interpolation
        print(f"Creating enhanced surface with {num_points} data points")
        
        # Create adaptive grid based on data density
        grid_x = min(50, max(20, num_points * 2))  # Adaptive grid size
        grid_y = min(30, max(15, num_points))
        
        moneyness_range = np.linspace(
            surface_data['moneyness'].min() * 0.95, 
            surface_data['moneyness'].max() * 1.05, 
            grid_x
        )
        dte_range = np.linspace(
            surface_data['dte'].min() * 0.95, 
            surface_data['dte'].max() * 1.05, 
            grid_y
        )
        
        X, Y = np.meshgrid(moneyness_range, dte_range)
        
        # Enhanced interpolation with error handling and smoothing
        points = np.column_stack((surface_data['moneyness'], surface_data['dte']))
        
        interpolation_method = "none"
        smoothing_applied = False
        smoothing_sigma = 0.0
        
        try:
            # Try cubic interpolation first
            Z = griddata(
                points, 
                surface_data['implied_volatility'], 
                (X, Y), 
                method='cubic',
                fill_value=np.nan
            )
            interpolation_method = "cubic"
            
            # Apply Gaussian smoothing to reduce noise
            if not np.all(np.isnan(Z)):
                valid_mask = ~np.isnan(Z)
                if np.sum(valid_mask) > 10:
                    # Apply adaptive smoothing based on data density
                    smoothing_sigma = 0.8 if num_points < 20 else 0.5
                    Z_smooth = gaussian_filter(Z, sigma=smoothing_sigma, mode='nearest')
                    Z = np.where(valid_mask, Z_smooth, Z)
                    smoothing_applied = True
                    print(f"Applied Gaussian smoothing (σ={smoothing_sigma}) to reduce surface noise")
                    
        except Exception as e:
            print(f"Cubic interpolation failed: {e}")
            metadata.add_warning(f"Cubic interpolation failed: {str(e)}")
            try:
                # Fall back to linear interpolation
                Z = griddata(
                    points, 
                    surface_data['implied_volatility'], 
                    (X, Y), 
                    method='linear',
                    fill_value=np.nan
                )
                interpolation_method = "linear"
                print("Using linear interpolation")
                
                # Apply lighter smoothing to linear interpolation
                if not np.all(np.isnan(Z)):
                    valid_mask = ~np.isnan(Z)
                    if np.sum(valid_mask) > 5:
                        smoothing_sigma = 0.3
                        Z_smooth = gaussian_filter(Z, sigma=smoothing_sigma, mode='nearest')
                        Z = np.where(valid_mask, Z_smooth, Z)
                        smoothing_applied = True
                        
            except Exception as e2:
                print(f"Linear interpolation failed: {e2}")
                metadata.add_warning(f"Linear interpolation failed: {str(e2)}")
                # Fall back to nearest neighbor
                Z = griddata(
                    points, 
                    surface_data['implied_volatility'], 
                    (X, Y), 
                    method='nearest',
                    fill_value=np.nan
                )
                interpolation_method = "nearest_neighbor"
                print("Using nearest neighbor interpolation")
        
        # Log interpolation details
        metadata.log_interpolation(
            interpolation_method, (grid_x, grid_y), smoothing_applied,
            smoothing_sigma=smoothing_sigma,
            data_points=num_points,
            grid_density=grid_x * grid_y,
            valid_interpolated_points=int(np.sum(~np.isnan(Z))) if 'Z' in locals() else 0
        )
        
        # Add enhanced surface with better styling
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            name='IV Surface',
            showscale=True,
            colorbar=dict(
                title="Implied Volatility",
                tickformat=".1%"
            ),
            opacity=0.8,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.1
            )
        ))
        
        # Add enhanced scatter points for actual data
        fig.add_trace(go.Scatter3d(
            x=surface_data['moneyness'],
            y=surface_data['dte'],
            z=surface_data['implied_volatility'],
            mode='markers',
            marker=dict(
                size=6,
                color='red',
                opacity=0.9,
                line=dict(width=1, color='white')
            ),
            name='Market Data',
            showlegend=True
        ))
    
    # Enhanced layout with better styling
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, family="Arial Black")
        ),
        scene=dict(
            xaxis_title="Moneyness (K/F)" if moneyness_type.startswith("K/F") else "Moneyness (S/K)",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor="rgba(240,240,240,0.1)",
            xaxis=dict(
                backgroundcolor="rgba(200,200,200,0.1)",
                gridcolor="rgba(150,150,150,0.3)",
                showbackground=True
            ),
            yaxis=dict(
                backgroundcolor="rgba(200,200,200,0.1)",
                gridcolor="rgba(150,150,150,0.3)",
                showbackground=True
            ),
            zaxis=dict(
                backgroundcolor="rgba(200,200,200,0.1)",
                gridcolor="rgba(150,150,150,0.3)",
                showbackground=True,
                tickformat=".1%"
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Embed metadata in the figure for full traceability
    metadata_summary = metadata.get_summary()
    fig.update_layout(
        annotations=[
            dict(
                text=f"Metadata Summary:<br>"
                     f"• Symbol: {metadata_summary['symbol']}<br>"
                     f"• Data points: {metadata_summary['final_data_points']}<br>"
                     f"• Interpolation: {metadata_summary['interpolation_method']}<br>"
                     f"• Outlier methods: {', '.join(metadata_summary['outlier_methods'])}<br>"
                     f"• Moneyness: {metadata_summary['moneyness_type']}<br>"
                     f"• Timestamp: {metadata_summary['construction_timestamp'][:19]}",
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                showarrow=False,
                font=dict(size=8),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1
            )
        ]
    )
    
    # Store full metadata in figure for programmatic access
    # Use a simple approach that works with all Plotly versions
    try:
        # Try to store metadata as a custom attribute
        fig._surface_metadata = metadata.export_metadata()
        fig._construction_summary = metadata_summary
    except Exception:
        # If that fails, just skip metadata storage
        pass
    
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

def calculate_liquidity_score(df: pd.DataFrame, underlying_price: float) -> pd.Series:
    """
    Calculate a composite liquidity score combining volume, open interest, and spread.
    Higher scores indicate better liquidity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Options data with volume, open_interest, bid, ask columns
    underlying_price : float
        Current underlying price for spread normalization
        
    Returns
    -------
    pd.Series
        Liquidity scores (0-100 scale)
    """
    # Initialize components
    volume_score = pd.Series(0.0, index=df.index)
    oi_score = pd.Series(0.0, index=df.index)
    spread_score = pd.Series(0.0, index=df.index)
    
    # Volume component (0-40 points)
    if 'volume' in df.columns:
        volume = df['volume'].fillna(0)
        if volume.max() > 0:
            # Log-scale normalization to handle extreme values
            volume_norm = np.log1p(volume) / np.log1p(volume.quantile(0.95))
            volume_score = np.clip(volume_norm * 40, 0, 40)
    
    # Open Interest component (0-40 points)
    if 'open_interest' in df.columns:
        oi = df['open_interest'].fillna(0)
        if oi.max() > 0:
            # Log-scale normalization
            oi_norm = np.log1p(oi) / np.log1p(oi.quantile(0.95))
            oi_score = np.clip(oi_norm * 40, 0, 40)
    
    # Spread component (0-20 points, inverted - tighter spreads get higher scores)
    if 'bid' in df.columns and 'ask' in df.columns:
        valid_spreads = (df['bid'] > 0) & (df['ask'] > df['bid'])
        spreads = df['ask'] - df['bid']
        
        # Normalize spreads by price level
        if underlying_price > 100:
            # Use percentage spreads for high-priced assets
            spread_pct = spreads / df['mid_price']
            # Invert: smaller spreads get higher scores
            spread_score = np.where(
                valid_spreads,
                np.clip(20 * (1 - np.clip(spread_pct / 0.20, 0, 1)), 0, 20),  # 20% spread = 0 points
                0
            )
        else:
            # Use dollar spreads for lower-priced assets
            # Invert: smaller spreads get higher scores
            spread_score = np.where(
                valid_spreads,
                np.clip(20 * (1 - np.clip(spreads / 5.0, 0, 1)), 0, 20),  # $5 spread = 0 points
                0
            )
    
    # Combine components
    liquidity_score = volume_score + oi_score + spread_score
    
    return liquidity_score

def get_yield_curve_rate(tenor_days: int) -> float:
    """
    Get risk-free rate from yield curve approximation.
    In production, this would pull from FRED, Bloomberg, or similar.
    
    Parameters
    ----------
    tenor_days : int
        Days to expiration
        
    Returns
    -------
    float
        Risk-free rate for the given tenor
    """
    # Simplified yield curve approximation
    # In production, use actual SOFR/Treasury curve
    base_rate = 0.0495  # Current base rate
    
    if tenor_days <= 30:
        return base_rate - 0.002  # Short-term discount
    elif tenor_days <= 90:
        return base_rate
    elif tenor_days <= 180:
        return base_rate + 0.001
    elif tenor_days <= 365:
        return base_rate + 0.003
    else:
        return base_rate + 0.005  # Long-term premium 