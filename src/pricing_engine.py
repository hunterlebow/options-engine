"""Model-based option pricing engine."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
import warnings
import logging

from .bsm import calculate_bsm_price_vectorized, calculate_greeks
from .models import SABRModel, QuadraticSmile as QuadraticSmileModel, CubicSmile as CubicSmileModel
from .providers.polygon import get_underlying_price, get_dividend_yield
from .config import config
from .providers.fred import get_risk_free_rate
from .data import HistoricalDataCollector

logger = logging.getLogger(__name__)

def price_with_models(df: pd.DataFrame, 
                     underlying_price: Optional[float] = None,
                     model_type: str = 'sabr',
                     symbol: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Price options using fitted volatility models.
    
    DEPRECATED: This function is replaced by the PricingEngine class.
    """
    warnings.warn(
        "price_with_models is deprecated and will be removed in a future version. "
        "Use the PricingEngine class instead.",
        DeprecationWarning
    )
    return df, {}


def validate_models(
    df: pd.DataFrame, 
    underlying_price: float, 
    symbol: str
) -> Dict[str, Dict]:
    """
    Validate SABR, quadratic, and cubic models against market data.
    
    DEPRECATED: This function is replaced by the PricingEngine class.
    """
    warnings.warn(
        "validate_models is deprecated and will be removed in a future version. "
        "Use the PricingEngine class instead.",
        DeprecationWarning
    )
    return {}


def compare_models(df: pd.DataFrame,
                  underlying_price: Optional[float] = None,
                  symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Compare different volatility models side by side.
    
    DEPRECATED: This function is replaced by the PricingEngine class.
    """
    warnings.warn(
        "compare_models is deprecated and will be removed in a future version. "
        "Use the PricingEngine class instead.",
        DeprecationWarning
    )
    return pd.DataFrame()


def get_best_model(df: pd.DataFrame,
                   underlying_price: Optional[float] = None,
                   symbol: Optional[str] = None,
                   metric: str = 'rmse_price') -> Tuple[str, Dict]:
    """
    Find the best performing volatility model.
    
    DEPRECATED: This function is replaced by the PricingEngine class.
    """
    warnings.warn(
        "get_best_model is deprecated and will be removed in a future version. "
        "Use the PricingEngine class instead.",
        DeprecationWarning
    )
    return "", {}


class PricingEngine:
    """
    The core engine for option pricing model validation and analysis.
    """
    def __init__(self):
        self.data_collector = HistoricalDataCollector(config.DATA_DIR)
        
        # Instantiate the refactored models
        self.models = {
            'sabr': SABRModel(),
            'quadratic': QuadraticSmileModel(),
            'cubic': CubicSmileModel()
        }
        logger.info("PricingEngine initialized with refactored volatility models.")

    def validate_models(self, symbol: str, date: str, 
                        min_dte: int, max_dte: int,
                        models: List[str]) -> Dict[str, Any]:
        """
        Validate specified volatility models against market data for a single day.
        
        This method performs an in-sample validation by:
        1. Loading the option chain for the given day.
        2. Filtering the data based on provided criteria.
        3. Calibrating each specified model to the filtered data.
        4. Predicting prices using the calibrated model.
        5. Calculating performance metrics (RMSE, MAE) against market prices.
        """
        logger.info(f"Starting validation for {symbol} on {date} with models: {models}")
        
        # Load data
        options_data = self.data_collector.load_daily_data(symbol, date)
        if options_data is None or options_data.empty:
            raise ValueError(f"No data available for {symbol} on {date}")
        
        total_options = len(options_data)

        # Filter data
        filtered_data = self._filter_options(options_data, min_dte, max_dte)
        if filtered_data.empty:
            logger.warning("No options left after filtering.")
            return self._empty_validation_result(total_options)
            
        # Validate each model
        performance_results = {}
        for model_name in models:
            if model_name not in self.models:
                logger.warning(f"Model '{model_name}' not recognized. Skipping.")
                continue
            
            model = self.models[model_name]
            
            try:
                # 1. Calibrate model
                params = model.calibrate(filtered_data)
                if not params:
                    raise RuntimeError("Calibration failed.")
                
                # 2. Predict prices (in-sample)
                predicted_prices = model.predict(filtered_data, params)
                
                # 3. Calculate metrics
                market_prices = filtered_data['mid_price'].values
                
                rmse = np.sqrt(np.mean((predicted_prices - market_prices)**2))
                mae = np.mean(np.abs(predicted_prices - market_prices))
                
                # R-squared
                ss_res = np.sum((market_prices - predicted_prices)**2)
                ss_tot = np.sum((market_prices - np.mean(market_prices))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                performance_results[model_name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
                
            except Exception as e:
                logger.error(f"Validation failed for {model_name}: {e}")
                performance_results[model_name] = {'error': str(e)}

        return {
            'model_performance': performance_results,
            'total_options': total_options,
            'filtered_options': len(filtered_data)
        }

    def _filter_options(self, df: pd.DataFrame, min_dte: int, max_dte: int) -> pd.DataFrame:
        """Applies basic filtering to the options data."""
        return df[
            (df['dte'] >= min_dte) & 
            (df['dte'] <= max_dte) &
            (df['bid'] > 0) & 
            (df['ask'] > 0)
        ].copy()

    def _empty_validation_result(self, total_options: int) -> Dict:
        """Returns a default result structure when validation can't run."""
        return {
            'model_performance': {},
            'total_options': total_options,
            'filtered_options': 0
        } 