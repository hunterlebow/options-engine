"""
Backtesting Engine for Options Pricing Models

This module provides a comprehensive backtesting framework for testing volatility models
and pricing strategies using historical options data with rolling window methodology.

Key Features:
- Rolling window calibration
- Out-of-sample testing
- Multiple volatility models (SABR, Quadratic, Cubic)
- Performance metrics and analysis
- Risk-adjusted returns calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from .models import SABRModel, QuadraticSmile as QuadraticSmileModel, CubicSmile as CubicSmileModel
from .data import HistoricalDataCollector
from .bsm import vectorized_bsm_price
from .providers.fred import get_risk_free_rate
from py_vollib.black_scholes.implied_volatility import implied_volatility

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    # Data parameters
    symbol: str
    start_date: str
    end_date: str
    
    # Rolling window parameters
    calibration_window: int = 30  # days
    rebalance_frequency: int = 5   # days
    
    # Model parameters
    models: List[str] = None  # ['sabr', 'quadratic', 'cubic']
    
    # Filtering parameters
    min_dte: int = 7
    max_dte: int = 90
    min_volume: int = 0
    max_spread_pct: float = 0.5
    
    # Performance parameters
    transaction_cost: float = 0.01  # $0.01 per contract
    risk_free_rate: Optional[float] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = ['sabr', 'quadratic', 'cubic']

@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    
    config: BacktestConfig
    
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    # Model comparison
    model_performance: Dict[str, Dict[str, float]]
    
    # Time series results
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    positions: pd.DataFrame
    
    # Detailed analysis
    trade_analysis: pd.DataFrame
    calibration_history: pd.DataFrame

class VolatilityBacktester:
    """
    Comprehensive backtesting engine for volatility models using a rolling window.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the backtesting engine.
        
        Parameters
        ----------
        data_dir : str
            Directory containing historical options data
        """
        self.data_dir = Path(data_dir)
        self.data_collector = HistoricalDataCollector(str(data_dir))
        
        # Instantiate the new, refactored model classes directly
        self.models = {
            'sabr': SABRModel(),
            'quadratic': QuadraticSmileModel(),
            'cubic': CubicSmileModel()
        }
        logger.info("Initialized VolatilityBacktester with refactored models.")
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Run a comprehensive backtest with rolling window methodology.
        
        Parameters
        ----------
        config : BacktestConfig
            Backtesting configuration
            
        Returns
        -------
        BacktestResult
            Comprehensive backtesting results
        """
        logger.info(f"Starting backtest for {config.symbol} from {config.start_date} to {config.end_date}")
        
        historical_data = self._load_historical_data(config)
        if historical_data.empty:
            raise ValueError(f"No historical data found for {config.symbol} in the given date range.")
        
        trading_dates = self._generate_trading_dates(historical_data, config)
        
        # Initialize tracking structures
        daily_pnl_history = []
        trade_history = []
        calibration_history = []
        
        # --- Rolling Window Backtesting Loop ---
        for current_date in trading_dates:
            logger.info(f"Processing trading date: {current_date.strftime('%Y-%m-%d')}")
            
            # 1. Define windows
            calib_end = current_date
            calib_start = calib_end - timedelta(days=config.calibration_window)
            oos_end = current_date + timedelta(days=config.rebalance_frequency)
            
            # 2. Get data for windows
            calib_data = self._get_data_for_period(historical_data, calib_start, calib_end, config)
            oos_data = self._get_data_for_period(historical_data, current_date, oos_end, config)
            
            if calib_data.empty or oos_data.empty:
                logger.warning(f"Skipping {current_date.strftime('%Y-%m-%d')}: Not enough data.")
                continue
            
            # 3. Calibrate models and generate signals
            signals = self._calibrate_and_predict(calib_data, oos_data, config)
            
            # 4. Execute trades and calculate PnL
            period_pnl, trades = self._execute_trades(signals, config)
            
            # 5. Update history
            daily_pnl_history.append({'date': current_date, 'pnl': period_pnl})
            trade_history.extend(trades)
            # (Calibration history can be added here if needed)

        return self._calculate_performance_metrics(config, daily_pnl_history, trade_history)

    def _calibrate_and_predict(self, calib_data, oos_data, config):
        """Calibrate models and generate trading signals."""
        signals = []
        for model_name in config.models:
            model = self.models.get(model_name)
            if not model:
                logger.warning(f"Model '{model_name}' not found. Skipping.")
                continue

            try:
                # Calibrate on historical window
                params = model.calibrate(calib_data)
                if not params:
                    logger.warning(f"Calibration failed for {model_name} on {calib_data['date'].min().strftime('%Y-%m-%d')}")
                    continue
                
                # Predict on out-of-sample window
                predicted_prices = model.predict(oos_data, params)
                
                # Generate signals
                oos_data_copy = oos_data.copy()
                oos_data_copy['predicted_price'] = predicted_prices
                oos_data_copy['price_diff_pct'] = (predicted_prices - oos_data_copy['mid_price']) / oos_data_copy['mid_price']
                
                # Signal logic: if predicted > market, buy. if predicted < market, sell.
                trade_threshold = 0.05  # 5% mispricing
                
                for _, row in oos_data_copy.iterrows():
                    if row['price_diff_pct'] > trade_threshold:
                        action = 'buy'
                    elif row['price_diff_pct'] < -trade_threshold:
                        action = 'sell'
                    else:
                        continue
                    
                    signals.append({
                        'model': model_name, 'action': action,
                        'market_price': row['mid_price'], 'predicted_price': row['predicted_price'],
                        'edge': row['price_diff_pct'], 'ticker': row['ticker'], 'date': row['date']
                    })
            except Exception as e:
                logger.error(f"Error during calibrate/predict for {model_name}: {e}")
        
        return signals

    def _execute_trades(self, signals, config):
        """Executes trades based on signals and calculates PnL."""
        trades = []
        total_pnl = 0
        
        for signal in signals:
            # Simple PnL model: PnL = (predicted_price - market_price)
            # This assumes we can enter at market and exit at our predicted "fair" price.
            if signal['action'] == 'buy':
                pnl = signal['predicted_price'] - signal['market_price']
            else: # sell
                pnl = signal['market_price'] - signal['predicted_price']
            
            pnl -= config.transaction_cost # Apply cost per trade
            
            trades.append({**signal, 'pnl': pnl})
            total_pnl += pnl
            
        return total_pnl, trades

    def _calculate_performance_metrics(self, config, daily_pnl, trade_history):
        """Calculates final performance metrics for the backtest."""
        returns_df = pd.DataFrame(daily_pnl).set_index('date')['pnl']
        trades_df = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()
        
        if returns_df.empty:
            return self._empty_result(config)
        
        cum_returns = returns_df.cumsum()
        total_return = cum_returns.iloc[-1]
        
        # Sharpe Ratio (annualized)
        daily_std = returns_df.std()
        sharpe_ratio = (returns_df.mean() / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        
        # Max Drawdown
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win Rate
        win_rate = (trades_df['pnl'] > 0).mean() if not trades_df.empty else 0
        
        # Model-specific performance
        model_perf = {}
        if not trades_df.empty:
            for model_name in config.models:
                model_trades = trades_df[trades_df['model'] == model_name]
                if not model_trades.empty:
                    model_perf[model_name] = {
                        'total_pnl': model_trades['pnl'].sum(),
                        'num_trades': len(model_trades),
                        'win_rate': (model_trades['pnl'] > 0).mean()
                    }
        
        logger.info(f"Backtest completed. Total Return: ${total_return:.2f}, Sharpe: {sharpe_ratio:.2f}")
        
        return BacktestResult(
            config=config, total_return=total_return, sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown, win_rate=win_rate, model_performance=model_perf,
            daily_returns=returns_df, cumulative_returns=cum_returns,
            trade_analysis=trades_df, calibration_history=pd.DataFrame() # Placeholder
        )

    def _load_historical_data(self, config: BacktestConfig):
        dates = self.data_collector.get_available_dates(config.symbol)
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d") - timedelta(days=config.calibration_window)
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
        
        relevant_dates = [d for d in dates if start_dt <= datetime.strptime(d, "%Y-%m-%d") <= end_dt]
        
        all_data = [self.data_collector.load_daily_data(config.symbol, d) for d in relevant_dates]
        df = pd.concat([d for d in all_data if d is not None and not d.empty])
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _generate_trading_dates(self, data, config):
        start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
        
        # Find available dates within the main backtest period
        available_dates = sorted(data['date'].unique())
        trading_dates = [d for d in available_dates if start_date <= d <= end_date]
        
        # Select dates based on rebalance frequency
        return trading_dates[::config.rebalance_frequency]

    def _get_data_for_period(self, data, start, end, config):
        period_data = data[(data['date'] >= start) & (data['date'] < end)].copy()
        
        # Apply filters
        return period_data[
            (period_data['dte'] >= config.min_dte) &
            (period_data['dte'] <= config.max_dte) &
            (period_data['spread_pct'] <= config.max_spread_pct)
        ]
        
    def _empty_result(self, config):
        return BacktestResult(
            config=config, total_return=0, sharpe_ratio=0, max_drawdown=0, win_rate=0,
            model_performance={}, daily_returns=pd.Series(), cumulative_returns=pd.Series(),
            trade_analysis=pd.DataFrame(), calibration_history=pd.DataFrame()
        )

def run_backtest_analysis(symbol: str, start_date: str, end_date: str, **kwargs) -> BacktestResult:
    config = BacktestConfig(symbol=symbol, start_date=start_date, end_date=end_date, **kwargs)
    backtester = VolatilityBacktester()
    return backtester.run_backtest(config) 