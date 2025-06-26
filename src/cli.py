#!/usr/bin/env python3
"""Command-line interface for the Options Pricing Engine."""

import argparse
import sys
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import click
from pathlib import Path
import logging
from datetime import datetime
import os

from .config import config
from .providers.polygon import get_option_chain, get_underlying_price, get_dividend_yield
from .bsm import calculate_bsm_price_vectorized
from .pricing_engine import PricingEngine
from .providers.fred import get_risk_free_rate
from .data import HistoricalDataCollector, collect_data_for_backtesting
from .backtest import run_backtest_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s:%(levelname)s:%(message)s'
)

@click.group()
def cli():
    """Options Pricing Engine - A comprehensive toolkit for options analysis."""
    # This function is the entry point for the Click command group.
    # It is intentionally left blank.
    pass

@cli.command()
@click.option('--symbol', default='QQQ', help='Stock symbol')
@click.option('--date', default='2024-01-02', help='Date (YYYY-MM-DD)')
@click.option('--min-dte', default=7, help='Minimum days to expiration')
@click.option('--max-dte', default=90, help='Maximum days to expiration')
@click.option('--models', default='sabr,quadratic,cubic', help='Comma-separated list of models')
def validate(symbol, date, min_dte, max_dte, models):
    """Validate pricing models against market data."""
    
    print(f"\n🎯 Options Pricing Engine - Model Validation")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📊 Symbol: {symbol}")
    print(f"📅 Date: {date}")
    print(f"⏰ DTE Range: {min_dte} to {max_dte} days")
    print(f"🧮 Models: {models}")
    print()
    
    try:
        # Parse models
        model_list = [m.strip() for m in models.split(',')]
        
        # Run validation
        engine = PricingEngine()
        results = engine.validate_models(
            symbol=symbol,
            date=date,
            min_dte=min_dte,
            max_dte=max_dte,
            models=model_list
        )
        
        # Display results
        print("📈 Model Performance Summary:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for model_name, metrics in results['model_performance'].items():
            print(f"\n{model_name.upper()} Model:")
            print(f"  • RMSE: ${metrics['rmse']:.2f}")
            print(f"  • MAE:  ${metrics['mae']:.2f}")
            print(f"  • R²:   {metrics['r2']:.3f}")
        
        # Show best model
        best_model = min(results['model_performance'].items(), 
                        key=lambda x: x[1]['rmse'])
        print(f"\n🏆 Best Model: {best_model[0].upper()} (RMSE: ${best_model[1]['rmse']:.2f})")
        
        # Show data summary
        print(f"\n📊 Data Summary:")
        print(f"  • Total Options: {results['total_options']:,}")
        print(f"  • After Filtering: {results['filtered_options']:,}")
        print(f"  • Success Rate: {results['filtered_options']/results['total_options']*100:.1f}%")
        
        print(f"\n✅ Validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--symbols', default='QQQ', help='Comma-separated list of symbols')
@click.option('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-01-31', help='End date (YYYY-MM-DD)')
@click.option('--min-dte', default=7, help='Minimum days to expiration')
@click.option('--max-dte', default=180, help='Maximum days to expiration')
@click.option('--max-workers', default=10, help='Maximum concurrent workers')
@click.option('--min-strike-pct', default=0.8, help='Minimum strike as % of underlying')
@click.option('--max-strike-pct', default=1.2, help='Maximum strike as % of underlying')
@click.option('--force-refresh', is_flag=True, help='Force refresh existing data')
def collect(symbols, start_date, end_date, min_dte, max_dte, max_workers, 
           min_strike_pct, max_strike_pct, force_refresh):
    """Collect historical options data for backtesting."""
    
    print(f"\n📊 Historical Data Collection")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🎯 Symbols: {symbols}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"⏰ DTE Range: {min_dte} to {max_dte} days")
    print(f"🔧 Workers: {max_workers}")
    print(f"💰 Strike Range: {min_strike_pct:.0%} to {max_strike_pct:.0%}")
    print()
    
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Collect data
        from .data import collect_data_for_backtesting
        
        results = collect_data_for_backtesting(
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            min_dte=min_dte,
            max_dte=max_dte,
            max_workers=max_workers,
            min_strike_pct=min_strike_pct,
            max_strike_pct=max_strike_pct
        )
        
        # Display results
        print("📈 Collection Summary:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for symbol, stats in results.items():
            print(f"\n{symbol}:")
            print(f"  • Files Collected: {stats['collected']}")
            print(f"  • Files Skipped: {stats['skipped']}")
            print(f"  • Files Failed: {stats['failed']}")
            print(f"  • Success Rate: {stats['collected']/(stats['collected']+stats['failed'])*100:.1f}%")
        
        print(f"\n✅ Data collection completed!")
        
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--symbol', default='QQQ', help='Stock symbol to backtest')
@click.option('--start-date', default='2024-01-01', help='Backtest start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-01-31', help='Backtest end date (YYYY-MM-DD)')
@click.option('--calibration-window', default=30, help='Calibration window in days')
@click.option('--rebalance-frequency', default=5, help='Rebalance frequency in days')
@click.option('--models', default='sabr,quadratic,cubic', help='Comma-separated list of models')
@click.option('--min-dte', default=7, help='Minimum days to expiration')
@click.option('--max-dte', default=90, help='Maximum days to expiration')
@click.option('--max-spread-pct', default=0.5, help='Maximum bid-ask spread percentage')
@click.option('--transaction-cost', default=0.01, help='Transaction cost per contract')
def backtest(symbol, start_date, end_date, calibration_window, rebalance_frequency,
            models, min_dte, max_dte, max_spread_pct, transaction_cost):
    """Run comprehensive backtesting analysis."""
    
    print(f"\n🚀 Options Backtesting Engine")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🎯 Symbol: {symbol}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"📊 Calibration Window: {calibration_window} days")
    print(f"🔄 Rebalance Frequency: {rebalance_frequency} days")
    print(f"🧮 Models: {models}")
    print(f"⏰ DTE Range: {min_dte} to {max_dte} days")
    print(f"💰 Max Spread: {max_spread_pct:.1%}")
    print(f"💸 Transaction Cost: ${transaction_cost:.2f}")
    print()
    
    try:
        # Parse models
        model_list = [m.strip() for m in models.split(',')]
        
        # Run backtest
        print("🔄 Running backtest analysis...")
        results = run_backtest_analysis(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            calibration_window=calibration_window,
            rebalance_frequency=rebalance_frequency,
            models=model_list,
            min_dte=min_dte,
            max_dte=max_dte,
            max_spread_pct=max_spread_pct,
            transaction_cost=transaction_cost
        )
        
        # Display results
        print("\n📈 Backtest Results:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"💰 Total Return: ${results.total_return:.2f}")
        print(f"📊 Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"📉 Max Drawdown: ${results.max_drawdown:.2f}")
        print(f"🎯 Win Rate: {results.win_rate:.1%}")
        print(f"📝 Total Trades: {len(results.trade_analysis)}")
        
        # Model performance breakdown
        print(f"\n🧮 Model Performance:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for model, perf in results.model_performance.items():
            print(f"{model.upper()}:")
            print(f"  • Total PnL: ${perf['total_pnl']:.2f}")
            print(f"  • Trades: {perf['num_trades']}")
            print(f"  • Win Rate: {perf['win_rate']:.1%}")
        
        # Save detailed results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save trade analysis
        if not results.trade_analysis.empty:
            trade_file = output_dir / f"{symbol}_backtest_trades_{start_date}_{end_date}.csv"
            results.trade_analysis.to_csv(trade_file, index=False)
            print(f"\n💾 Trade analysis saved to: {trade_file}")
        
        # Save calibration history
        if not results.calibration_history.empty:
            calib_file = output_dir / f"{symbol}_calibration_history_{start_date}_{end_date}.csv"
            results.calibration_history.to_csv(calib_file, index=False)
            print(f"💾 Calibration history saved to: {calib_file}")
        
        print(f"\n✅ Backtesting completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during backtesting: {e}")
        raise click.ClickException(str(e))

@cli.command()
def status():
    """Show data collection and system status."""
    
    print(f"\n📊 Options Engine Status")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    try:
        from .data import HistoricalDataCollector
        
        collector = HistoricalDataCollector()
        status = collector.get_collection_status()
        
        print(f"📅 Last Updated: {status.get('last_updated', 'Never')}")
        print(f"📁 Total Files: {status.get('total_files', 0)}")
        print(f"💾 Total Size: {status.get('total_size_mb', 0):.1f} MB")
        
        if 'symbols' in status:
            print(f"\n🎯 Available Symbols:")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            for symbol, info in status['symbols'].items():
                print(f"{symbol}:")
                print(f"  • Date Range: {info.get('first_date', 'N/A')} to {info.get('last_date', 'N/A')}")
                print(f"  • Files: {info.get('total_files', 0)}")
                print(f"  • Size: {info.get('total_size_mb', 0):.1f} MB")
                print(f"  • Last Collection: {info.get('last_collection', 'Never')}")
        
        print(f"\n✅ System operational!")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        raise click.ClickException(str(e))

def main():
    """
    Main entry point for the CLI.
    
    This function should not be called directly. It is configured as the
    entry point in setup.cfg and invokes the Click command-line interface.
    """
    # Initialize logging, etc. here if needed
    
    # Run the Click command-line interface
    cli()

if __name__ == "__main__":
    main() 