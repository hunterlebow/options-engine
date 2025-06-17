"""Command-line interface for the options engine."""

import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd

from .bsm_pricing import calculate_bsm_price
from .mispricing import compute_mispricing, get_top_mispriced, identify_arbitrage_opportunities
from .polygon_api import get_option_chain, get_underlying_price, get_dividend_yield
from .surface_utils import build_surface, plot_surface_3d

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Production Options Engine - Real-time mispricing detection for any ticker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic scan for SPY options
  python -m src.cli --symbol SPY
  
  # Scan QQQ with custom parameters
  python -m src.cli --symbol QQQ --min-dte 7 --max-dte 21 --threshold 5.0
  
  # Analyze individual stocks
  python -m src.cli --symbol AAPL --min-dte 14 --max-dte 30
  python -m src.cli --symbol TSLA --min-dte 7 --max-dte 14 --show-surface
  
  # Save results and show surface
  python -m src.cli --symbol META --output meta_analysis.csv --show-surface
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Underlying symbol (e.g., SPY, QQQ, AAPL, TSLA, META, HOOD, etc.)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--min-dte",
        type=int,
        default=7,
        help="Minimum days to expiry (default: 7)",
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=21,
        help="Maximum days to expiry (default: 21)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Mispricing threshold percentage (default: 3.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (CSV format)",
    )
    parser.add_argument(
        "--show-surface",
        action="store_true",
        help="Display 3D volatility surface",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top opportunities to show (default: 10)",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=10,
        help="Minimum daily volume filter (default: 10)",
    )
    parser.add_argument(
        "--max-spread",
        type=float,
        default=3.0,
        help="Maximum bid-ask spread filter (default: $3.00)",
    )
    parser.add_argument(
        "--dividend-yield",
        type=float,
        help="Override dividend yield (as percentage, e.g., 1.3 for 1.3%). If not provided, fetched dynamically.",
    )
    
    args = parser.parse_args()
    
    try:
        # Header
        print("=" * 60)
        print("🚀 PRODUCTION OPTIONS ENGINE")
        print("=" * 60)
        print(f"Symbol: {args.symbol.upper()}")
        print(f"DTE Range: {args.min_dte}-{args.max_dte} days")
        print(f"Mispricing Threshold: {args.threshold}%")
        print("-" * 60)
        
        # Get underlying price and dividend info
        print(f"📡 Fetching market data for {args.symbol.upper()}...")
        underlying_price = get_underlying_price(args.symbol)
        print(f"📈 Current {args.symbol.upper()} price: ${underlying_price:.2f}")
        
        # Get dividend yield (dynamic or override)
        if args.dividend_yield is not None:
            dividend_yield = args.dividend_yield / 100.0  # Convert percentage to decimal
            print(f"💰 Using override dividend yield: {dividend_yield:.3%}")
        else:
            dividend_yield = get_dividend_yield(args.symbol)
            print(f"💰 Dynamic dividend yield: {dividend_yield:.3%}")
        
        # Get options chain
        print(f"\n📡 Fetching options chain from Polygon.io...")
        df = get_option_chain(
            args.symbol,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
        )
        
        if df is None or len(df) == 0:
            print(f"❌ No options data found for {args.symbol.upper()}. Check symbol and API key.")
            sys.exit(1)
            
        print(f"✅ Retrieved {len(df):,} options contracts")
        
        # Apply liquidity filters
        print("\n🔍 Applying liquidity filters...")
        
        # Adjust spread filter based on underlying price (percentage-based for expensive stocks)
        if underlying_price > 100:
            # For expensive stocks, use percentage-based spread filter
            max_spread_pct = 0.05  # 5% max spread
            spread_filter = (df['ask'] - df['bid']) / df['mid_price'] <= max_spread_pct
            print(f"   Using percentage-based spread filter: {max_spread_pct:.1%} for ${underlying_price:.0f} stock")
        else:
            # For cheaper stocks/ETFs, use dollar-based filter
            spread_filter = (df['ask'] - df['bid']) <= args.max_spread
            print(f"   Using dollar-based spread filter: ${args.max_spread:.2f}")
        
        # Filter for tradeable options
        tradeable = df[
            (df['bid'] > 0) &  # Must have bid
            (df['ask'] > df['bid']) &  # Valid spread
            spread_filter &  # Reasonable spread (dynamic)
            (df['open_interest'] >= 10) &  # Minimum open interest
            (df['volume'].fillna(0) >= args.min_volume) &  # Minimum volume
            (df['implied_volatility'].notna())  # Must have IV
        ].copy()
        
        if len(tradeable) == 0:
            print(f"❌ No tradeable options found for {args.symbol.upper()} after filtering")
            sys.exit(1)
            
        print(f"✅ {len(tradeable):,} tradeable options after filtering")
        
        # Calculate BSM prices with dynamic dividend yield
        print("\n⚡ Calculating BSM theoretical prices...")
        
        bsm_results = []
        for idx, option in tradeable.iterrows():
            try:
                bsm_price = calculate_bsm_price(
                    S=underlying_price,
                    K=option['strike'],
                    T=option['dte'] / 365.0,
                    r=0.05,  # 5% risk-free rate (could be made dynamic too)
                    sigma=option['implied_volatility'],  # Already in decimal format from Polygon
                    option_type=option['option_type'],
                    q=dividend_yield,  # Use dynamic dividend yield
                    symbol=args.symbol  # Pass symbol for any additional lookups
                )
                
                bsm_results.append({
                    'bsm_price': bsm_price,
                    'market_price': option['mid_price'],
                    'price_diff': bsm_price - option['mid_price'],
                    'price_diff_pct': ((bsm_price - option['mid_price']) / option['mid_price']) * 100
                })
                
            except Exception as e:
                bsm_results.append({
                    'bsm_price': None,
                    'market_price': option['mid_price'],
                    'price_diff': None,
                    'price_diff_pct': None
                })
        
        # Add BSM results
        bsm_df = pd.DataFrame(bsm_results)
        tradeable_with_bsm = pd.concat([tradeable.reset_index(drop=True), bsm_df], axis=1)
        tradeable_with_bsm = tradeable_with_bsm.dropna(subset=['bsm_price'])
        
        print(f"✅ BSM calculations complete for {len(tradeable_with_bsm):,} options")
        
        # Detect mispricing
        print("\n🎯 Detecting mispricing opportunities...")
        mispricing_results = compute_mispricing(tradeable_with_bsm, underlying_price)
        
        # Get top opportunities
        top_mispriced = get_top_mispriced(mispricing_results, n=args.top_n)
        
        # Display results
        print(f"\n🏆 TOP {args.top_n} MISPRICING OPPORTUNITIES FOR {args.symbol.upper()}")
        print("-" * 80)
        
        if len(top_mispriced) == 0:
            print("No significant mispricing opportunities found.")
        else:
            print(f"{'#':<3} {'Type':<4} {'Strike':<7} {'Exp':<10} {'Market':<8} {'BSM':<8} {'Diff%':<7} {'Strategy'}")
            print("-" * 80)
            
            for i, (_, row) in enumerate(top_mispriced.iterrows(), 1):
                strategy = "SELL" if row['price_diff_pct'] < 0 else "BUY"
                print(f"{i:<3} {row['option_type'].upper():<4} "
                      f"${row['strike']:<6.0f} {str(row['expiration_date'])[:10]:<10} "
                      f"${row['market_price']:<7.2f} ${row['bsm_price']:<7.2f} "
                      f"{row['price_diff_pct']:<6.1f}% {strategy}")
        
        # Risk analysis
        if len(top_mispriced) > 0:
            print(f"\n⚠️  RISK ANALYSIS")
            print("-" * 40)
            
            avg_spread = (top_mispriced['ask'] - top_mispriced['bid']).mean()
            avg_volume = top_mispriced['volume'].fillna(0).mean()
            avg_oi = top_mispriced['open_interest'].mean()
            avg_dte = top_mispriced['dte'].mean()
            
            print(f"Average bid-ask spread: ${avg_spread:.2f}")
            print(f"Average daily volume: {avg_volume:.0f}")
            print(f"Average open interest: {avg_oi:.0f}")
            print(f"Average days to expiry: {avg_dte:.1f}")
            
            if avg_dte < 7:
                print("⚠️  WARNING: Short-term options - high gamma risk!")
        
        # Market summary
        print(f"\n📊 MARKET SUMMARY")
        print("-" * 40)
        
        calls = mispricing_results[mispricing_results['option_type'] == 'call']
        puts = mispricing_results[mispricing_results['option_type'] == 'put']
        
        if len(calls) > 0 and len(puts) > 0:
            avg_call_iv = calls['implied_volatility'].mean()
            avg_put_iv = puts['implied_volatility'].mean()
            skew = avg_put_iv - avg_call_iv
            
            print(f"Average Call IV: {avg_call_iv:.1f}%")
            print(f"Average Put IV: {avg_put_iv:.1f}%")
            print(f"Put-Call Skew: {skew:.1f}% {'(Put premium)' if skew > 0 else '(Call premium)'}")
        
        # Save results
        if args.output:
            print(f"\n💾 Saving results to {args.output}...")
            mispricing_results.to_csv(args.output, index=False)
            print("✅ Results saved successfully")
        
        # Show volatility surface
        if args.show_surface:
            print("\n🌊 Generating 3D volatility surface...")
            surface_data = build_surface(mispricing_results)
            fig = plot_surface_3d(surface_data, underlying_price, 
                                title=f"{args.symbol.upper()} Options Volatility Surface")
            fig.show()
            print("✅ Volatility surface displayed")
        
        print(f"\n🎉 Analysis complete! Found {len(top_mispriced)} opportunities.")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 