"""
Backtest Runner - CLI and programmatic interface
"""

from datetime import datetime, timedelta
from backtesting.engine import BacktestEngine
from database.models import SessionLocal, Watchlist, CompanyProfile
from sqlalchemy import select
from loguru import logger
import sys

def run_single_backtest():
    """Run backtest for a single strategy and ticker"""
    print("📊 Backtest Configuration")
    print("=" * 60)
    
    # Get strategy
    engine = BacktestEngine()
    strategies = engine.strategy_registry.list_strategies()
    
    if not strategies:
        print("❌ No strategies available")
        return
    
    print("\nAvailable Strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy}")
    
    strategy_idx = int(input("\nSelect strategy (number): ")) - 1
    strategy_name = strategies[strategy_idx]
    
    # Get ticker
    ticker = input("\nEnter ticker (e.g., RELIANCE.NS): ").strip()
    
    # Get date range
    days_back = int(input("Days to look back (default 365): ") or "365")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Get position size
    position_size = float(input("Position size % of capital (default 10): ") or "10") / 100
    
    print(f"\n🚀 Running backtest...")
    print(f"   Strategy: {strategy_name}")
    print(f"   Ticker: {ticker}")
    print(f"   Period: {start_date.date()} to {end_date.date()}")
    print(f"   Initial Capital: ₹{engine.initial_capital:,.2f}")
    print()
    
    result = engine.run_backtest(strategy_name, ticker, start_date, end_date, position_size)
    
    # Display results
    print("\n" + "=" * 60)
    print("📈 BACKTEST RESULTS")
    print("=" * 60)
    print(f"Strategy: {result.strategy_name}")
    print(f"Ticker: {result.ticker}")
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print()
    print(f"Total Trades: {result.total_trades}")
    print(f"Winning Trades: {result.winning_trades}")
    print(f"Losing Trades: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate:.2f}%")
    print()
    print(f"Total Profit: ₹{result.total_profit:,.2f}")
    print(f"Total Loss: ₹{result.total_loss:,.2f}")
    print(f"Net Profit: ₹{result.net_profit:,.2f}")
    print(f"Return %: {(result.net_profit / engine.initial_capital) * 100:.2f}%")
    print()
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Average Win: ₹{result.avg_win:,.2f}")
    print(f"Average Loss: ₹{result.avg_loss:,.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    print("=" * 60)
    
    engine.close()

def run_all_strategies_backtest():
    """Run backtest for all strategies on a ticker"""
    print("📊 Backtest All Strategies")
    print("=" * 60)
    
    ticker = input("Enter ticker (e.g., RELIANCE.NS): ").strip()
    days_back = int(input("Days to look back (default 365): ") or "365")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    engine = BacktestEngine()
    
    print(f"\n🚀 Running backtests for all strategies on {ticker}...")
    results = engine.run_backtest_all_strategies(ticker, start_date, end_date)
    
    # Display comparison
    print("\n" + "=" * 60)
    print("📊 STRATEGY COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<30} {'Trades':<8} {'Win Rate':<10} {'Net P&L':<15} {'Profit Factor':<15}")
    print("-" * 60)
    
    for strategy_name, result in results.items():
        print(
            f"{strategy_name:<30} "
            f"{result.total_trades:<8} "
            f"{result.win_rate:>6.1f}%   "
            f"₹{result.net_profit:>12,.2f}  "
            f"{result.profit_factor:>12.2f}"
        )
    
    print("=" * 60)
    
    engine.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            run_all_strategies_backtest()
        else:
            print("Usage: python -m backtesting.runner [all]")
    else:
        run_single_backtest()

