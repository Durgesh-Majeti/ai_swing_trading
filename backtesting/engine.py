"""
Backtesting Engine - Test strategies on historical data
"""

import pandas as pd
from datetime import datetime, timedelta
from database.models import (
    SessionLocal, MarketData, TradeSignal, CompanyProfile, Watchlist
)
from sqlalchemy import select, func
from loguru import logger
from typing import Dict, List, Optional
from strategies.registry import StrategyRegistry
import numpy as np

class BacktestResult:
    """Container for backtest results"""
    def __init__(self, strategy_name: str, ticker: str):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.trades = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.net_profit = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.start_date = None
        self.end_date = None

class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.session = SessionLocal()
        self.initial_capital = initial_capital
        self.strategy_registry = StrategyRegistry()
    
    def run_backtest(
        self, 
        strategy_name: str,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        position_size_pct: float = 0.10  # 10% of capital per trade
    ) -> BacktestResult:
        """
        Run backtest for a strategy on a specific ticker
        
        Args:
            strategy_name: Name of strategy to test
            ticker: Stock ticker to test
            start_date: Start of backtest period
            end_date: End of backtest period
            position_size_pct: Percentage of capital to use per trade
            
        Returns:
            BacktestResult object with performance metrics
        """
        logger.info(f"📊 Backtesting {strategy_name} on {ticker} from {start_date.date()} to {end_date.date()}")
        
        result = BacktestResult(strategy_name, ticker)
        result.start_date = start_date
        result.end_date = end_date
        
        # Get strategy
        strategy = self.strategy_registry.get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Strategy {strategy_name} not found")
            return result
        
        # Get historical market data
        stmt = select(MarketData).filter_by(ticker=ticker).filter(
            MarketData.date >= start_date,
            MarketData.date <= end_date
        ).order_by(MarketData.date)
        
        market_data = self.session.scalars(stmt).all()
        
        if len(market_data) < 30:
            logger.warning(f"Insufficient data for {ticker}")
            return result
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([{
            'date': md.date,
            'open': md.open,
            'high': md.high,
            'low': md.low,
            'close': md.close,
            'volume': md.volume
        } for md in market_data])
        
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Simulate trading
        capital = self.initial_capital
        position = None  # {ticker, entry_price, quantity, entry_date, stop_loss, target}
        equity_curve = [capital]
        peak_equity = capital
        
        for date_idx in df.index:
            current_price = df.loc[date_idx, 'close']
            current_date = date_idx if isinstance(date_idx, datetime) else pd.Timestamp(date_idx).to_pydatetime()
            
            # Check if we have an open position
            if position:
                # Check stop loss
                if position['side'] == 'BUY':
                    if current_price <= position['stop_loss']:
                        # Stop loss hit
                        pnl = (position['stop_loss'] - position['entry_price']) * position['quantity']
                        capital += pnl
                        result.trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': position['stop_loss'],
                            'quantity': position['quantity'],
                            'side': position['side'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                            'exit_reason': 'STOP_LOSS'
                        })
                        position = None
                    elif current_price >= position['target']:
                        # Target hit
                        pnl = (position['target'] - position['entry_price']) * position['quantity']
                        capital += pnl
                        result.trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': position['target'],
                            'quantity': position['quantity'],
                            'side': position['side'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                            'exit_reason': 'TARGET'
                        })
                        position = None
                    else:
                        # Update position value
                        current_value = current_price * position['quantity']
                        equity_curve.append(capital + (current_value - position['entry_price'] * position['quantity']))
                else:  # SELL position
                    if current_price >= position['stop_loss']:
                        # Stop loss hit (for short)
                        pnl = (position['entry_price'] - position['stop_loss']) * position['quantity']
                        capital += pnl
                        result.trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': position['stop_loss'],
                            'quantity': position['quantity'],
                            'side': position['side'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                            'exit_reason': 'STOP_LOSS'
                        })
                        position = None
                    elif current_price <= position['target']:
                        # Target hit (for short)
                        pnl = (position['entry_price'] - position['target']) * position['quantity']
                        capital += pnl
                        result.trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': position['target'],
                            'quantity': position['quantity'],
                            'side': position['side'],
                            'pnl': pnl,
                            'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                            'exit_reason': 'TARGET'
                        })
                        position = None
                    else:
                        current_value = position['entry_price'] * position['quantity']
                        unrealized = (position['entry_price'] - current_price) * position['quantity']
                        equity_curve.append(capital + unrealized)
            else:
                # No position - check for new signal
                # Calculate indicators on the fly from historical data up to current date
                try:
                    # Get data up to current date
                    historical_df = df.loc[df.index <= date_idx].copy()
                    
                    if len(historical_df) < 50:  # Need enough data for indicators
                        continue
                    
                    # Calculate indicators
                    import pandas_ta as ta
                    historical_df['RSI'] = ta.rsi(historical_df['close'], length=14)
                    macd_data = ta.macd(historical_df['close'])
                    if macd_data is not None and isinstance(macd_data, pd.DataFrame):
                        historical_df['MACD'] = macd_data['MACD_12_26_9']
                        historical_df['MACD_Signal'] = macd_data['MACDs_12_26_9']
                    historical_df['SMA_50'] = ta.sma(historical_df['close'], length=50)
                    historical_df['SMA_200'] = ta.sma(historical_df['close'], length=200)
                    historical_df['ATR'] = ta.atr(historical_df['high'], historical_df['low'], historical_df['close'], length=14)
                    
                    # Get latest values
                    latest = historical_df.iloc[-1]
                    rsi = latest['RSI'] if pd.notna(latest['RSI']) else None
                    macd = latest['MACD'] if pd.notna(latest['MACD']) else None
                    macd_signal = latest['MACD_Signal'] if pd.notna(latest['MACD_Signal']) else None
                    sma_50 = latest['SMA_50'] if pd.notna(latest['SMA_50']) else None
                    sma_200 = latest['SMA_200'] if pd.notna(latest['SMA_200']) else None
                    atr = latest['ATR'] if pd.notna(latest['ATR']) else None
                    
                    # Generate signal based on strategy type
                    signal = None
                    stop_loss = None
                    target_price = None
                    
                    if strategy_name == "Technical_RSI_MACD":
                        # Technical strategy logic
                        if rsi and rsi < 30 and macd and macd_signal and macd > macd_signal:
                            signal = "BUY"
                        elif rsi and rsi > 70 and macd and macd_signal and macd < macd_signal:
                            signal = "SELL"
                        
                        if signal and atr:
                            atr_multiplier = 2.0
                            stop_loss_pct = (atr * atr_multiplier) / current_price
                            if signal == "BUY":
                                stop_loss = current_price * (1 - stop_loss_pct)
                                target_price = current_price * (1 + stop_loss_pct * 2)
                            else:
                                stop_loss = current_price * (1 + stop_loss_pct)
                                target_price = current_price * (1 - stop_loss_pct * 2)
                    
                    elif strategy_name == "Hybrid_AI_Technical_Fundamental":
                        # Hybrid strategy - simplified for backtesting (no AI, no fundamentals)
                        buy_score = 0
                        sell_score = 0
                        
                        if rsi:
                            if rsi < 35:
                                buy_score += 50  # Higher weight without AI
                            elif rsi > 65:
                                sell_score += 50
                        
                        if macd and macd_signal:
                            if macd > macd_signal:
                                buy_score += 25
                            else:
                                sell_score += 25
                        
                        if sma_50 and sma_200:
                            if current_price > sma_50 > sma_200:
                                buy_score += 25
                            elif current_price < sma_50 < sma_200:
                                sell_score += 25
                        
                        threshold = 50  # Lower threshold without AI
                        if buy_score >= threshold:
                            signal = "BUY"
                        elif sell_score >= threshold:
                            signal = "SELL"
                        
                        if signal and atr:
                            atr_multiplier = 2.0
                            stop_loss_pct = (atr * atr_multiplier) / current_price
                            if signal == "BUY":
                                stop_loss = current_price * (1 - stop_loss_pct)
                                target_price = current_price * (1 + stop_loss_pct * 2)
                            else:
                                stop_loss = current_price * (1 + stop_loss_pct)
                                target_price = current_price * (1 - stop_loss_pct * 2)
                    
                    if signal and stop_loss and target_price:
                        # Calculate position size
                        position_value = capital * position_size_pct
                        quantity = int(position_value / current_price)
                        
                        if quantity > 0:
                            position = {
                                'ticker': ticker,
                                'side': signal,
                                'entry_price': current_price,
                                'quantity': quantity,
                                'entry_date': current_date,
                                'stop_loss': stop_loss,
                                'target': target_price
                            }
                            
                            # Deduct capital for position
                            capital -= (current_price * quantity)
                            
                except Exception as e:
                    logger.debug(f"Error generating signal for {ticker} on {current_date}: {e}")
                    continue
            
            # Update peak equity for drawdown calculation
            if capital > peak_equity:
                peak_equity = capital
            
            # Calculate drawdown
            drawdown = ((capital - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0
            if drawdown < result.max_drawdown:
                result.max_drawdown = drawdown
        
        # Close any remaining position at end date
        if position:
            final_price = df.iloc[-1]['close']
            if position['side'] == 'BUY':
                pnl = (final_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - final_price) * position['quantity']
            
            capital += pnl
            result.trades.append({
                'entry_date': position['entry_date'],
                'exit_date': end_date,
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'quantity': position['quantity'],
                'side': position['side'],
                'pnl': pnl,
                'pnl_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                'exit_reason': 'END_DATE'
            })
        
        # Calculate metrics
        result.total_trades = len(result.trades)
        winning_trades = [t for t in result.trades if t['pnl'] > 0]
        losing_trades = [t for t in result.trades if t['pnl'] < 0]
        
        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        
        if winning_trades:
            result.total_profit = sum(t['pnl'] for t in winning_trades)
            result.avg_win = result.total_profit / len(winning_trades)
        
        if losing_trades:
            result.total_loss = abs(sum(t['pnl'] for t in losing_trades))
            result.avg_loss = result.total_loss / len(losing_trades)
        
        result.net_profit = capital - self.initial_capital
        result.win_rate = (result.winning_trades / result.total_trades * 100) if result.total_trades > 0 else 0
        result.profit_factor = (result.total_profit / result.total_loss) if result.total_loss > 0 else 0
        
        logger.success(
            f"✅ Backtest complete: {result.total_trades} trades, "
            f"Net P&L: ₹{result.net_profit:,.2f}, Win Rate: {result.win_rate:.1f}%"
        )
        
        strategy.close()
        return result
    
    def run_backtest_all_strategies(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        position_size_pct: float = 0.10
    ) -> Dict[str, BacktestResult]:
        """Run backtest for all available strategies"""
        results = {}
        
        for strategy_name in self.strategy_registry.list_strategies():
            try:
                result = self.run_backtest(strategy_name, ticker, start_date, end_date, position_size_pct)
                results[strategy_name] = result
            except Exception as e:
                logger.error(f"Backtest failed for {strategy_name}: {e}")
                continue
        
        return results
    
    def compare_strategies(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        position_size_pct: float = 0.10
    ) -> pd.DataFrame:
        """Compare all strategies across multiple tickers"""
        all_results = []
        
        for ticker in tickers:
            logger.info(f"Backtesting all strategies on {ticker}...")
            results = self.run_backtest_all_strategies(ticker, start_date, end_date, position_size_pct)
            
            for strategy_name, result in results.items():
                all_results.append({
                    'Strategy': strategy_name,
                    'Ticker': ticker,
                    'Total Trades': result.total_trades,
                    'Win Rate %': result.win_rate,
                    'Net Profit': result.net_profit,
                    'Profit Factor': result.profit_factor,
                    'Max Drawdown %': result.max_drawdown,
                    'Avg Win': result.avg_win,
                    'Avg Loss': result.avg_loss
                })
        
        return pd.DataFrame(all_results)
    
    def close(self):
        """Close database session"""
        self.session.close()
        for strategy in self.strategy_registry.strategies.values():
            strategy.close()

