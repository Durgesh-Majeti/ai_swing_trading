"""
Database models for storing backtest results
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from database.models import Base
from datetime import datetime

class BacktestRun(Base):
    """Stores backtest run metadata"""
    __tablename__ = "backtest_runs"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.now)
    strategy_name = Column(String, index=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"), index=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    position_size_pct = Column(Float)
    
    # Results
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    net_profit = Column(Float)
    total_profit = Column(Float)
    total_loss = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    final_capital = Column(Float)
    
    # Metadata
    notes = Column(Text)

class BacktestTrade(Base):
    """Stores individual trades from backtest"""
    __tablename__ = "backtest_trades"
    
    id = Column(Integer, primary_key=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id"), index=True)
    
    entry_date = Column(DateTime)
    exit_date = Column(DateTime)
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Integer)
    side = Column(String)  # BUY or SELL
    pnl = Column(Float)
    pnl_pct = Column(Float)
    exit_reason = Column(String)  # STOP_LOSS, TARGET, END_DATE

