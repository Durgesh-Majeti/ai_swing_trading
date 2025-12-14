"""
Strategy Engine - The Decision Maker
Runs all strategies and generates trade signals
"""

from database.models import SessionLocal, TradeSignal, Watchlist, CompanyProfile, Index, StrategyMetadata
from sqlalchemy import select
from loguru import logger
from strategies.registry import StrategyRegistry
from datetime import datetime
from typing import Optional

class StrategyEngine:
    """Orchestrates strategy execution and signal generation"""
    
    def __init__(self, index_name: Optional[str] = None):
        """
        Initialize Strategy Engine
        
        Args:
            index_name: Optional index name (e.g., "NIFTY_50"). If None, processes all indices.
        """
        self.session = SessionLocal()
        self.index_name = index_name
        
        # Get index_id if index_name is provided
        index_id = None
        if index_name:
            index = self.session.scalar(select(Index).filter_by(name=index_name, is_active=True))
            if index:
                index_id = index.id
        
        # Initialize registry with index information
        self.registry = StrategyRegistry(index_id=index_id, index_name=index_name)
    
    def run_daily_analysis(self, index_name: Optional[str] = None):
        """
        Run all strategies and generate signals
        
        Args:
            index_name: Optional index name to filter by. Overrides instance index_name if provided.
        """
        index_filter = index_name or self.index_name
        
        logger.info(f"⚖️  Starting Strategy Engine{' for ' + index_filter if index_filter else ''}...")
        
        # Get index if filtering
        index = None
        if index_filter:
            index = self.session.scalar(select(Index).filter_by(name=index_filter, is_active=True))
            if not index:
                logger.warning(f"Index {index_filter} not found or inactive")
                return
        
        # Get active watchlist, filtered by index if specified
        if index:
            watchlist = self.session.scalars(
                select(Watchlist).filter_by(is_active=True, index_id=index.id)
            ).all()
        else:
            watchlist = self.session.scalars(
                select(Watchlist).filter_by(is_active=True)
            ).all()
        
        if not watchlist:
            # Fallback: get companies from index if specified
            if index:
                tickers = [c.ticker for c in index.companies]
            else:
                companies = self.session.scalars(select(CompanyProfile)).all()
                tickers = [c.ticker for c in companies]
        else:
            tickers = [w.ticker for w in watchlist]
        
        # Get index-specific strategies
        if index:
            # Get strategies for this index
            index_strategies = self.session.scalars(
                select(StrategyMetadata).filter_by(index_id=index.id, is_active=True)
            ).all()
            strategy_names = [s.strategy_name for s in index_strategies]
        else:
            strategy_names = None  # Use all strategies
        
        signals_generated = 0
        
        for ticker in tickers:
            try:
                # Run strategies (filtered by index if specified)
                if strategy_names:
                    # Only run index-specific strategies
                    signals = []
                    for strategy_name in strategy_names:
                        strategy = self.registry.get_strategy(strategy_name)
                        if strategy:
                            try:
                                signal = strategy.generate_signal(ticker)
                                if signal:
                                    signal['strategy_name'] = strategy_name
                                    signals.append(signal)
                            except Exception as e:
                                logger.error(f"Error running strategy {strategy_name} for {ticker}: {e}")
                else:
                    # Run all strategies
                    signals = self.registry.run_all_strategies(ticker)
                
                for signal_data in signals:
                    # Check if signal already exists (avoid duplicates)
                    stmt = select(TradeSignal).filter_by(
                        ticker=ticker,
                        strategy_name=signal_data['strategy_name'],
                        status="NEW"
                    ).order_by(TradeSignal.created_at.desc())
                    
                    existing = self.session.scalars(stmt).first()
                    
                    # Only create new signal if we don't have a recent one (within 24 hours)
                    if existing and (datetime.now() - existing.created_at).days < 1:
                        continue
                    
                    # Create trade signal with index_id
                    trade_signal = TradeSignal(
                        ticker=ticker,
                        index_id=index.id if index else None,  # Store index_id for isolation
                        strategy_name=signal_data['strategy_name'],
                        signal=signal_data['signal'],
                        status="NEW",
                        entry_price=signal_data['entry_price'],
                        stop_loss=signal_data['stop_loss'],
                        target_price=signal_data['target_price'],
                        quantity=signal_data['quantity'],
                        reasoning=signal_data.get('reasoning', ''),
                        priority=5  # Default priority
                    )
                    
                    self.session.add(trade_signal)
                    signals_generated += 1
                    
                    logger.info(
                        f"📊 {ticker}: {signal_data['signal']} signal from {signal_data['strategy_name']}"
                    )
                
            except Exception as e:
                logger.error(f"❌ Error processing {ticker}: {e}")
                continue
        
        self.session.commit()
        logger.success(f"✅ Strategy Engine complete: {signals_generated} signals generated")
        
        # Close all strategy sessions
        for strategy in self.registry.strategies.values():
            strategy.close()
        
        self.session.close()

if __name__ == "__main__":
                        # Run for selected index if specified
                        index_name = selected_index_name if selected_index_name else None
                        engine = StrategyEngine(index_name=index_name)
                        engine.run_daily_analysis(index_name=index_name)

