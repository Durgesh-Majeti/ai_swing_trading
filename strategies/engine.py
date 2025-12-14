"""
Strategy Engine - The Decision Maker
Runs all strategies and generates trade signals
"""

from database.models import SessionLocal, TradeSignal, Watchlist, CompanyProfile
from sqlalchemy import select
from loguru import logger
from strategies.registry import StrategyRegistry
from datetime import datetime

class StrategyEngine:
    """Orchestrates strategy execution and signal generation"""
    
    def __init__(self):
        self.session = SessionLocal()
        self.registry = StrategyRegistry()
    
    def run_daily_analysis(self):
        """Run all strategies and generate signals"""
        logger.info("⚖️  Starting Strategy Engine...")
        
        # Get active watchlist
        watchlist = self.session.scalars(
            select(Watchlist).filter_by(is_active=True)
        ).all()
        
        if not watchlist:
            # Fallback to all companies
            companies = self.session.scalars(select(CompanyProfile)).all()
            tickers = [c.ticker for c in companies]
        else:
            tickers = [w.ticker for w in watchlist]
        
        signals_generated = 0
        
        for ticker in tickers:
            try:
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
                    
                    # Create trade signal
                    trade_signal = TradeSignal(
                        ticker=ticker,
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
    engine = StrategyEngine()
    engine.run_daily_analysis()

