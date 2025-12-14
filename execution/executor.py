"""
Execution Engine - Executes trade signals with risk management
"""

from database.models import SessionLocal, Order, TradeSignal, Portfolio, MarketData
from sqlalchemy import select
from loguru import logger
from datetime import datetime
from execution.risk_manager import RiskManager
from typing import Optional

class ExecutionEngine:
    """Executes trade signals with safety checks"""
    
    def __init__(self, mode: str = "PAPER"):
        """
        Initialize execution engine
        
        Args:
            mode: "PAPER" for paper trading, "LIVE" for live trading
        """
        self.session = SessionLocal()
        self.mode = mode
        self.risk_manager = RiskManager()
        logger.info(f"🚀 Execution Engine initialized in {mode} mode")
    
    def process_new_signals(self):
        """Process all NEW trade signals"""
        logger.info("📋 Processing new trade signals...")
        
        # Get all NEW signals
        stmt = select(TradeSignal).filter_by(status="NEW").order_by(TradeSignal.created_at)
        signals = self.session.scalars(stmt).all()
        
        if not signals:
            logger.info("No new signals to process")
            return
        
        processed = 0
        rejected = 0
        
        for signal in signals:
            try:
                # Risk validation
                is_valid, reason = self.risk_manager.validate_signal(signal)
                
                if not is_valid:
                    signal.status = "REJECTED"
                    logger.warning(f"❌ Signal {signal.id} rejected: {reason}")
                    rejected += 1
                    continue
                
                # Create order
                order = self._create_order(signal)
                
                if order:
                    # Execute order (simulated for paper trading)
                    if self.mode == "PAPER":
                        self._execute_paper_order(order)
                    else:
                        # In live mode, this would call broker API
                        logger.warning("⚠️  LIVE trading not implemented yet")
                        order.status = "REJECTED"
                    
                    signal.status = "PROCESSED"
                    processed += 1
                    logger.success(f"✅ Signal {signal.id} processed -> Order {order.id}")
                else:
                    signal.status = "REJECTED"
                    rejected += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing signal {signal.id}: {e}")
                signal.status = "REJECTED"
                continue
        
        self.session.commit()
        logger.success(f"✅ Execution complete: {processed} processed, {rejected} rejected")
    
    def _create_order(self, signal: TradeSignal) -> Optional[Order]:
        """Create order from signal"""
        try:
            order = Order(
                signal_id=signal.id,
                ticker=signal.ticker,
                order_type="MARKET",  # Can be enhanced to support LIMIT orders
                side=signal.signal,  # BUY or SELL
                quantity=signal.quantity,
                price=signal.entry_price,
                status="SUBMITTED",
                stop_loss=signal.stop_loss,
                target_price=signal.target_price,
                mode=self.mode
            )
            
            self.session.add(order)
            self.session.commit()
            return order
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            self.session.rollback()
            return None
    
    def _execute_paper_order(self, order: Order):
        """Execute order in paper trading mode (simulated)"""
        try:
            # Get current market price
            stmt = select(MarketData).filter_by(ticker=order.ticker).order_by(
                MarketData.date.desc()
            )
            latest = self.session.scalars(stmt).first()
            
            if not latest:
                order.status = "REJECTED"
                return
            
            # Simulate execution at current price
            order.filled_at = datetime.now()
            order.filled_price = latest.close  # Execute at current market price
            order.status = "FILLED"
            
            # Update portfolio
            self._update_portfolio(order)
            
            logger.info(f"📝 Paper trade executed: {order.side} {order.quantity} {order.ticker} @ {order.filled_price}")
            
        except Exception as e:
            logger.error(f"Failed to execute paper order: {e}")
            order.status = "REJECTED"
    
    def _update_portfolio(self, order: Order):
        """Update portfolio after order fill"""
        if order.status != "FILLED":
            return
        
        stmt = select(Portfolio).filter_by(ticker=order.ticker)
        position = self.session.scalars(stmt).first()
        
        if order.side == "BUY":
            if position:
                # Update existing position
                total_cost = (position.avg_entry_price * position.quantity) + (order.filled_price * order.quantity)
                total_quantity = position.quantity + order.quantity
                position.avg_entry_price = total_cost / total_quantity
                position.quantity = total_quantity
                position.current_price = order.filled_price
                position.stop_loss = order.stop_loss
                position.target_price = order.target_price
                position.last_updated = datetime.now()
            else:
                # Create new position
                position = Portfolio(
                    ticker=order.ticker,
                    quantity=order.quantity,
                    avg_entry_price=order.filled_price,
                    current_price=order.filled_price,
                    stop_loss=order.stop_loss,
                    target_price=order.target_price,
                    order_id=order.id
                )
                self.session.add(position)
        
        elif order.side == "SELL":
            if position and position.quantity >= order.quantity:
                # Calculate realized P&L
                realized_pnl = (order.filled_price - position.avg_entry_price) * order.quantity
                position.realized_pnl += realized_pnl
                
                # Reduce position
                position.quantity -= order.quantity
                position.current_price = order.filled_price
                position.last_updated = datetime.now()
                
                # If position is closed, mark order as CLOSED
                if position.quantity == 0:
                    order.status = "CLOSED"
                    # Optionally remove from portfolio or keep for history
            else:
                logger.warning(f"Cannot sell {order.quantity} of {order.ticker} - insufficient position")
                order.status = "REJECTED"
        
        self.session.commit()
    
    def update_portfolio_prices(self):
        """Update current prices in portfolio"""
        logger.info("💰 Updating portfolio prices...")
        
        positions = self.session.scalars(select(Portfolio)).all()
        
        for position in positions:
            try:
                stmt = select(MarketData).filter_by(ticker=position.ticker).order_by(
                    MarketData.date.desc()
                )
                latest = self.session.scalars(stmt).first()
                
                if latest:
                    position.current_price = latest.close
                    position.unrealized_pnl = (latest.close - position.avg_entry_price) * position.quantity
                    position.last_updated = datetime.now()
                    
            except Exception as e:
                logger.error(f"Failed to update price for {position.ticker}: {e}")
        
        self.session.commit()
        logger.success("✅ Portfolio prices updated")
    
    def close(self):
        """Close database session"""
        self.risk_manager.close()
        self.session.close()

if __name__ == "__main__":
    engine = ExecutionEngine(mode="PAPER")
    engine.process_new_signals()
    engine.update_portfolio_prices()
    engine.close()

