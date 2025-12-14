"""
Utility to reset/initialize an index - deletes all operational data for a specific index
"""

from database.models import SessionLocal, Index, Portfolio, TradeSignal, Order, AIPredictions, Watchlist, StrategyMetadata
from sqlalchemy import select, delete
from loguru import logger
from typing import Optional

def reset_index(index_name: str, keep_watchlist: bool = False, keep_strategies: bool = False):
    """
    Reset/initialize an index by deleting all operational data.
    
    Args:
        index_name: Name of the index (e.g., "NIFTY_50")
        keep_watchlist: If True, keep watchlist entries
        keep_strategies: If True, keep strategy metadata
    
    Returns:
        dict with counts of deleted records
    """
    session = SessionLocal()
    
    try:
        # Get index
        index = session.scalar(select(Index).filter_by(name=index_name))
        if not index:
            logger.error(f"Index {index_name} not found")
            return {"error": f"Index {index_name} not found"}
        
        logger.info(f"🔄 Resetting index: {index.display_name} ({index_name})")
        
        deleted_counts = {
            "portfolio": 0,
            "trade_signals": 0,
            "orders": 0,
            "ai_predictions": 0,
            "watchlist": 0,
            "strategy_metadata": 0
        }
        
        # Delete Portfolio positions
        portfolio_positions = session.scalars(
            select(Portfolio).filter_by(index_id=index.id)
        ).all()
        for pos in portfolio_positions:
            session.delete(pos)
        deleted_counts["portfolio"] = len(portfolio_positions)
        logger.info(f"  Deleted {deleted_counts['portfolio']} portfolio positions")
        
        # Delete Trade Signals
        signals = session.scalars(
            select(TradeSignal).filter_by(index_id=index.id)
        ).all()
        for signal in signals:
            session.delete(signal)
        deleted_counts["trade_signals"] = len(signals)
        logger.info(f"  Deleted {deleted_counts['trade_signals']} trade signals")
        
        # Delete Orders
        orders = session.scalars(
            select(Order).filter_by(index_id=index.id)
        ).all()
        for order in orders:
            session.delete(order)
        deleted_counts["orders"] = len(orders)
        logger.info(f"  Deleted {deleted_counts['orders']} orders")
        
        # Delete AI Predictions
        predictions = session.scalars(
            select(AIPredictions).filter_by(index_id=index.id)
        ).all()
        for pred in predictions:
            session.delete(pred)
        deleted_counts["ai_predictions"] = len(predictions)
        logger.info(f"  Deleted {deleted_counts['ai_predictions']} AI predictions")
        
        # Delete Watchlist (optional)
        if not keep_watchlist:
            watchlist_entries = session.scalars(
                select(Watchlist).filter_by(index_id=index.id)
            ).all()
            for entry in watchlist_entries:
                session.delete(entry)
            deleted_counts["watchlist"] = len(watchlist_entries)
            logger.info(f"  Deleted {deleted_counts['watchlist']} watchlist entries")
        
        # Delete Strategy Metadata (optional)
        if not keep_strategies:
            strategies = session.scalars(
                select(StrategyMetadata).filter_by(index_id=index.id)
            ).all()
            for strategy in strategies:
                session.delete(strategy)
            deleted_counts["strategy_metadata"] = len(strategies)
            logger.info(f"  Deleted {deleted_counts['strategy_metadata']} strategy metadata entries")
        
        session.commit()
        logger.info(f"✅ Index {index.display_name} reset completed!")
        
        return deleted_counts
        
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Failed to reset index: {e}")
        raise
    finally:
        session.close()

def initialize_index(index_name: str):
    """
    Initialize an index from scratch - same as reset but with clearer naming.
    
    Args:
        index_name: Name of the index (e.g., "NIFTY_50")
    
    Returns:
        dict with counts of deleted records
    """
    return reset_index(index_name, keep_watchlist=False, keep_strategies=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        index_name = sys.argv[1]
        result = initialize_index(index_name)
        print(f"Reset result: {result}")
    else:
        print("Usage: python -m utils.index_reset <INDEX_NAME>")
        print("Example: python -m utils.index_reset NIFTY_50")

