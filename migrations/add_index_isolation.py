"""
Migration: Add index_id to operational tables for complete index isolation

This migration adds index_id foreign key to:
- Portfolio (positions)
- TradeSignal (signals)
- Order (orders)
- AIPredictions (predictions)

This ensures complete isolation between indices.
"""

from database.models import SessionLocal, Index, Portfolio, TradeSignal, Order, AIPredictions, CompanyProfile
from sqlalchemy import text
from loguru import logger

def run_migration():
    """Add index_id columns to operational tables"""
    session = SessionLocal()
    
    try:
        logger.info("🔧 Starting index isolation migration...")
        
        # Get database connection
        connection = session.connection()
        
        # Add index_id to Portfolio table
        logger.info("Adding index_id to portfolio table...")
        try:
            connection.execute(text("ALTER TABLE portfolio ADD COLUMN index_id INTEGER"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_portfolio_index_id ON portfolio(index_id)"))
            logger.info("✅ Added index_id to portfolio")
        except Exception as e:
            if "duplicate column name" not in str(e).lower() and "already exists" not in str(e).lower():
                logger.warning(f"Portfolio index_id may already exist: {e}")
        
        # Add index_id to TradeSignal table
        logger.info("Adding index_id to trade_signals table...")
        try:
            connection.execute(text("ALTER TABLE trade_signals ADD COLUMN index_id INTEGER"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_trade_signals_index_id ON trade_signals(index_id)"))
            logger.info("✅ Added index_id to trade_signals")
        except Exception as e:
            if "duplicate column name" not in str(e).lower() and "already exists" not in str(e).lower():
                logger.warning(f"TradeSignal index_id may already exist: {e}")
        
        # Add index_id to Order table
        logger.info("Adding index_id to orders table...")
        try:
            connection.execute(text("ALTER TABLE orders ADD COLUMN index_id INTEGER"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_orders_index_id ON orders(index_id)"))
            logger.info("✅ Added index_id to orders")
        except Exception as e:
            if "duplicate column name" not in str(e).lower() and "already exists" not in str(e).lower():
                logger.warning(f"Order index_id may already exist: {e}")
        
        # Add index_id to AIPredictions table
        logger.info("Adding index_id to ai_predictions table...")
        try:
            connection.execute(text("ALTER TABLE ai_predictions ADD COLUMN index_id INTEGER"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_ai_predictions_index_id ON ai_predictions(index_id)"))
            logger.info("✅ Added index_id to ai_predictions")
        except Exception as e:
            if "duplicate column name" not in str(e).lower() and "already exists" not in str(e).lower():
                logger.warning(f"AIPredictions index_id may already exist: {e}")
        
        # Migrate existing data: Assign index_id based on company's first index
        logger.info("Migrating existing data to assign index_id...")
        
        # Get all indices
        indices = session.query(Index).all()
        index_map = {}  # ticker -> index_id
        
        for index in indices:
            for company in index.companies:
                if company.ticker not in index_map:
                    index_map[company.ticker] = index.id
        
        # Update Portfolio
        logger.info("Updating existing portfolio positions...")
        for ticker, index_id in index_map.items():
            connection.execute(
                text("UPDATE portfolio SET index_id = :index_id WHERE ticker = :ticker"),
                {"index_id": index_id, "ticker": ticker}
            )
        
        # Update TradeSignal
        logger.info("Updating existing trade signals...")
        for ticker, index_id in index_map.items():
            connection.execute(
                text("UPDATE trade_signals SET index_id = :index_id WHERE ticker = :ticker"),
                {"index_id": index_id, "ticker": ticker}
            )
        
        # Update Order
        logger.info("Updating existing orders...")
        for ticker, index_id in index_map.items():
            connection.execute(
                text("UPDATE orders SET index_id = :index_id WHERE ticker = :ticker"),
                {"index_id": index_id, "ticker": ticker}
            )
        
        # Update AIPredictions
        logger.info("Updating existing AI predictions...")
        for ticker, index_id in index_map.items():
            connection.execute(
                text("UPDATE ai_predictions SET index_id = :index_id WHERE ticker = :ticker"),
                {"index_id": index_id, "ticker": ticker}
            )
        
        session.commit()
        logger.info("✅ Index isolation migration completed successfully!")
        
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    run_migration()

