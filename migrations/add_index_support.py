"""
Migration: Add index support for multi-index trading
- Creates indices table
- Creates company_index_mapping junction table
- Updates watchlist to include index_id
- Updates strategy_metadata to include index_id
- Initializes default indices (NIFTY_50, NIFTY_100, NIFTY_500)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import Base, Index, CompanyProfile, Watchlist, StrategyMetadata, engine, company_index_mapping
from sqlalchemy import inspect, select, text
from loguru import logger
from datetime import datetime

def run_migration():
    """Add index support to the database"""
    logger.info("🔄 Running migration: Add index support")
    
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    # Step 1: Create indices table
    if "indices" not in existing_tables:
        logger.info("Creating indices table...")
        Index.__table__.create(engine, checkfirst=True)
        logger.success("✅ Created indices table")
    else:
        logger.info("✅ Table 'indices' already exists")
    
    # Step 2: Create company_index_mapping junction table
    if "company_index_mapping" not in existing_tables:
        logger.info("Creating company_index_mapping table...")
        company_index_mapping.create(engine, checkfirst=True)
        logger.success("✅ Created company_index_mapping table")
    else:
        logger.info("✅ Table 'company_index_mapping' already exists")
    
    # Step 3: Update watchlist table to add index_id column
    if "watchlist" in existing_tables:
        try:
            # Check if index_id column exists
            watchlist_columns = [col['name'] for col in inspector.get_columns('watchlist')]
            if 'index_id' not in watchlist_columns:
                logger.info("Adding index_id column to watchlist table...")
                with engine.connect() as conn:
                    # Remove unique constraint on ticker if it exists (SQLite doesn't support DROP CONSTRAINT directly)
                    # We'll handle this by allowing duplicates temporarily
                    conn.execute(text("ALTER TABLE watchlist ADD COLUMN index_id INTEGER"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_watchlist_index_id ON watchlist(index_id)"))
                    conn.commit()
                logger.success("✅ Added index_id column to watchlist")
            else:
                logger.info("✅ Column 'index_id' already exists in watchlist")
        except Exception as e:
            logger.warning(f"⚠️  Could not update watchlist table: {e}")
    
    # Step 4: Update strategy_metadata table to add index_id column
    if "strategy_metadata" in existing_tables:
        try:
            strategy_columns = [col['name'] for col in inspector.get_columns('strategy_metadata')]
            if 'index_id' not in strategy_columns:
                logger.info("Adding index_id column to strategy_metadata table...")
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE strategy_metadata ADD COLUMN index_id INTEGER"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_strategy_metadata_index_id ON strategy_metadata(index_id)"))
                    conn.commit()
                logger.success("✅ Added index_id column to strategy_metadata")
            else:
                logger.info("✅ Column 'index_id' already exists in strategy_metadata")
        except Exception as e:
            logger.warning(f"⚠️  Could not update strategy_metadata table: {e}")
    
    # Step 5: Initialize all NSE indices
    logger.info("Initializing all NSE indices...")
    from database.models import SessionLocal
    from engine.loaders.nse_index_discovery import ALL_NSE_INDICES
    
    session = SessionLocal()
    
    try:
        created = 0
        for index_name, index_info in ALL_NSE_INDICES.items():
            existing = session.scalar(select(Index).filter_by(name=index_name))
            if not existing:
                new_index = Index(
                    name=index_name,
                    display_name=index_info['display_name'],
                    description=f"Companies from {index_info['display_name']} index",
                    is_active=True
                )
                session.add(new_index)
                created += 1
                if created <= 10:  # Log first 10
                    logger.info(f"  Added index: {index_info['display_name']}")
        
        session.commit()
        logger.success(f"✅ Initialized {created} NSE indices (total: {len(ALL_NSE_INDICES)})")
        
        # Step 6: Migrate existing watchlist entries to NIFTY_50 (default)
        nifty_50 = session.scalar(select(Index).filter_by(name="NIFTY_50"))
        if nifty_50:
            # Update existing watchlist entries without index_id
            try:
                with engine.connect() as conn:
                    result = conn.execute(
                        text("UPDATE watchlist SET index_id = :index_id WHERE index_id IS NULL"),
                        {"index_id": nifty_50.id}
                    )
                    conn.commit()
                    logger.info(f"  Migrated {result.rowcount} watchlist entries to NIFTY_50")
            except Exception as e:
                logger.warning(f"⚠️  Could not migrate watchlist entries: {e}")
        
        # Step 7: Migrate existing strategy_metadata entries to NIFTY_50 (default)
        if nifty_50:
            try:
                with engine.connect() as conn:
                    result = conn.execute(
                        text("UPDATE strategy_metadata SET index_id = :index_id WHERE index_id IS NULL"),
                        {"index_id": nifty_50.id}
                    )
                    conn.commit()
                    logger.info(f"  Migrated {result.rowcount} strategy metadata entries to NIFTY_50")
            except Exception as e:
                logger.warning(f"⚠️  Could not migrate strategy metadata: {e}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize indices: {e}")
        session.rollback()
        raise
    finally:
        session.close()
    
    logger.success("✅ Migration complete: Index support added")

if __name__ == "__main__":
    run_migration()

