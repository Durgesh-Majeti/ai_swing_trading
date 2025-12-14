"""
Initialize Watchlist - Add stocks to watchlist by index
"""

from database.models import SessionLocal, Watchlist, CompanyProfile, Index
from sqlalchemy import select
from loguru import logger
from typing import Optional

def initialize_watchlist(index_name: Optional[str] = "NIFTY_50"):
    """
    Add companies from a specific index to watchlist
    
    Args:
        index_name: Name of the index (e.g., "NIFTY_50", "NIFTY_100"). Defaults to "NIFTY_50".
    """
    session = SessionLocal()
    
    try:
        # Get index
        index = session.scalar(select(Index).filter_by(name=index_name, is_active=True))
        if not index:
            logger.error(f"Index {index_name} not found. Please create it first.")
            return
        
        # Get companies in this index
        companies = index.companies
        
        if not companies:
            logger.warning(f"No companies found in index {index_name}. Please assign companies to index first.")
            return
        
        added = 0
        for company in companies:
            # Check if already in watchlist for this index
            existing = session.scalar(
                select(Watchlist).filter_by(ticker=company.ticker, index_id=index.id)
            )
            
            if not existing:
                watchlist_entry = Watchlist(ticker=company.ticker, index_id=index.id)
                session.add(watchlist_entry)
                added += 1
            elif not existing.is_active:
                existing.is_active = True
                added += 1
        
        session.commit()
        logger.success(f"✅ Added {added} stocks from {index.display_name} to watchlist")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize watchlist: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    initialize_watchlist()

