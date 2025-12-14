"""
Initialize Watchlist - Add all Nifty 50 stocks to watchlist
"""

from database.models import SessionLocal, Watchlist, CompanyProfile
from sqlalchemy import select
from loguru import logger

def initialize_watchlist():
    """Add all Nifty 50 companies to watchlist"""
    session = SessionLocal()
    
    try:
        # Get all companies
        companies = session.scalars(select(CompanyProfile)).all()
        
        if not companies:
            logger.warning("No companies found. Please sync companies first using main.py")
            return
        
        added = 0
        for company in companies:
            # Check if already in watchlist
            existing = session.scalar(select(Watchlist).filter_by(ticker=company.ticker))
            
            if not existing:
                watchlist_entry = Watchlist(ticker=company.ticker)
                session.add(watchlist_entry)
                added += 1
            elif not existing.is_active:
                existing.is_active = True
                added += 1
        
        session.commit()
        logger.success(f"✅ Added {added} stocks to watchlist")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize watchlist: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    initialize_watchlist()

