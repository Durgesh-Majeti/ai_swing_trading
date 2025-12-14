"""
Utility to load historical data for training and backtesting
Supports parameterized data loading for different time periods
"""

from engine.etl import ETLModule
from ai.feature_store import FeatureStoreEngine
from database.models import SessionLocal, Index
from sqlalchemy import select
from loguru import logger
from datetime import datetime, timedelta
from typing import Optional, Union


def load_historical_data(
    index_name: str,
    years: Optional[float] = None,
    start_date: Optional[Union[datetime, str]] = None,
    end_date: Optional[Union[datetime, str]] = None,
    force_refresh: bool = False,
    generate_features: bool = True
):
    """
    Load historical market data for a specific index
    
    Args:
        index_name: Name of the index (e.g., "NIFTY_50")
        years: Number of years of historical data (e.g., 5.0 for 5 years)
        start_date: Start date for data range (datetime or string "YYYY-MM-DD")
        end_date: End date for data range (datetime or string "YYYY-MM-DD")
        force_refresh: If True, re-downloads all data even if it exists
        generate_features: If True, generates features after loading data
    
    Returns:
        dict: Summary of loaded data
    """
    session = SessionLocal()
    
    try:
        # Get index
        index = session.scalar(select(Index).filter_by(name=index_name, is_active=True))
        if not index:
            logger.error(f"Index {index_name} not found or inactive")
            return {"success": False, "error": f"Index {index_name} not found"}
        
        logger.info(f"📊 Loading historical data for {index.display_name}...")
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Initialize ETL module with index (can use index_name or index_id)
        etl = ETLModule(index_name=index_name)
        
        # Load market data
        logger.info(f"📈 Fetching market data...")
        etl.sync_market_data(
            years=years,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh
        )
        
        # Calculate technical indicators
        logger.info(f"📊 Calculating technical indicators...")
        etl.calculate_technical_indicators()
        
        # Generate features if requested
        if generate_features:
            logger.info(f"🔧 Generating features...")
            feature_engine = FeatureStoreEngine(index_id=index.id)
            feature_engine.generate_all_features()
            feature_engine.close()
        
        # Get summary
        from database.models import MarketData, CompanyProfile
        tickers = [c.ticker for c in index.companies]
        
        total_records = 0
        date_ranges = {}
        
        for ticker in tickers:
            stmt = select(MarketData).filter_by(ticker=ticker).order_by(MarketData.date)
            records = session.scalars(stmt).all()
            if records:
                total_records += len(records)
                date_ranges[ticker] = {
                    "start": records[0].date,
                    "end": records[-1].date,
                    "count": len(records)
                }
        
        logger.success(f"✅ Historical data loaded successfully!")
        logger.info(f"   Total companies: {len(tickers)}")
        logger.info(f"   Total records: {total_records}")
        
        if date_ranges:
            min_date = min(r["start"] for r in date_ranges.values())
            max_date = max(r["end"] for r in date_ranges.values())
            logger.info(f"   Date range: {min_date.date()} to {max_date.date()}")
        
        return {
            "success": True,
            "index_name": index_name,
            "index_display_name": index.display_name,
            "total_companies": len(tickers),
            "total_records": total_records,
            "date_ranges": date_ranges
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to load historical data: {e}")
        return {"success": False, "error": str(e)}
    finally:
        session.close()


def load_nifty50_5years(force_refresh: bool = False, generate_features: bool = True):
    """
    Convenience function to load 5 years of data for Nifty 50
    
    Args:
        force_refresh: If True, re-downloads all data
        generate_features: If True, generates features after loading
    
    Returns:
        dict: Summary of loaded data
    """
    return load_historical_data(
        index_name="NIFTY_50",
        years=5.0,
        force_refresh=force_refresh,
        generate_features=generate_features
    )


if __name__ == "__main__":
    import sys
    
    # Default: Load 5 years of Nifty 50 data
    if len(sys.argv) > 1:
        index_name = sys.argv[1]
        years = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
        force_refresh = "--force" in sys.argv
        
        result = load_historical_data(
            index_name=index_name,
            years=years,
            force_refresh=force_refresh
        )
    else:
        # Load 5 years of Nifty 50 data
        result = load_nifty50_5years(force_refresh=False)
    
    if result.get("success"):
        logger.success("✅ Data loading complete!")
    else:
        logger.error(f"❌ Data loading failed: {result.get('error')}")

