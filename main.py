"""
Main Entry Point - Legacy script for initial data setup
Note: Use ETL module or dashboard for regular operations
"""

from engine.loaders.profile_loader import sync_index_companies
from engine.etl import ETLModule
from loguru import logger

def main():
    """Legacy main function - use ETL module or dashboard instead"""
    logger.info("--- 🏭 STOCK ENGINE ETL PIPELINE (Legacy) ---")
    logger.warning("⚠️  This is a legacy script. Use ETL module or dashboard for regular operations.")
    
    # Step 1: Sync Company List
    index_name = input("Enter index name (e.g., NIFTY_50) or press Enter for NIFTY_50: ").strip()
    if not index_name:
        index_name = "NIFTY_50"
    
    cmd = input(f"1. Sync Company List for {index_name}? (y/n): ")
    if cmd.lower() == 'y':
        sync_index_companies(index_name)

    # Step 2: Run ETL
    cmd = input("2. Run ETL (Market Data + Indicators)? (y/n): ")
    if cmd.lower() == 'y':
        years = input("Enter years of data to fetch (default: 1): ").strip()
        years = float(years) if years else 1.0
        
        etl = ETLModule(index_name=index_name)
        etl.run_full_sync(years=years)

    logger.success("✅ Data Pipeline Complete.")

if __name__ == "__main__":
    main()