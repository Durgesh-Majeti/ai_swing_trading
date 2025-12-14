"""
Quick Start Script - Initialize the system
"""

from loguru import logger
import sys

def quick_start():
    """Run initial setup"""
    logger.info("🚀 Quick Start - Initializing Nifty 50 AI Swing Trader...")
    
    steps = [
        ("Initialize Database", "init_db.py"),
        ("Sync Nifty 50 Companies", "main.py"),
        ("Initialize Watchlist", "utils/watchlist_init.py"),
        ("Run Initial ETL", "orchestrator.py etl"),
    ]
    
    logger.info("\n📋 Setup Steps:")
    for i, (name, script) in enumerate(steps, 1):
        logger.info(f"  {i}. {name}")
    
    proceed = input("\nProceed with setup? (y/n): ")
    
    if proceed.lower() != 'y':
        logger.info("Setup cancelled")
        return
    
    # Step 1: Initialize Database
    logger.info("\n" + "="*60)
    logger.info("Step 1: Initializing Database...")
    logger.info("="*60)
    try:
        from init_db import initialize_database
        initialize_database()
    except Exception as e:
        logger.error(f"Failed: {e}")
        return
    
    # Step 2: Sync Companies
    logger.info("\n" + "="*60)
    logger.info("Step 2: Syncing Nifty 50 Companies...")
    logger.info("="*60)
    logger.info("Note: This will fetch company data from NSE")
    try:
        from engine.loaders.profile_loader import sync_nifty_companies
        sync_nifty_companies()
    except Exception as e:
        logger.error(f"Failed: {e}")
        return
    
    # Step 3: Initialize Watchlist
    logger.info("\n" + "="*60)
    logger.info("Step 3: Initializing Watchlist...")
    logger.info("="*60)
    try:
        from utils.watchlist_init import initialize_watchlist
        initialize_watchlist()
    except Exception as e:
        logger.error(f"Failed: {e}")
        return
    
    # Step 4: Run ETL
    logger.info("\n" + "="*60)
    logger.info("Step 4: Running Initial ETL (this may take a few minutes)...")
    logger.info("="*60)
    try:
        from engine.etl import ETLModule
        etl = ETLModule()
        etl.run_full_sync()
    except Exception as e:
        logger.error(f"Failed: {e}")
        return
    
    logger.success("\n" + "="*60)
    logger.success("✅ Quick Start Complete!")
    logger.success("="*60)
    logger.info("\nNext Steps:")
    logger.info("  1. Generate features: python -m ai.feature_store")
    logger.info("  2. Train a model: python ai/train_model.py")
    logger.info("  3. Start dashboard: streamlit run dashboard.py")
    logger.info("  4. Run workflows: python orchestrator.py [command]")

if __name__ == "__main__":
    quick_start()

