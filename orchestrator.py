"""
Main Orchestrator - Entry point for the trading system
"""

import sys
from loguru import logger
from automation.scheduler import TradingScheduler

def main():
    """Main entry point"""
    logger.info("🚀 Nifty 50 AI Swing Trader - Starting...")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        scheduler = TradingScheduler()
        
        if command == "etl":
            logger.info("Running ETL workflow...")
            scheduler.market_close_workflow()
        elif command == "inference":
            logger.info("Running AI inference...")
            scheduler.evening_analysis_workflow()
        elif command == "strategy":
            logger.info("Running strategy engine...")
            scheduler.strategy.run_daily_analysis()
        elif command == "execute":
            logger.info("Running execution engine...")
            scheduler.pre_market_workflow()
        elif command == "all":
            logger.info("Running full workflow...")
            scheduler.market_close_workflow()
            scheduler.evening_analysis_workflow()
            scheduler.pre_market_workflow()
        elif command == "schedule":
            logger.info("Starting scheduler...")
            scheduler.start_scheduler()
        else:
            logger.error(f"Unknown command: {command}")
            logger.info("Available commands: etl, inference, strategy, execute, all, schedule")
    else:
        logger.info("Usage: python orchestrator.py [command]")
        logger.info("Commands:")
        logger.info("  etl       - Run ETL data collection")
        logger.info("  inference - Run AI inference")
        logger.info("  strategy  - Run strategy engine")
        logger.info("  execute   - Run execution engine")
        logger.info("  all       - Run full workflow")
        logger.info("  schedule  - Start automated scheduler")

if __name__ == "__main__":
    main()

