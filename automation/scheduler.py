"""
Scheduler - Daily workflow automation
"""

import schedule
import time
from loguru import logger
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.etl import ETLModule
from ai.inference import InferenceEngine
from ai.feature_store import FeatureStoreEngine
from strategies.engine import StrategyEngine
from execution.executor import ExecutionEngine

class TradingScheduler:
    """Orchestrates daily trading workflow"""
    
    def __init__(self):
        self.etl = ETLModule()
        self.feature_store = FeatureStoreEngine()
        self.inference = InferenceEngine()
        self.strategy = StrategyEngine()
        self.executor = ExecutionEngine(mode="PAPER")
    
    def market_close_workflow(self):
        """Runs at market close (15:30)"""
        logger.info("=" * 60)
        logger.info("🕐 MARKET CLOSE WORKFLOW STARTED")
        logger.info("=" * 60)
        
        try:
            # 1. ETL - Fetch market data
            logger.info("Step 1: Running ETL...")
            self.etl.run_full_sync()
            
            logger.success("✅ Market Close Workflow Complete")
        except Exception as e:
            logger.error(f"❌ Market Close Workflow Failed: {e}")
    
    def evening_analysis_workflow(self):
        """Runs in evening (17:00)"""
        logger.info("=" * 60)
        logger.info("🌙 EVENING ANALYSIS WORKFLOW STARTED")
        logger.info("=" * 60)
        
        try:
            # 1. Generate Features
            logger.info("Step 1: Generating features...")
            self.feature_store.generate_all_features()
            
            # 2. Run AI Inference (optional - continues even if no model available)
            logger.info("Step 2: Running AI inference...")
            try:
                self.inference.run_daily_inference()
            except Exception as e:
                logger.warning(f"⚠️  AI Inference skipped (no model or error): {e}")
                logger.info("Continuing with strategy engine (will use technical/fundamental analysis only)")
            
            # 3. Run Strategy Engine
            logger.info("Step 3: Running strategy engine...")
            self.strategy.run_daily_analysis()
            
            logger.success("✅ Evening Analysis Workflow Complete")
        except Exception as e:
            logger.error(f"❌ Evening Analysis Workflow Failed: {e}")
    
    def pre_market_workflow(self):
        """Runs pre-market (09:00)"""
        logger.info("=" * 60)
        logger.info("🌅 PRE-MARKET WORKFLOW STARTED")
        logger.info("=" * 60)
        
        try:
            # 1. Process new signals
            logger.info("Step 1: Processing trade signals...")
            self.executor.process_new_signals()
            
            # 2. Update portfolio prices
            logger.info("Step 2: Updating portfolio prices...")
            self.executor.update_portfolio_prices()
            
            logger.success("✅ Pre-Market Workflow Complete")
        except Exception as e:
            logger.error(f"❌ Pre-Market Workflow Failed: {e}")
    
    def start_scheduler(self):
        """Start the scheduled tasks"""
        logger.info("🚀 Starting Trading Scheduler...")
        
        # Schedule tasks
        schedule.every().day.at("15:30").do(self.market_close_workflow)  # Market close
        schedule.every().day.at("17:00").do(self.evening_analysis_workflow)  # Evening
        schedule.every().day.at("09:00").do(self.pre_market_workflow)  # Pre-market
        
        logger.info("📅 Scheduled tasks:")
        logger.info("  - Market Close (15:30): ETL Data Collection")
        logger.info("  - Evening (17:00): AI Inference + Strategy Generation")
        logger.info("  - Pre-Market (09:00): Signal Execution")
        logger.info("")
        logger.info("⏳ Scheduler running... (Press Ctrl+C to stop)")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("🛑 Scheduler stopped by user")
    
    def run_manual_workflow(self, workflow_name: str):
        """Run a workflow manually"""
        if workflow_name == "market_close":
            self.market_close_workflow()
        elif workflow_name == "evening":
            self.evening_analysis_workflow()
        elif workflow_name == "pre_market":
            self.pre_market_workflow()
        else:
            logger.error(f"Unknown workflow: {workflow_name}")

if __name__ == "__main__":
    scheduler = TradingScheduler()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        workflow = sys.argv[1]
        scheduler.run_manual_workflow(workflow)
    else:
        scheduler.start_scheduler()

