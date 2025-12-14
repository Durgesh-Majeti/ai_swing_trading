"""
Migration: Add strategy_metadata table
Adds table to store strategy documentation and metadata
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import Base, StrategyMetadata, engine
from sqlalchemy import inspect, select
from loguru import logger
from datetime import datetime

def run_migration():
    """Add strategy_metadata table if it doesn't exist"""
    logger.info("🔄 Running migration: Add strategy_metadata table")
    
    # Check if table exists
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    if "strategy_metadata" in existing_tables:
        logger.info("✅ Table 'strategy_metadata' already exists")
        return
    
    # Create table
    logger.info("Creating strategy_metadata table...")
    StrategyMetadata.__table__.create(engine, checkfirst=True)
    logger.success("✅ Created strategy_metadata table")
    
    # Initialize default strategy metadata
    logger.info("Initializing default strategy metadata...")
    from database.models import SessionLocal
    
    session = SessionLocal()
    
    try:
        # Technical Strategy
        if not session.scalar(select(StrategyMetadata).filter_by(strategy_name="Technical_RSI_MACD")):
            technical_meta = StrategyMetadata(
                strategy_name="Technical_RSI_MACD",
                display_name="Technical Analysis (RSI + MACD)",
                category="Technical",
                description="A technical analysis-based strategy that uses RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), and moving averages to identify entry and exit points.",
                how_it_works="The strategy generates BUY signals when RSI is below 30 (oversold condition) and MACD shows a bullish crossover. SELL signals are generated when RSI is above 70 (overbought condition) and MACD shows a bearish crossover. Moving averages are used to confirm the trend direction.",
                entry_conditions="BUY Signal: RSI < 30 (oversold) AND MACD line crosses above signal line (bullish) AND price is above 20-day SMA (uptrend confirmation)",
                exit_conditions="SELL Signal: RSI > 70 (overbought) OR MACD line crosses below signal line (bearish) OR stop loss/target price is hit",
                risk_management="Uses stop loss at 2% below entry price and target price at 4% above entry price. Position sizing is based on available capital and risk tolerance.",
                recommended_timeframe="Swing Trading",
                risk_level="Medium",
                author="System",
                version="1.0"
            )
            session.add(technical_meta)
        
        # Hybrid Strategy
        if not session.scalar(select(StrategyMetadata).filter_by(strategy_name="Hybrid_AI_Technical_Fundamental")):
            hybrid_meta = StrategyMetadata(
                strategy_name="Hybrid_AI_Technical_Fundamental",
                display_name="Hybrid AI Strategy",
                category="Hybrid",
                description="A comprehensive strategy that combines technical analysis, fundamental analysis, and AI predictions to generate high-confidence trading signals. The strategy dynamically adjusts weights based on signal availability.",
                how_it_works="The strategy combines three signal sources: (1) Technical indicators (RSI, MACD, moving averages), (2) Fundamental metrics (P/E ratio, debt-to-equity, profit margins), and (3) AI predictions (if available). Each source contributes a weighted score, and the final signal is generated when the combined score exceeds a threshold. The strategy gracefully handles AI model unavailability by redistributing weights to technical and fundamental signals.",
                entry_conditions="BUY Signal: Combined weighted score > threshold (default 0.6). Score components: Technical (40%), Fundamental (30%), AI (30% if available, else redistributed). All components must be positive.",
                exit_conditions="SELL Signal: Combined weighted score < -threshold OR stop loss/target price is hit OR fundamental/technical conditions deteriorate",
                risk_management="Uses stop loss at 2% below entry price and target price at 4% above entry price. Position sizing considers volatility (VIX) and available capital. Higher VIX reduces position size.",
                recommended_timeframe="Swing Trading",
                risk_level="Medium",
                author="System",
                version="1.0"
            )
            session.add(hybrid_meta)
        
        session.commit()
        logger.success("✅ Initialized default strategy metadata")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize default metadata: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    run_migration()

