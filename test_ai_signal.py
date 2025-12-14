"""
Test script to diagnose AI Signal Strategy issues
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from strategies.registry import StrategyRegistry
from database.models import SessionLocal, Index
from sqlalchemy import select
from loguru import logger

def test_ai_signal_strategy():
    """Test AI Signal Strategy signal generation"""
    
    session = SessionLocal()
    
    try:
        # Get NIFTY 50 index
        index = session.scalar(select(Index).filter_by(name="NIFTY_50", is_active=True))
        if not index:
            logger.error("NIFTY_50 index not found")
            return
        
        logger.info(f"Testing AI Signal Strategy for {index.display_name}")
        
        # Create strategy registry with index
        registry = StrategyRegistry(index_id=index.id, index_name=index.name)
        
        # Get AI Signal Strategy
        strategy = registry.get_strategy("AI_Signal_Strategy")
        if not strategy:
            logger.error("AI_Signal_Strategy not found in registry")
            logger.info(f"Available strategies: {registry.list_strategies()}")
            return
        
        logger.info("✅ AI_Signal_Strategy loaded")
        
        # Check if models are loaded
        logger.info(f"Signal model loaded: {strategy.signal_model is not None}")
        logger.info(f"Return model loaded: {strategy.return_model is not None}")
        logger.info(f"Target price model loaded: {strategy.target_price_model is not None}")
        
        if not (strategy.signal_model or strategy.return_model or strategy.target_price_model):
            logger.error("❌ No models loaded!")
            return
        
        # Test with a sample ticker from the index
        if index.companies:
            test_ticker = index.companies[0].ticker
            logger.info(f"Testing with ticker: {test_ticker}")
            
            # Try to generate signal
            signal = strategy.generate_signal(test_ticker)
            
            if signal:
                logger.success(f"✅ Signal generated successfully!")
                logger.info(f"Signal: {signal.get('signal')}")
                logger.info(f"Entry Price: ₹{signal.get('entry_price'):.2f}")
                logger.info(f"Stop Loss: ₹{signal.get('stop_loss'):.2f}")
                logger.info(f"Target Price: ₹{signal.get('target_price'):.2f}")
                logger.info(f"Quantity: {signal.get('quantity')}")
                logger.info(f"Profit Score: {signal.get('profit_score'):.1f}")
                logger.info(f"Confidence: {signal.get('confidence_score'):.1%}")
            else:
                logger.warning("⚠️  No signal generated (might be HOLD or filtered out)")
        else:
            logger.warning("No companies in index to test with")
        
        strategy.close()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        session.close()

if __name__ == "__main__":
    test_ai_signal_strategy()

