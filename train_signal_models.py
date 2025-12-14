"""
Training Script for Signal Generation Models
Trains models that can generate trading signals with profit potential
Supports index-specific model training
"""

import sys
from loguru import logger
from ai.signal_models import SignalModelTrainer
from database.models import SessionLocal, Index
from sqlalchemy import select


def normalize_index_name(index_name: str) -> str:
    """
    Normalize index name for use in model names
    e.g., "NIFTY_50" -> "NIFTY50", "NIFTY 100" -> "NIFTY100"
    """
    if not index_name:
        return ""
    # Remove spaces and underscores, convert to uppercase
    normalized = index_name.replace(" ", "").replace("_", "").upper()
    return normalized


def main():
    """Main training function"""
    logger.info("🚀 Starting Signal Model Training...")
    
    # Check if index is specified
    index_name = None
    if len(sys.argv) > 1:
        index_name = sys.argv[1].upper()
        logger.info(f"Training models for index: {index_name}")
    
    # Get index_id if index_name is provided
    index_id = None
    index_display_name = None
    if index_name:
        session = SessionLocal()
        index = session.scalar(select(Index).filter_by(name=index_name, is_active=True))
        if index:
            index_id = index.id
            index_display_name = index.display_name
            logger.info(f"Found index: {index_display_name} (ID: {index_id})")
        else:
            logger.error(f"❌ Index {index_name} not found or inactive. Please check the index name.")
            session.close()
            return
        session.close()
    else:
        logger.warning("⚠️  No index specified. Training for all indices (not recommended).")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Training cancelled.")
            return
    
    # Create trainer
    trainer = SignalModelTrainer(index_id=index_id)
    
    # Normalize index name for model naming
    normalized_index_name = normalize_index_name(index_name) if index_name else ""
    
    try:
        # Train all models with index-specific names
        models = trainer.train_all_models(index_name=normalized_index_name)
        
        if models["signal_classifier"]:
            logger.success("✅ Signal Classifier trained successfully")
        else:
            logger.warning("⚠️  Signal Classifier training failed")
        
        if models["return_predictor"]:
            logger.success("✅ Return Predictor trained successfully")
        else:
            logger.warning("⚠️  Return Predictor training failed")
        
        if models["target_price_predictor"]:
            logger.success("✅ Target Price Predictor trained successfully")
        else:
            logger.warning("⚠️  Target Price Predictor training failed")
        
        logger.success("🎉 Training complete!")
        logger.info("\nNext steps:")
        logger.info("1. Review model performance metrics in the database")
        if normalized_index_name:
            logger.info(f"2. Activate models using:")
            logger.info(f"   python -m ai.model_registry activate SignalClassifier_{normalized_index_name}_v1")
            logger.info(f"   python -m ai.model_registry activate ReturnPredictor_{normalized_index_name}_v1")
            logger.info(f"   python -m ai.model_registry activate TargetPricePredictor_{normalized_index_name}_v1")
        logger.info("3. The AI_Signal_Strategy will automatically use index-specific models")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

