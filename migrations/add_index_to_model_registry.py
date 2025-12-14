"""
Migration: Add index_id to ModelRegistry table for index-specific model filtering

This migration adds index_id foreign key to ModelRegistry to ensure
models are only visible for their specific index.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import SessionLocal, ModelRegistry, Index
from sqlalchemy import text, select
from loguru import logger
import re

def extract_index_from_model_name(model_name: str) -> str:
    """Extract index name from model name (e.g., 'SignalClassifier_NIFTY50_v1' -> 'NIFTY_50')"""
    # Pattern: ModelType_INDEXNAME_v1
    # Examples: SignalClassifier_NIFTY50_v1, ReturnPredictor_NIFTY50_v1
    # We need to extract NIFTY50 and convert to NIFTY_50 for database lookup
    
    # Try to find NIFTY followed by digits
    match = re.search(r'NIFTY(\d+)', model_name.upper())
    if match:
        number = match.group(1)  # e.g., "50"
        return f"NIFTY_{number}"  # Return "NIFTY_50" for database lookup
    
    return None

def run_migration():
    """Add index_id column to ModelRegistry table"""
    session = SessionLocal()
    
    try:
        logger.info("🔧 Starting ModelRegistry index_id migration...")
        
        # Get database connection
        connection = session.connection()
        
        # Add index_id to ModelRegistry table
        logger.info("Adding index_id to model_registry table...")
        try:
            connection.execute(text("ALTER TABLE model_registry ADD COLUMN index_id INTEGER"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_model_registry_index_id ON model_registry(index_id)"))
            logger.info("✅ Added index_id to model_registry")
        except Exception as e:
            if "duplicate column name" not in str(e).lower() and "already exists" not in str(e).lower():
                logger.warning(f"ModelRegistry index_id may already exist: {e}")
        
        # Commit the schema change
        session.commit()
        
        # Now populate index_id for existing models based on model name
        logger.info("Populating index_id for existing models...")
        
        # Get all models without index_id
        from sqlalchemy import select
        models = session.scalars(select(ModelRegistry).filter_by(index_id=None)).all()
        
        updated_count = 0
        for model in models:
            model_name = model.model_name
            
            # Extract index name from model name
            index_name = extract_index_from_model_name(model_name)
            
            if index_name:
                # Find matching index - try different name formats
                index = session.scalar(select(Index).filter_by(name=index_name))
                if not index:
                    # Try with underscore
                    index_name_with_underscore = index_name.replace("NIFTY", "NIFTY_")
                    index = session.scalar(select(Index).filter_by(name=index_name_with_underscore))
                
                if index:
                    model.index_id = index.id
                    updated_count += 1
                    logger.info(f"✅ Updated model {model_name} with index_id {index.id}")
                else:
                    logger.warning(f"⚠️  Could not find index for model {model_name} (extracted: {index_name})")
            else:
                logger.debug(f"No index pattern found in model name: {model_name}")
        
        session.commit()
        logger.success(f"✅ Migration complete! Updated {updated_count} models with index_id")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    run_migration()

