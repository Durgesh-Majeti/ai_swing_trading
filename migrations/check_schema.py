"""
Check and update database schema to match models
"""

import sqlite3
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import Base, engine
from sqlalchemy import inspect

def check_and_update_schema():
    """Check if database schema matches models and update if needed"""
    db_path = Path("stock_data.db")
    
    if not db_path.exists():
        logger.warning("Database file not found. Creating new database...")
        Base.metadata.create_all(bind=engine)
        logger.success("✅ Database created with all tables")
        return True
    
    try:
        # Use SQLAlchemy to check schema
        inspector = inspect(engine)
        
        # Get all tables from models
        expected_tables = Base.metadata.tables.keys()
        existing_tables = inspector.get_table_names()
        
        # Check for missing tables
        missing_tables = set(expected_tables) - set(existing_tables)
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            logger.info("Creating missing tables...")
            Base.metadata.create_all(bind=engine)
            logger.success("✅ Missing tables created")
        
        # Check for missing columns in existing tables
        issues_found = False
        for table_name in expected_tables:
            if table_name in existing_tables:
                expected_columns = {col.name for col in Base.metadata.tables[table_name].columns}
                existing_columns = {col['name'] for col in inspector.get_columns(table_name)}
                
                missing_columns = expected_columns - existing_columns
                if missing_columns:
                    logger.warning(f"Table '{table_name}' missing columns: {missing_columns}")
                    issues_found = True
        
        if not issues_found and not missing_tables:
            logger.success("✅ Database schema is up to date")
        else:
            logger.info("⚠️  Schema differences found. Run migrations or recreate database.")
            logger.info("   To recreate: Delete stock_data.db and run init_db.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Schema check failed: {e}")
        return False

if __name__ == "__main__":
    check_and_update_schema()

