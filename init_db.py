from database.models import Base, engine
from loguru import logger

def initialize_database():
    logger.info("🔨 Building Database Schema...")
    
    # This command checks your models.py and creates the tables if they don't exist
    # Note: This only creates missing tables, not missing columns in existing tables
    # For schema updates, use migration scripts in migrations/ directory
    Base.metadata.create_all(bind=engine)
    
    # Count actual tables created
    from sqlalchemy import inspect
    inspector = inspect(engine)
    table_count = len(inspector.get_table_names())
    
    logger.success(f"✅ Database 'stock_data.db' is ready with {table_count} tables.")
    logger.info("💡 If you need to update existing tables, run migration scripts from migrations/")

if __name__ == "__main__":
    initialize_database()