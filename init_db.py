from database.models import Base, engine
from loguru import logger

def initialize_database():
    logger.info("🔨 Building Database Schema...")
    
    # This command checks your models.py and creates the tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    logger.success("✅ Database 'stock_data.db' is ready with 7 tables.")

if __name__ == "__main__":
    initialize_database()