"""
Migration: Add quantity column to trade_signals table
"""

import sqlite3
from pathlib import Path
from loguru import logger

def migrate_database():
    """Add quantity column to trade_signals table if it doesn't exist"""
    db_path = Path("stock_data.db")
    
    if not db_path.exists():
        logger.error("Database file not found. Please run init_db.py first.")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(trade_signals)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'quantity' in columns:
            logger.info("✅ Column 'quantity' already exists in trade_signals table")
            conn.close()
            return True
        
        # Add the quantity column
        logger.info("Adding 'quantity' column to trade_signals table...")
        cursor.execute("ALTER TABLE trade_signals ADD COLUMN quantity INTEGER")
        
        # Set default value for existing rows
        cursor.execute("UPDATE trade_signals SET quantity = 10 WHERE quantity IS NULL")
        
        conn.commit()
        conn.close()
        
        logger.success("✅ Successfully added 'quantity' column to trade_signals table")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"❌ Database migration failed: {e}")
        return False

if __name__ == "__main__":
    migrate_database()

