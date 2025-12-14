"""
Migration: Add missing columns to trade_signals table
Adds: priority, reasoning (if they don't exist)
"""

import sqlite3
from pathlib import Path
from loguru import logger

def migrate_database():
    """Add missing columns to trade_signals table"""
    db_path = Path("stock_data.db")
    
    if not db_path.exists():
        logger.error("Database file not found. Please run init_db.py first.")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check existing columns
        cursor.execute("PRAGMA table_info(trade_signals)")
        columns = {column[1]: column for column in cursor.fetchall()}
        
        changes_made = False
        
        # Add priority column if missing
        if 'priority' not in columns:
            logger.info("Adding 'priority' column to trade_signals table...")
            cursor.execute("ALTER TABLE trade_signals ADD COLUMN priority INTEGER DEFAULT 5")
            cursor.execute("UPDATE trade_signals SET priority = 5 WHERE priority IS NULL")
            changes_made = True
        
        # Add reasoning column if missing
        if 'reasoning' not in columns:
            logger.info("Adding 'reasoning' column to trade_signals table...")
            cursor.execute("ALTER TABLE trade_signals ADD COLUMN reasoning TEXT")
            changes_made = True
        
        if changes_made:
            conn.commit()
            logger.success("✅ Successfully added missing columns to trade_signals table")
        else:
            logger.info("✅ All columns already exist in trade_signals table")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        logger.error(f"❌ Database migration failed: {e}")
        return False

if __name__ == "__main__":
    migrate_database()

