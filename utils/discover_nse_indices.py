"""
Utility to discover all available NSE indices and create them in the database
"""

from loguru import logger
from engine.loaders.nse_index_discovery import ALL_NSE_INDICES, discover_available_indices, get_index_url
from database.models import SessionLocal, Index
from sqlalchemy import select

def create_all_indices_in_db():
    """Create all known NSE indices in the database"""
    session = SessionLocal()
    
    try:
        created = 0
        updated = 0
        
        for index_name, index_info in ALL_NSE_INDICES.items():
            # Check if index exists
            existing = session.scalar(select(Index).filter_by(name=index_name))
            
            if not existing:
                # Create new index
                new_index = Index(
                    name=index_name,
                    display_name=index_info['display_name'],
                    description=f"Companies from {index_info['display_name']} index",
                    is_active=True
                )
                session.add(new_index)
                created += 1
                logger.info(f"✅ Created index: {index_info['display_name']}")
            else:
                # Update existing index if needed
                if existing.display_name != index_info['display_name']:
                    existing.display_name = index_info['display_name']
                    updated += 1
                    logger.info(f"🔄 Updated index: {index_info['display_name']}")
        
        session.commit()
        logger.success(f"✅ Created {created} new indices, updated {updated} existing indices")
        
        return created, updated
        
    except Exception as e:
        logger.error(f"❌ Failed to create indices: {e}")
        session.rollback()
        return 0, 0
    finally:
        session.close()

def discover_and_create_indices():
    """Discover available indices and create them in database"""
    logger.info("🔍 Discovering and creating all NSE indices...")
    
    # First, create all known indices in database
    created, updated = create_all_indices_in_db()
    
    # Then discover which ones are actually available
    logger.info("\n🔍 Checking which indices are available on NSE...")
    available = discover_available_indices()
    
    available_count = sum(1 for _, _, avail in available if avail)
    logger.info(f"\n📊 Summary: {available_count}/{len(available)} indices are available on NSE")
    
    return available

if __name__ == "__main__":
    discover_and_create_indices()

