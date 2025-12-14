"""
Utility script to sync companies for all indices
"""

from loguru import logger
from engine.loaders.profile_loader import sync_all_indices

if __name__ == "__main__":
    logger.info("🔄 Syncing all indices from NSE...")
    results = sync_all_indices()
    
    # Print summary
    print("\n" + "="*60)
    print("SYNC COMPLETE")
    print("="*60)
    for index_name, result in results.items():
        print(f"{index_name}: {result['new_companies']} new companies, {result['assigned_companies']} assigned")

