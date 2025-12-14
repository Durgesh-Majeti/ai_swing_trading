from engine.loaders.profile_loader import sync_nifty_companies
from engine.loaders.price_loader import sync_price_history
# from database.models import init_db # Assuming you moved init logic here or import from init_db

def main():
    print("--- 🏭 STOCK ENGINE ETL PIPELINE ---")
    
    # Step 1: Ensure Tables Exist
    # (You can run init_db.py manually, or call it here)
    
    # Step 2: Seed Company List
    cmd = input("1. Sync Company List (Nifty 50)? (y/n): ")
    if cmd.lower() == 'y':
        sync_nifty_companies()

    # Step 3: Hydrate Prices
    cmd = input("2. Sync Market Data (OHLC)? (y/n): ")
    if cmd.lower() == 'y':
        sync_price_history()

    print("\n✅ Data Pipeline Complete.")

if __name__ == "__main__":
    main()