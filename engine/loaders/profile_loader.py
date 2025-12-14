import pandas as pd
import requests
import io
from database.models import SessionLocal, CompanyProfile
from sqlalchemy import select
from loguru import logger

def sync_nifty_companies():
    """
    Fetches the official Nifty 50 CSV directly from NSE India archives.
    """
    session = SessionLocal()
    logger.info("🌍 Syncing Company Profiles from NSE Official Source...")

    # Official NSE Archive URL for Nifty 50
    # If this ever changes, you can simply update this string.
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # 1. Fetch the CSV
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 2. Convert raw text to DataFrame
        # The CSV has columns: 'Company Name', 'Industry', 'Symbol', 'Series', 'ISIN Code'
        csv_data = io.StringIO(response.text)
        nifty_df = pd.read_csv(csv_data)

        count = 0
        total = len(nifty_df)

        for index, row in nifty_df.iterrows():
            # NSE CSV Symbols don't have '.NS', so we add it
            symbol = f"{row['Symbol']}.NS"
            
            # Check if company already exists
            stmt = select(CompanyProfile).filter_by(ticker=symbol)
            existing = session.scalars(stmt).first()
            
            if not existing:
                company = CompanyProfile(
                    ticker=symbol,
                    name=row['Company Name'],
                    sector=row['Industry'], # NSE CSV provides Industry directly!
                    industry=row['Industry'],
                    exchange="NSE",
                    currency="INR",
                    description=f"ISIN: {row['ISIN Code']}" # Storing ISIN as description for now
                )
                session.add(company)
                count += 1
        
        session.commit()
        if count > 0:
            logger.success(f"✅ Successfully added {count} new companies from NSE CSV.")
        else:
            logger.info("👌 Database is already up to date.")

    except Exception as e:
        logger.error(f"❌ Failed to fetch from NSE: {e}")
        logger.warning("⚠️ Recommendation: Create a local 'nifty50.csv' file if this persists.")

    finally:
        session.close()