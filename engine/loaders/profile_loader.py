import pandas as pd
import requests
import io
from database.models import SessionLocal, CompanyProfile, Index
from sqlalchemy import select
from loguru import logger
from typing import Optional, Dict

# Import comprehensive index list
from engine.loaders.nse_index_discovery import ALL_NSE_INDICES, get_index_url, get_all_index_urls

# Get all index URLs
INDEX_URLS: Dict[str, str] = get_all_index_urls()

def sync_index_companies(index_name: str, url: Optional[str] = None):
    """
    Fetches companies from NSE for a specific index and assigns them to that index.
    
    Args:
        index_name: Name of the index (e.g., "NIFTY_50", "NIFTY_100")
        url: Optional custom URL. If not provided, uses INDEX_URLS mapping.
    
    Returns:
        tuple: (new_companies_count, assigned_companies_count)
    """
    session = SessionLocal()
    logger.info(f"🌍 Syncing Company Profiles from NSE for {index_name}...")

    # Get URL
    if url is None:
        url = INDEX_URLS.get(index_name)
        if not url:
            # Try to generate URL dynamically
            url = get_index_url(index_name)
            logger.info(f"Using generated URL for {index_name}: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Get or create index
        index = session.scalar(select(Index).filter_by(name=index_name))
        if not index:
            logger.warning(f"Index {index_name} not found in database. Please run migration first.")
            session.close()
            return 0, 0

        # Fetch the CSV
        logger.info(f"Fetching data from: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Convert raw text to DataFrame
        # The CSV has columns: 'Company Name', 'Industry', 'Symbol', 'Series', 'ISIN Code'
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)

        if df.empty:
            logger.warning(f"No data found in CSV for {index_name}")
            session.close()
            return 0, 0

        new_count = 0
        assigned_count = 0
        total_companies = len(df)

        logger.info(f"Processing {total_companies} companies for {index.display_name}...")

        for idx, row in df.iterrows():
            # NSE CSV Symbols don't have '.NS', so we add it
            symbol = f"{row['Symbol']}.NS"
            
            # Check if company already exists
            stmt = select(CompanyProfile).filter_by(ticker=symbol)
            company = session.scalars(stmt).first()
            
            if not company:
                company = CompanyProfile(
                    ticker=symbol,
                    name=row['Company Name'],
                    sector=row.get('Industry', 'Unknown'),  # NSE CSV provides Industry
                    industry=row.get('Industry', 'Unknown'),
                    exchange="NSE",
                    currency="INR",
                    description=f"ISIN: {row.get('ISIN Code', 'N/A')}"
                )
                session.add(company)
                new_count += 1
            
            # Assign to index if not already assigned
            if index not in company.indices:
                company.indices.append(index)
                assigned_count += 1
        
        session.commit()
        
        if new_count > 0:
            logger.success(f"✅ Added {new_count} new companies to database")
        if assigned_count > 0:
            logger.success(f"✅ Assigned {assigned_count} companies to {index.display_name}")
        if new_count == 0 and assigned_count == 0:
            logger.info(f"👌 {index.display_name} is already up to date ({total_companies} companies)")
        
        return new_count, assigned_count

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to fetch from NSE for {index_name}: {e}")
        logger.warning(f"⚠️ URL: {url}")
        session.rollback()
        return 0, 0
    except Exception as e:
        logger.error(f"❌ Error processing {index_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        session.rollback()
        return 0, 0
    finally:
        session.close()

def sync_nifty_companies(index_name: str = "NIFTY_50"):
    """
    Legacy function for backward compatibility.
    Fetches companies from NSE for the specified index.
    
    Args:
        index_name: Name of the index (default: "NIFTY_50")
    """
    return sync_index_companies(index_name)

def sync_all_indices():
    """
    Sync companies for all available indices.
    
    Returns:
        dict: Summary of sync results for each index
    """
    logger.info("🚀 Starting sync for all indices...")
    
    results = {}
    
    for index_name in INDEX_URLS.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"Syncing {index_name}...")
        logger.info(f"{'='*60}")
        
        new_count, assigned_count = sync_index_companies(index_name)
        results[index_name] = {
            'new_companies': new_count,
            'assigned_companies': assigned_count,
            'success': True
        }
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 SYNC SUMMARY")
    logger.info(f"{'='*60}")
    
    total_new = sum(r['new_companies'] for r in results.values())
    total_assigned = sum(r['assigned_companies'] for r in results.values())
    
    for index_name, result in results.items():
        logger.info(f"{index_name}: {result['new_companies']} new, {result['assigned_companies']} assigned")
    
    logger.info(f"\nTotal: {total_new} new companies, {total_assigned} assignments")
    logger.success("✅ All indices synced!")
    
    return results