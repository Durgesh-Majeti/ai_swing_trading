"""
NSE Index Discovery - Discover and map all available NSE indices
"""

import requests
from typing import Dict, List, Tuple
from loguru import logger

# Base URL pattern for NSE indices
BASE_URL = "https://nsearchives.nseindia.com/content/indices/"

def generate_index_url(index_name: str) -> str:
    """
    Generate NSE CSV URL from index name.
    
    Converts index names like "NIFTY_50" to "ind_nifty50list.csv"
    """
    # Convert to lowercase and replace underscores
    name_lower = index_name.lower().replace("_", "")
    
    # Handle special cases
    name_mapping = {
        "nifty50": "nifty50",
        "nifty100": "nifty100",
        "nifty200": "nifty200",
        "nifty500": "nifty500",
        "niftymidcap50": "niftymidcap50",
        "niftymidcap100": "niftymidcap100",
        "niftymidcap150": "niftymidcap150",
        "niftymidcap250": "niftymidcap250",
        "niftysmallcap50": "niftysmallcap50",
        "niftysmallcap100": "niftysmallcap100",
        "niftysmallcap250": "niftysmallcap250",
        "niftylargemidcap250": "niftylargemidcap250",
        "niftymicrocap250": "niftymicrocap250",
        "niftytotalmarket": "niftytotalmarket",
    }
    
    # Use mapping if available, otherwise use the converted name
    csv_name = name_mapping.get(name_lower, name_lower)
    
    return f"{BASE_URL}ind_{csv_name}list.csv"

# Comprehensive list of all NSE indices
ALL_NSE_INDICES: Dict[str, Dict[str, str]] = {
    # Benchmark Indices
    "NIFTY_50": {"display_name": "Nifty 50", "url": "ind_nifty50list.csv"},
    "NIFTY_100": {"display_name": "Nifty 100", "url": "ind_nifty100list.csv"},
    "NIFTY_200": {"display_name": "Nifty 200", "url": "ind_nifty200list.csv"},
    "NIFTY_500": {"display_name": "Nifty 500", "url": "ind_nifty500list.csv"},
    "NIFTY_TOTAL_MARKET": {"display_name": "Nifty Total Market", "url": "ind_niftytotalmarketlist.csv"},
    
    # Market Cap Based
    "NIFTY_LARGEMIDCAP_250": {"display_name": "Nifty LargeMidcap 250", "url": "ind_niftylargemidcap250list.csv"},
    "NIFTY_MIDCAP_50": {"display_name": "Nifty Midcap 50", "url": "ind_niftymidcap50list.csv"},
    "NIFTY_MIDCAP_100": {"display_name": "Nifty Midcap 100", "url": "ind_niftymidcap100list.csv"},
    "NIFTY_MIDCAP_150": {"display_name": "Nifty Midcap 150", "url": "ind_niftymidcap150list.csv"},
    "NIFTY_MIDCAP_250": {"display_name": "Nifty Midcap 250", "url": "ind_niftymidcap250list.csv"},
    "NIFTY_SMALLCAP_50": {"display_name": "Nifty Smallcap 50", "url": "ind_niftysmallcap50list.csv"},
    "NIFTY_SMALLCAP_100": {"display_name": "Nifty Smallcap 100", "url": "ind_niftysmallcap100list.csv"},
    "NIFTY_SMALLCAP_250": {"display_name": "Nifty Smallcap 250", "url": "ind_niftysmallcap250list.csv"},
    "NIFTY_MICROCAP_250": {"display_name": "Nifty Microcap 250", "url": "ind_niftymicrocap250list.csv"},
    
    # Sectoral Indices
    "NIFTY_AUTO": {"display_name": "Nifty Auto", "url": "ind_niftyautolist.csv"},
    "NIFTY_BANK": {"display_name": "Nifty Bank", "url": "ind_niftybanklist.csv"},
    "NIFTY_ENERGY": {"display_name": "Nifty Energy", "url": "ind_niftyenergylist.csv"},
    "NIFTY_FINANCIAL_SERVICES": {"display_name": "Nifty Financial Services", "url": "ind_niftyfinancialserviceslist.csv"},
    "NIFTY_FMCG": {"display_name": "Nifty FMCG", "url": "ind_niftyfmcglist.csv"},
    "NIFTY_HEALTHCARE": {"display_name": "Nifty Healthcare", "url": "ind_niftyhealthcarelist.csv"},
    "NIFTY_IT": {"display_name": "Nifty IT", "url": "ind_niftyitlist.csv"},
    "NIFTY_MEDIA": {"display_name": "Nifty Media", "url": "ind_niftymedialist.csv"},
    "NIFTY_METAL": {"display_name": "Nifty Metal", "url": "ind_niftymetallist.csv"},
    "NIFTY_PHARMA": {"display_name": "Nifty Pharma", "url": "ind_niftypharmalist.csv"},
    "NIFTY_PSU_BANK": {"display_name": "Nifty PSU Bank", "url": "ind_niftypsubanklist.csv"},
    "NIFTY_PVT_BANK": {"display_name": "Nifty Private Bank", "url": "ind_niftypvtbanklist.csv"},
    "NIFTY_REALTY": {"display_name": "Nifty Realty", "url": "ind_niftyrealtylist.csv"},
    "NIFTY_CONSUMER_DURABLES": {"display_name": "Nifty Consumer Durables", "url": "ind_niftyconsumerdurbleslist.csv"},
    "NIFTY_OIL_GAS": {"display_name": "Nifty Oil & Gas", "url": "ind_niftyoilgaslist.csv"},
    "NIFTY_INFRASTRUCTURE": {"display_name": "Nifty Infrastructure", "url": "ind_niftyinfrastructurelist.csv"},
    "NIFTY_COMMODITIES": {"display_name": "Nifty Commodities", "url": "ind_niftycommoditieslist.csv"},
    "NIFTY_SERVICES_SECTOR": {"display_name": "Nifty Services Sector", "url": "ind_niftyservicessectorlist.csv"},
    
    # Thematic Indices
    "NIFTY_ADITYA_BIRLA_GROUP": {"display_name": "Nifty Aditya Birla Group", "url": "ind_niftyadityabirlagrouplist.csv"},
    "NIFTY_CPSE": {"display_name": "Nifty CPSE", "url": "ind_niftycpselist.csv"},
    "NIFTY_DIVIDEND_OPPORTUNITIES_50": {"display_name": "Nifty Dividend Opportunities 50", "url": "ind_niftydividendopportunities50list.csv"},
    "NIFTY_GROWTH_SECTORS_15": {"display_name": "Nifty Growth Sectors 15", "url": "ind_niftygrowthsectors15list.csv"},
    "NIFTY_INDIA_CONSUMPTION": {"display_name": "Nifty India Consumption", "url": "ind_niftyindiaconsumptionlist.csv"},
    "NIFTY_INDIA_DIGITAL": {"display_name": "Nifty India Digital", "url": "ind_niftyindiadigitallist.csv"},
    "NIFTY_INDIA_MANUFACTURING": {"display_name": "Nifty India Manufacturing", "url": "ind_niftyindiamanufacturinglist.csv"},
    "NIFTY_MNC": {"display_name": "Nifty MNC", "url": "ind_niftymnclist.csv"},
    "NIFTY_NEXT_50": {"display_name": "Nifty Next 50", "url": "ind_niftynext50list.csv"},
    "NIFTY_PSE": {"display_name": "Nifty PSE", "url": "ind_niftypselist.csv"},
    "NIFTY_QUALITY_30": {"display_name": "Nifty Quality 30", "url": "ind_niftyquality30list.csv"},
    "NIFTY_SHARIAH_25": {"display_name": "Nifty Shariah 25", "url": "ind_niftyshariah25list.csv"},
    "NIFTY_TATA_GROUP": {"display_name": "Nifty Tata Group", "url": "ind_niftytatagrouplist.csv"},
    "NIFTY_MAHINDRA_GROUP": {"display_name": "Nifty Mahindra Group", "url": "ind_niftymahindragrouplist.csv"},
    
    # Strategy Indices
    "NIFTY_ALPHA_50": {"display_name": "Nifty Alpha 50", "url": "ind_niftyalpha50list.csv"},
    "NIFTY_HIGH_BETA_50": {"display_name": "Nifty High Beta 50", "url": "ind_niftyhighbeta50list.csv"},
    "NIFTY_LOW_VOLATILITY_50": {"display_name": "Nifty Low Volatility 50", "url": "ind_niftylowvolatility50list.csv"},
    "NIFTY_MOMENTUM_50": {"display_name": "Nifty Momentum 50", "url": "ind_niftymomentum50list.csv"},
    "NIFTY_QUALITY_LOW_VOLATILITY_30": {"display_name": "Nifty Quality Low Volatility 30", "url": "ind_niftyqualitylowvolatility30list.csv"},
    "NIFTY_100_EQUAL_WEIGHT": {"display_name": "Nifty 100 Equal Weight", "url": "ind_nifty100equalweightlist.csv"},
    "NIFTY_50_EQUAL_WEIGHT": {"display_name": "Nifty 50 Equal Weight", "url": "ind_nifty50equalweightlist.csv"},
    "NIFTY_500_EQUAL_WEIGHT": {"display_name": "Nifty 500 Equal Weight", "url": "ind_nifty500equalweightlist.csv"},
}

def get_index_url(index_name: str) -> str:
    """Get full URL for an index"""
    if index_name in ALL_NSE_INDICES:
        return f"{BASE_URL}{ALL_NSE_INDICES[index_name]['url']}"
    else:
        # Try to generate URL dynamically
        return generate_index_url(index_name)

def discover_available_indices() -> List[Tuple[str, str, bool]]:
    """
    Discover which indices are actually available on NSE by testing URLs.
    
    Returns:
        List of tuples: (index_name, display_name, is_available)
    """
    logger.info("🔍 Discovering available NSE indices...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    available = []
    
    for index_name, index_info in ALL_NSE_INDICES.items():
        url = f"{BASE_URL}{index_info['url']}"
        
        try:
            response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
            if response.status_code == 200:
                available.append((index_name, index_info['display_name'], True))
                logger.debug(f"✅ {index_name} is available")
            else:
                available.append((index_name, index_info['display_name'], False))
                logger.debug(f"❌ {index_name} returned {response.status_code}")
        except Exception as e:
            available.append((index_name, index_info['display_name'], False))
            logger.debug(f"❌ {index_name} failed: {e}")
    
    available_count = sum(1 for _, _, avail in available if avail)
    logger.info(f"📊 Found {available_count}/{len(available)} indices available")
    
    return available

def get_all_index_urls() -> Dict[str, str]:
    """Get mapping of all index names to their URLs"""
    return {
        name: f"{BASE_URL}{info['url']}"
        for name, info in ALL_NSE_INDICES.items()
    }

