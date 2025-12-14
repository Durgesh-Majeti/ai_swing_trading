"""
ETL Module - The "Hunter-Gatherer"
Wakes up at specific times to collect data from various sources.
"""

import yfinance as yf
from database.models import (
    SessionLocal, CompanyProfile, MarketData, FundamentalData, 
    MacroIndicator, Watchlist, TechnicalIndicators, Index
)
from sqlalchemy import select, func
from loguru import logger
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Union
import pandas_ta as ta

class ETLModule:
    """Main ETL orchestrator"""
    
    def __init__(self, index_id: Optional[int] = None, index_name: Optional[str] = None):
        """
        Initialize ETL Module
        
        Args:
            index_id: Optional index ID to filter by
            index_name: Optional index name (e.g., "NIFTY_50") - used if index_id not provided
        """
        self.session = SessionLocal()
        
        # Resolve index_id from index_name if needed
        if index_id is None and index_name:
            index = self.session.scalar(select(Index).filter_by(name=index_name, is_active=True))
            if index:
                index_id = index.id
                logger.info(f"Resolved index {index_name} to ID {index_id}")
            else:
                logger.warning(f"Index {index_name} not found")
        
        self.index_id = index_id
    
    def run_full_sync(
        self,
        years: Optional[float] = None,
        start_date: Optional[Union[date, datetime]] = None,
        end_date: Optional[Union[date, datetime]] = None,
        force_refresh: bool = False
    ):
        """
        Run complete data sync
        
        Args:
            years: Number of years of historical data to fetch
            start_date: Start date for data range
            end_date: End date for data range
            force_refresh: If True, re-downloads all data
        """
        index_name = None
        if self.index_id:
            index = self.session.scalar(select(Index).filter_by(id=self.index_id))
            index_name = index.display_name if index else None
        
        logger.info(f"🚀 Starting Full ETL Pipeline{' for ' + index_name if index_name else ''}...")
        
        try:
            # 1. Sync Market Data
            self.sync_market_data(
                years=years,
                start_date=start_date,
                end_date=end_date,
                force_refresh=force_refresh
            )
            
            # 2. Sync Macro Indicators
            self.sync_macro_indicators()
            
            # 3. Calculate Technical Indicators
            self.calculate_technical_indicators()
            
            # 4. Sync Fundamentals (if available)
            # self.sync_fundamentals()  # Can be implemented later
            
            logger.success("✅ ETL Pipeline Complete")
        except Exception as e:
            logger.error(f"❌ ETL Pipeline Failed: {e}")
            self.session.rollback()
        finally:
            self.session.close()
    
    def sync_market_data(
        self, 
        years: Optional[float] = None,
        start_date: Optional[Union[date, datetime]] = None,
        end_date: Optional[Union[date, datetime]] = None,
        force_refresh: bool = False
    ):
        """
        Fetch daily OHLCV data for watchlist stocks (filtered by index if specified)
        
        Args:
            years: Number of years of historical data to fetch (e.g., 5.0 for 5 years)
            start_date: Start date for data range (overrides years if provided)
            end_date: End date for data range (defaults to today if not provided)
            force_refresh: If True, re-downloads all data even if it exists in DB
        """
        logger.info("📈 Syncing Market Data...")
        
        # Get active watchlist, filtered by index if specified
        if self.index_id:
            watchlist = self.session.scalars(
                select(Watchlist).filter_by(is_active=True, index_id=self.index_id)
            ).all()
            if not watchlist:
                # Fallback: get companies from index
                index = self.session.scalar(select(Index).filter_by(id=self.index_id))
                if index:
                    tickers = [c.ticker for c in index.companies]
                else:
                    tickers = []
            else:
                tickers = [w.ticker for w in watchlist]
        else:
            # Get all active watchlist
            watchlist = self.session.scalars(
                select(Watchlist).filter_by(is_active=True)
            ).all()
            
            if not watchlist:
                # If no watchlist, sync all companies
                companies = self.session.scalars(select(CompanyProfile)).all()
                tickers = [c.ticker for c in companies]
            else:
                tickers = [w.ticker for w in watchlist]
        
        logger.info(f"Syncing {len(tickers)} tickers...")
        
        # Determine download parameters
        if start_date and end_date:
            # Use date range
            period_str = None
            if isinstance(start_date, date):
                start_date = datetime.combine(start_date, datetime.min.time())
            if isinstance(end_date, date):
                end_date = datetime.combine(end_date, datetime.min.time())
        elif years:
            # Use period
            period_str = f"{int(years)}y" if years >= 1 else f"{int(years * 365)}d"
            start_date = None
            end_date = None
        else:
            # Default: 1 year
            period_str = "1y"
            start_date = None
            end_date = None
        
        for ticker in tickers:
            try:
                # Fetch data based on parameters
                if period_str:
                    df = yf.download(ticker, period=period_str, interval="1d", progress=False)
                else:
                    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
                
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue
                
                # Handle multi-index columns (yfinance sometimes returns multi-index)
                if isinstance(df.columns, pd.MultiIndex):
                    # Flatten multi-index columns - take first level
                    df.columns = df.columns.get_level_values(0)
                
                # Check latest date in DB (unless force_refresh)
                last_entry = None
                if not force_refresh:
                    stmt = select(MarketData).filter_by(ticker=ticker).order_by(MarketData.date.desc())
                    last_entry = self.session.scalars(stmt).first()
                
                new_records = []
                for date_idx, row in df.iterrows():
                    # Convert to date object
                    if isinstance(date_idx, pd.Timestamp):
                        date_val = date_idx.date()
                    elif isinstance(date_idx, datetime):
                        date_val = date_idx.date()
                    elif isinstance(date_idx, date):
                        date_val = date_idx
                    else:
                        # Try to convert
                        try:
                            date_val = pd.to_datetime(date_idx).date()
                        except:
                            logger.warning(f"Could not parse date {date_idx} for {ticker}")
                            continue
                    
                    # Skip if we already have this date (unless force_refresh)
                    if not force_refresh and last_entry:
                        if date_val <= last_entry.date:
                            continue
                    
                    # Extract scalar values from row (handle Series if multi-index)
                    open_val = row['Open']
                    high_val = row['High']
                    low_val = row['Low']
                    close_val = row['Close']
                    volume_val = row['Volume']
                    
                    # Convert Series to scalar if needed
                    if isinstance(open_val, pd.Series):
                        open_val = open_val.iloc[0] if len(open_val) > 0 else 0
                    if isinstance(high_val, pd.Series):
                        high_val = high_val.iloc[0] if len(high_val) > 0 else 0
                    if isinstance(low_val, pd.Series):
                        low_val = low_val.iloc[0] if len(low_val) > 0 else 0
                    if isinstance(close_val, pd.Series):
                        close_val = close_val.iloc[0] if len(close_val) > 0 else 0
                    if isinstance(volume_val, pd.Series):
                        volume_val = volume_val.iloc[0] if len(volume_val) > 0 else 0
                    
                    record = MarketData(
                        ticker=ticker,
                        date=date_val,
                        open=float(open_val) if pd.notna(open_val) else 0.0,
                        high=float(high_val) if pd.notna(high_val) else 0.0,
                        low=float(low_val) if pd.notna(low_val) else 0.0,
                        close=float(close_val) if pd.notna(close_val) else 0.0,
                        volume=int(volume_val) if pd.notna(volume_val) else 0
                    )
                    new_records.append(record)
                
                if new_records:
                    self.session.add_all(new_records)
                    self.session.commit()
                    logger.success(f"✅ {ticker}: +{len(new_records)} days")
                else:
                    logger.info(f"⏭️  {ticker}: Up to date")
                    
            except Exception as e:
                logger.error(f"❌ Failed {ticker}: {e}")
                self.session.rollback()
    
    def sync_macro_indicators(self):
        """Fetch macro-economic indicators"""
        logger.info("🌍 Syncing Macro Indicators...")
        
        today = datetime.now().date()
        
        # India VIX
        try:
            vix_ticker = "^INDIAVIX"
            df = yf.download(vix_ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                latest = df.iloc[-1]
                self._save_macro_indicator("INDIA_VIX", latest['Close'], today, "points")
                logger.success("✅ India VIX updated")
        except Exception as e:
            logger.warning(f"⚠️  Failed to fetch VIX: {e}")
        
        # Crude Oil (WTI)
        try:
            crude_ticker = "CL=F"  # WTI Crude
            df = yf.download(crude_ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                latest = df.iloc[-1]
                self._save_macro_indicator("CRUDE_OIL", latest['Close'], today, "USD/barrel")
                logger.success("✅ Crude Oil updated")
        except Exception as e:
            logger.warning(f"⚠️  Failed to fetch Crude: {e}")
        
        # USD/INR
        try:
            usdinr_ticker = "INR=X"
            df = yf.download(usdinr_ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                latest = df.iloc[-1]
                self._save_macro_indicator("USD_INR", latest['Close'], today, "INR")
                logger.success("✅ USD/INR updated")
        except Exception as e:
            logger.warning(f"⚠️  Failed to fetch USD/INR: {e}")
    
    def _save_macro_indicator(self, name: str, value: float, date_val: date, unit: str):
        """Save or update macro indicator"""
        stmt = select(MacroIndicator).filter_by(
            indicator_name=name,
            date=datetime.combine(date_val, datetime.min.time())
        )
        existing = self.session.scalars(stmt).first()
        
        if existing:
            existing.value = value
        else:
            indicator = MacroIndicator(
                indicator_name=name,
                date=datetime.combine(date_val, datetime.min.time()),
                value=value,
                unit=unit
            )
            self.session.add(indicator)
        
        self.session.commit()
    
    def calculate_technical_indicators(self):
        """Calculate and store technical indicators (filtered by index if specified)"""
        logger.info("📊 Calculating Technical Indicators...")
        
        # Get tickers, filtered by index if specified
        if self.index_id:
            watchlist = self.session.scalars(
                select(Watchlist).filter_by(is_active=True, index_id=self.index_id)
            ).all()
            if not watchlist:
                # Fallback: get companies from index
                index = self.session.scalar(select(Index).filter_by(id=self.index_id))
                if index:
                    tickers = [c.ticker for c in index.companies]
                else:
                    tickers = []
            else:
                tickers = [w.ticker for w in watchlist]
        else:
            watchlist = self.session.scalars(
                select(Watchlist).filter_by(is_active=True)
            ).all()
            
            if not watchlist:
                companies = self.session.scalars(select(CompanyProfile)).all()
                tickers = [c.ticker for c in companies]
            else:
                tickers = [w.ticker for w in watchlist]
        
        for ticker in tickers:
            try:
                # Get market data
                stmt = select(MarketData).filter_by(ticker=ticker).order_by(MarketData.date)
                market_data = self.session.scalars(stmt).all()
                
                if len(market_data) < 200:  # Need enough data for indicators
                    continue
                
                # Remove duplicates by date (keep the latest entry for each date)
                seen_dates = {}
                for md in market_data:
                    date_key = md.date if isinstance(md.date, date) else md.date.date()
                    if date_key not in seen_dates or md.date > seen_dates[date_key].date:
                        seen_dates[date_key] = md
                
                market_data = list(seen_dates.values())
                # Sort by date to maintain order
                market_data.sort(key=lambda x: x.date)
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'date': md.date,
                    'open': md.open,
                    'high': md.high,
                    'low': md.low,
                    'close': md.close,
                    'volume': md.volume
                } for md in market_data])
                
                # Remove duplicate dates (keep the last entry for each date)
                if df['date'].duplicated().any():
                    logger.debug(f"{ticker}: Found duplicate dates, keeping last entry for each date")
                    df = df.drop_duplicates(subset='date', keep='last')
                
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Ensure index is unique (final check)
                if df.index.duplicated().any():
                    logger.warning(f"{ticker}: Duplicate indices after processing, removing duplicates")
                    df = df[~df.index.duplicated(keep='last')]
                
                # Calculate indicators using pandas_ta
                df['RSI'] = ta.rsi(df['close'], length=14)
                macd_data = ta.macd(df['close'])
                if macd_data is not None and isinstance(macd_data, pd.DataFrame):
                    df['MACD'] = macd_data['MACD_12_26_9']
                    df['MACD_Signal'] = macd_data['MACDs_12_26_9']
                else:
                    df['MACD'] = None
                    df['MACD_Signal'] = None
                
                df['SMA_50'] = ta.sma(df['close'], length=50)
                df['SMA_200'] = ta.sma(df['close'], length=200)
                df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                
                # Save latest indicators
                latest = df.iloc[-1]
                latest_date = df.index[-1]
                
                # Check if indicator exists for this date
                stmt = select(TechnicalIndicators).filter_by(
                    ticker=ticker,
                    date=latest_date
                )
                existing = self.session.scalars(stmt).first()
                
                if existing:
                    existing.rsi_14 = float(latest['RSI']) if pd.notna(latest['RSI']) else None
                    existing.macd = float(latest['MACD']) if pd.notna(latest['MACD']) else None
                    existing.macd_signal = float(latest['MACD_Signal']) if pd.notna(latest['MACD_Signal']) else None
                    existing.sma_50 = float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else None
                    existing.sma_200 = float(latest['SMA_200']) if pd.notna(latest['SMA_200']) else None
                    existing.atr = float(latest['ATR']) if pd.notna(latest['ATR']) else None
                else:
                    indicator = TechnicalIndicators(
                        ticker=ticker,
                        date=latest_date,
                        rsi_14=float(latest['RSI']) if pd.notna(latest['RSI']) else None,
                        macd=float(latest['MACD']) if pd.notna(latest['MACD']) else None,
                        macd_signal=float(latest['MACD_Signal']) if pd.notna(latest['MACD_Signal']) else None,
                        sma_50=float(latest['SMA_50']) if pd.notna(latest['SMA_50']) else None,
                        sma_200=float(latest['SMA_200']) if pd.notna(latest['SMA_200']) else None,
                        atr=float(latest['ATR']) if pd.notna(latest['ATR']) else None
                    )
                    self.session.add(indicator)
                
                self.session.commit()
                logger.info(f"✅ {ticker}: Indicators calculated")
                
            except Exception as e:
                logger.error(f"❌ Failed to calculate indicators for {ticker}: {e}")
                self.session.rollback()
    
    def get_zone_statistics(self) -> dict:
        """
        Get statistics for each data zone
        
        Returns:
            dict with statistics for each zone
        """
        stats = {}
        
        # Market Data Zone
        if self.index_id:
            index = self.session.scalar(select(Index).filter_by(id=self.index_id))
            if index:
                tickers = [c.ticker for c in index.companies]
                market_data_count = self.session.scalar(
                    select(func.count(MarketData.id))
                    .where(MarketData.ticker.in_(tickers))
                ) or 0
                latest_market_date = self.session.scalar(
                    select(func.max(MarketData.date))
                    .where(MarketData.ticker.in_(tickers))
                )
            else:
                market_data_count = 0
                latest_market_date = None
        else:
            market_data_count = self.session.scalar(select(func.count(MarketData.id))) or 0
            latest_market_date = self.session.scalar(select(func.max(MarketData.date)))
        
        stats['market_data'] = {
            'count': market_data_count,
            'latest_date': latest_market_date
        }
        
        # Technical Indicators Zone
        if self.index_id:
            index = self.session.scalar(select(Index).filter_by(id=self.index_id))
            if index:
                tickers = [c.ticker for c in index.companies]
                indicators_count = self.session.scalar(
                    select(func.count(TechnicalIndicators.id))
                    .where(TechnicalIndicators.ticker.in_(tickers))
                ) or 0
            else:
                indicators_count = 0
        else:
            indicators_count = self.session.scalar(select(func.count(TechnicalIndicators.id))) or 0
        
        stats['technical_indicators'] = {
            'count': indicators_count
        }
        
        # Macro Indicators Zone
        macro_count = self.session.scalar(select(func.count(MacroIndicator.id))) or 0
        latest_macro_date = self.session.scalar(select(func.max(MacroIndicator.date)))
        stats['macro_indicators'] = {
            'count': macro_count,
            'latest_date': latest_macro_date
        }
        
        # Feature Store Zone (if available)
        try:
            from database.models import FeatureStore
            if self.index_id:
                index = self.session.scalar(select(Index).filter_by(id=self.index_id))
                if index:
                    tickers = [c.ticker for c in index.companies]
                    feature_count = self.session.scalar(
                        select(func.count(FeatureStore.id))
                        .where(FeatureStore.ticker.in_(tickers))
                    ) or 0
                else:
                    feature_count = 0
            else:
                feature_count = self.session.scalar(select(func.count(FeatureStore.id))) or 0
            
            stats['feature_store'] = {
                'count': feature_count
            }
        except ImportError:
            stats['feature_store'] = {'count': 0}
        
        return stats

if __name__ == "__main__":
    etl = ETLModule()
    etl.run_full_sync()

