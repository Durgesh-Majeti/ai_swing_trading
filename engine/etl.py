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
from datetime import datetime, date
from typing import Optional
import pandas_ta as ta

class ETLModule:
    """Main ETL orchestrator"""
    
    def __init__(self, index_id: Optional[int] = None):
        self.session = SessionLocal()
        self.index_id = index_id
    
    def run_full_sync(self):
        """Run complete data sync"""
        index_name = None
        if self.index_id:
            index = self.session.scalar(select(Index).filter_by(id=self.index_id))
            index_name = index.display_name if index else None
        
        logger.info(f"🚀 Starting Full ETL Pipeline{' for ' + index_name if index_name else ''}...")
        
        try:
            # 1. Sync Market Data
            self.sync_market_data()
            
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
    
    def sync_market_data(self):
        """Fetch daily OHLCV data for watchlist stocks (filtered by index if specified)"""
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
        
        for ticker in tickers:
            try:
                # Fetch 1 year of data
                df = yf.download(ticker, period="1y", interval="1d", progress=False)
                
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue
                
                # Check latest date in DB
                stmt = select(MarketData).filter_by(ticker=ticker).order_by(MarketData.date.desc())
                last_entry = self.session.scalars(stmt).first()
                
                new_records = []
                for date_idx, row in df.iterrows():
                    # Convert to datetime if needed
                    if isinstance(date_idx, pd.Timestamp):
                        date_val = date_idx.to_pydatetime()
                    else:
                        date_val = date_idx
                    
                    # Skip if we already have this date
                    if last_entry and date_val <= last_entry.date:
                        continue
                    
                    record = MarketData(
                        ticker=ticker,
                        date=date_val,
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=int(row['Volume']) if pd.notna(row['Volume']) else 0
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
        """Calculate and store technical indicators"""
        logger.info("📊 Calculating Technical Indicators...")
        
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
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'date': md.date,
                    'open': md.open,
                    'high': md.high,
                    'low': md.low,
                    'close': md.close,
                    'volume': md.volume
                } for md in market_data])
                
                df.set_index('date', inplace=True)
                
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

if __name__ == "__main__":
    etl = ETLModule()
    etl.run_full_sync()

