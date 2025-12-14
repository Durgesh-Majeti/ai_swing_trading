"""
Feature Store - Transforms raw data into ML-ready features
"""

import pandas as pd
import numpy as np
from database.models import (
    SessionLocal, MarketData, TechnicalIndicators, 
    FundamentalData, MacroIndicator, FeatureStore, CompanyProfile, Watchlist, Index
)
from sqlalchemy import select, func
from loguru import logger
from datetime import datetime, timedelta, date
from typing import Optional
import pandas_ta as ta

class FeatureStoreEngine:
    """Transforms raw market data into ML features"""
    
    def __init__(self, index_id: Optional[int] = None):
        self.session = SessionLocal()
        self.index_id = index_id
    
    def generate_features(self, ticker: str, lookback_days: int = 60):
        """Generate features for a specific ticker"""
        try:
            # Get market data
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            stmt = select(MarketData).filter_by(ticker=ticker).filter(
                MarketData.date >= cutoff_date
            ).order_by(MarketData.date)
            
            market_data = self.session.scalars(stmt).all()
            
            if len(market_data) < 30:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
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
            
            # Calculate technical features
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd_data = ta.macd(df['close'])
            if macd_data is not None and isinstance(macd_data, pd.DataFrame):
                df['macd'] = macd_data['MACD_12_26_9']
            else:
                df['macd'] = 0
            
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Volatility (rolling std of returns)
            df['volatility'] = df['log_return'].rolling(window=20).std()
            
            # Price momentum
            df['price_momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
            
            # Volume trend
            df['volume_trend'] = df['volume'].rolling(window=10).mean() / df['volume'].rolling(window=30).mean()
            
            # Get latest fundamental data
            stmt = select(FundamentalData).filter_by(ticker=ticker).order_by(
                FundamentalData.report_date.desc()
            )
            latest_fundamental = self.session.scalars(stmt).first()
            
            pe_ratio = latest_fundamental.pe_ratio if latest_fundamental else None
            roe = latest_fundamental.roe if latest_fundamental else None
            debt_to_equity = latest_fundamental.debt_to_equity if latest_fundamental else None
            
            # Get latest macro indicators
            today = datetime.now().date()
            vix = self._get_latest_macro("INDIA_VIX", today)
            crude_oil = self._get_latest_macro("CRUDE_OIL", today)
            usd_inr = self._get_latest_macro("USD_INR", today)
            
            # Get latest row (most recent features)
            latest = df.iloc[-1]
            latest_date = df.index[-1]
            
            # Create feature record
            feature_record = FeatureStore(
                ticker=ticker,
                date=latest_date,
                log_return=float(latest['log_return']) if pd.notna(latest['log_return']) else None,
                rsi=float(latest['rsi']) if pd.notna(latest['rsi']) else None,
                macd=float(latest['macd']) if pd.notna(latest['macd']) else None,
                sma_50=float(latest['sma_50']) if pd.notna(latest['sma_50']) else None,
                sma_200=float(latest['sma_200']) if pd.notna(latest['sma_200']) else None,
                atr=float(latest['atr']) if pd.notna(latest['atr']) else None,
                volatility=float(latest['volatility']) if pd.notna(latest['volatility']) else None,
                pe_ratio=pe_ratio,
                roe=roe,
                debt_to_equity=debt_to_equity,
                vix=vix,
                crude_oil=crude_oil,
                usd_inr=usd_inr,
                price_momentum=float(latest['price_momentum']) if pd.notna(latest['price_momentum']) else None,
                volume_trend=float(latest['volume_trend']) if pd.notna(latest['volume_trend']) else None
            )
            
            # Check if feature exists for this date
            stmt = select(FeatureStore).filter_by(ticker=ticker, date=latest_date)
            existing = self.session.scalars(stmt).first()
            
            if existing:
                # Update existing
                for key, value in feature_record.__dict__.items():
                    if key != '_sa_instance_state' and key != 'id':
                        setattr(existing, key, value)
            else:
                self.session.add(feature_record)
            
            self.session.commit()
            logger.success(f"✅ Features generated for {ticker}")
            
            return feature_record
            
        except Exception as e:
            logger.error(f"❌ Failed to generate features for {ticker}: {e}")
            self.session.rollback()
            return None
    
    def _get_latest_macro(self, indicator_name: str, date_val):
        """Get latest macro indicator value"""
        stmt = select(MacroIndicator).filter_by(indicator_name=indicator_name).order_by(
            MacroIndicator.date.desc()
        )
        latest = self.session.scalars(stmt).first()
        return latest.value if latest else None
    
    def get_feature_vector(self, ticker: str) -> pd.Series:
        """Get latest feature vector as pandas Series for model inference"""
        stmt = select(FeatureStore).filter_by(ticker=ticker).order_by(
            FeatureStore.date.desc()
        )
        latest = self.session.scalars(stmt).first()
        
        if not latest:
            return None
        
        # Convert to feature vector
        features = {
            'log_return': latest.log_return or 0,
            'rsi': latest.rsi or 50,
            'macd': latest.macd or 0,
            'sma_50': latest.sma_50 or 0,
            'sma_200': latest.sma_200 or 0,
            'atr': latest.atr or 0,
            'volatility': latest.volatility or 0,
            'pe_ratio': latest.pe_ratio or 0,
            'roe': latest.roe or 0,
            'debt_to_equity': latest.debt_to_equity or 0,
            'vix': latest.vix or 0,
            'crude_oil': latest.crude_oil or 0,
            'usd_inr': latest.usd_inr or 0,
            'price_momentum': latest.price_momentum or 0,
            'volume_trend': latest.volume_trend or 1
        }
        
        return pd.Series(features)
    
    def generate_all_features(self):
        """Generate features for watchlist stocks (filtered by index if specified)"""
        index_name = None
        if self.index_id:
            index = self.session.scalar(select(Index).filter_by(id=self.index_id))
            index_name = index.display_name if index else None
        
        logger.info(f"🔧 Generating features{' for ' + index_name if index_name else ' for all stocks'}...")
        
        # Get watchlist companies, filtered by index if specified
        if self.index_id:
            watchlist = self.session.scalars(
                select(Watchlist).filter_by(is_active=True, index_id=self.index_id)
            ).all()
            if not watchlist:
                # Fallback: get companies from index
                index = self.session.scalar(select(Index).filter_by(id=self.index_id))
                if index:
                    companies = index.companies
                else:
                    companies = []
            else:
                tickers = [w.ticker for w in watchlist]
                companies = self.session.scalars(
                    select(CompanyProfile).filter(CompanyProfile.ticker.in_(tickers))
                ).all()
        else:
            # Get all active watchlist
            watchlist_tickers = self.session.scalars(
                select(Watchlist.ticker).filter_by(is_active=True)
            ).all()
            
            if watchlist_tickers:
                companies = self.session.scalars(
                    select(CompanyProfile).filter(CompanyProfile.ticker.in_(watchlist_tickers))
                ).all()
            else:
                companies = self.session.scalars(select(CompanyProfile)).all()
        
        for company in companies:
            self.generate_features(company.ticker)
        
        logger.success("✅ Feature generation complete")
    
    def close(self):
        """Close database session"""
        if hasattr(self, 'session') and self.session:
            self.session.close()
            self.session = None

if __name__ == "__main__":
    engine = FeatureStoreEngine()
    engine.generate_all_features()
    engine.close()

