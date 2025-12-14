"""
Enhanced Feature Engineering - Advanced features for signal generation models
Includes lag features, rolling statistics, cross-asset features, and pattern detection
"""

import pandas as pd
import numpy as np
from database.models import (
    SessionLocal, MarketData, TechnicalIndicators, 
    FundamentalData, MacroIndicator, FeatureStore, CompanyProfile, Index
)
from sqlalchemy import select, func
from loguru import logger
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List
import pandas_ta as ta


class EnhancedFeatureEngine:
    """Generates advanced features for signal generation models"""
    
    def __init__(self, index_id: Optional[int] = None):
        self.session = SessionLocal()
        self.index_id = index_id
    
    def get_sequence_features(self, ticker: str, sequence_length: int = 30) -> Optional[pd.DataFrame]:
        """
        Get sequence of features for time-series models
        
        Returns:
            DataFrame with shape (sequence_length, n_features) or None
        """
        try:
            # Get market data - need enough for indicators (SMA_200 needs 200 days)
            # We need at least 200 days for SMA_200, plus buffer for other indicators
            cutoff_date = datetime.now() - timedelta(days=250)  # Get more data for indicators
            stmt = select(MarketData).filter_by(ticker=ticker).filter(
                MarketData.date >= cutoff_date
            ).order_by(MarketData.date)
            
            market_data = self.session.scalars(stmt).all()
            
            # Remove duplicates by date (keep the latest entry for each date)
            # This handles cases where the same date might be inserted multiple times
            seen_dates = {}
            for md in market_data:
                date_key = md.date if isinstance(md.date, date) else md.date.date()
                if date_key not in seen_dates:
                    seen_dates[date_key] = md
                else:
                    # Compare dates properly - keep the later one
                    existing_md = seen_dates[date_key]
                    existing_date = existing_md.date if isinstance(existing_md.date, date) else existing_md.date.date()
                    current_date = md.date if isinstance(md.date, date) else md.date.date()
                    if current_date >= existing_date:
                        seen_dates[date_key] = md
            
            market_data = list(seen_dates.values())
            # Sort by date to maintain order
            market_data.sort(key=lambda x: x.date)
            
            if len(market_data) < 100:  # Need at least 100 days
                logger.debug(f"{ticker}: Insufficient market data: {len(market_data)} days (need at least 100)")
                return None
            
            logger.debug(f"{ticker}: Found {len(market_data)} days of market data")
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': md.date,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            } for md in market_data])
            
            # Handle duplicate dates - keep the last entry for each date
            if df['date'].duplicated().any():
                logger.debug(f"{ticker}: Found duplicate dates, keeping last entry for each date")
                df = df.drop_duplicates(subset='date', keep='last')
            
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Ensure index is unique (final check)
            if df.index.duplicated().any():
                logger.warning(f"{ticker}: Duplicate indices after processing, removing duplicates")
                df = df[~df.index.duplicated(keep='last')]
            
            # Calculate all features
            feature_df = self._calculate_all_features(df, ticker)
            
            if feature_df is None or len(feature_df) == 0:
                logger.debug(f"{ticker}: Feature calculation returned None or empty DataFrame")
                return None
            
            logger.debug(f"{ticker}: Calculated features, shape: {feature_df.shape}")
            
            # Drop rows with all NaN (but keep rows with some NaN that were filled)
            # The _calculate_all_features already fills NaN, so this should be minimal
            feature_df = feature_df.dropna(how='all')
            
            if len(feature_df) < sequence_length:
                logger.debug(f"{ticker}: Insufficient feature rows: {len(feature_df)} < {sequence_length}")
                return None
            
            # Get last sequence_length rows
            result = feature_df.tail(sequence_length)
            logger.debug(f"{ticker}: Returning feature sequence with shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting sequence features for {ticker}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def get_latest_features(self, ticker: str) -> Optional[pd.Series]:
        """
        Get latest feature vector for point-in-time prediction
        
        Returns:
            Series with feature values or None
        """
        sequence = self.get_sequence_features(ticker, sequence_length=1)
        if sequence is None or len(sequence) == 0:
            return None
        
        return sequence.iloc[-1]
    
    def _calculate_all_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Calculate all features from raw market data"""
        
        # ========== Basic Price Features ==========
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['return'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # ========== Technical Indicators ==========
        # Calculate indicators that don't require too much history first
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        macd_data = ta.macd(df['close'])
        if macd_data is not None and isinstance(macd_data, pd.DataFrame):
            df['macd'] = macd_data['MACD_12_26_9']
            df['macd_signal'] = macd_data['MACDs_12_26_9']
            df['macd_hist'] = macd_data['MACDh_12_26_9']
        else:
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_hist'] = 0
        
        # Calculate SMAs - use available data, fill with forward fill if needed
        df['sma_5'] = ta.sma(df['close'], length=5)
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        
        # SMA_200 needs 200 days - calculate only if we have enough data
        if len(df) >= 200:
            df['sma_200'] = ta.sma(df['close'], length=200)
        else:
            # Use SMA_50 as fallback if we don't have 200 days
            df['sma_200'] = df['sma_50']
        
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        bb_data = ta.bbands(df['close'], length=20)
        if bb_data is not None and isinstance(bb_data, pd.DataFrame):
            df['bb_upper'] = bb_data['BBU_20_2.0'] if 'BBU_20_2.0' in bb_data.columns else bb_data.iloc[:, 0]
            df['bb_middle'] = bb_data['BBM_20_2.0'] if 'BBM_20_2.0' in bb_data.columns else bb_data.iloc[:, 1]
            df['bb_lower'] = bb_data['BBL_20_2.0'] if 'BBL_20_2.0' in bb_data.columns else bb_data.iloc[:, 2]
        else:
            df['bb_upper'] = 0
            df['bb_middle'] = 0
            df['bb_lower'] = 0
        
        # ========== Lag Features ==========
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'return_lag_{lag}'] = df['return'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # ========== Rolling Window Features ==========
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'rolling_max_{window}'] = df['close'].rolling(window=window).max()
            df[f'rolling_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'rolling_volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
        
        # ========== Volatility Features ==========
        df['volatility_5'] = df['return'].rolling(window=5).std()
        df['volatility_10'] = df['return'].rolling(window=10).std()
        df['volatility_20'] = df['return'].rolling(window=20).std()
        
        # ========== Momentum Features ==========
        df['price_momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['price_momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        df['price_momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        
        df['volume_momentum'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['volume_trend'] = df['volume'].rolling(window=10).mean() / df['volume'].rolling(window=30).mean()
        
        # ========== Pattern Features ==========
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(int)
        
        # Support and Resistance (simplified)
        df['support_level'] = df['low'].rolling(window=20).min()
        df['resistance_level'] = df['high'].rolling(window=20).max()
        df['distance_to_support'] = (df['close'] - df['support_level']) / df['close']
        df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
        
        # Breakout patterns
        df['breakout_above_resistance'] = (df['close'] > df['resistance_level'].shift(1)).astype(int)
        df['breakdown_below_support'] = (df['close'] < df['support_level'].shift(1)).astype(int)
        
        # ========== Relative Strength ==========
        # Price relative to moving averages
        df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['price_vs_sma_200'] = (df['close'] - df['sma_200']) / df['sma_200']
        
        # ========== Index-Relative Features ==========
        index_features = self._get_index_relative_features(ticker, df)
        if index_features is not None and not index_features.empty:
            # Ensure indices are unique before concatenating
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]
            if index_features.index.duplicated().any():
                index_features = index_features[~index_features.index.duplicated(keep='last')]
            df = pd.concat([df, index_features], axis=1)
        
        # ========== Fundamental Features ==========
        fundamental_features = self._get_fundamental_features(ticker)
        if fundamental_features:
            # Broadcast fundamental features to all dates (they're static)
            for key, value in fundamental_features.items():
                df[key] = value
        
        # ========== Macro Features ==========
        macro_features = self._get_macro_features(df.index)
        if macro_features is not None and not macro_features.empty:
            # Ensure indices are unique before concatenating
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]
            if macro_features.index.duplicated().any():
                macro_features = macro_features[~macro_features.index.duplicated(keep='last')]
            df = pd.concat([df, macro_features], axis=1)
        
        # ========== Clean up ==========
        # Ensure index is unique before any operations
        if df.index.duplicated().any():
            logger.debug(f"{ticker}: Removing duplicate indices in feature calculation")
            df = df[~df.index.duplicated(keep='last')]
        
        # Fill NaN values - use forward fill first, then backward fill, then 0
        df = df.ffill().bfill().fillna(0)
        
        # Replace inf values
        df = df.replace([np.inf, -np.inf], 0)
        
        # Ensure we have numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Final check - ensure index is unique
        if df.index.duplicated().any():
            logger.warning(f"{ticker}: Duplicate indices still present after cleanup, removing")
            df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def _get_index_relative_features(self, ticker: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get features relative to index performance"""
        try:
            # Get index for this ticker
            company = self.session.scalar(
                select(CompanyProfile).filter_by(ticker=ticker)
            )
            if not company or not company.indices:
                return None
            
            # Use first index
            index = company.indices[0]
            
            # Get index tickers (simplified: use first few as proxy)
            # In production, you'd calculate actual index value
            # For now, we'll skip this or use a simplified approach
            return None  # Placeholder - implement index calculation if needed
            
        except Exception as e:
            logger.debug(f"Could not get index features for {ticker}: {e}")
            return None
    
    def _get_fundamental_features(self, ticker: str) -> Optional[Dict[str, float]]:
        """Get fundamental features (static for now)"""
        try:
            stmt = select(FundamentalData).filter_by(ticker=ticker).order_by(
                FundamentalData.report_date.desc()
            )
            latest = self.session.scalars(stmt).first()
            
            if not latest:
                return {
                    'pe_ratio': 0,
                    'pb_ratio': 0,
                    'roe': 0,
                    'debt_to_equity': 0,
                    'revenue_growth': 0,
                    'eps': 0
                }
            
            return {
                'pe_ratio': float(latest.pe_ratio) if latest.pe_ratio else 0,
                'pb_ratio': float(latest.pb_ratio) if latest.pb_ratio else 0,
                'roe': float(latest.roe) if latest.roe else 0,
                'debt_to_equity': float(latest.debt_to_equity) if latest.debt_to_equity else 0,
                'revenue_growth': float(latest.revenue_growth) if latest.revenue_growth else 0,
                'eps': float(latest.eps) if latest.eps else 0
            }
        except Exception as e:
            logger.debug(f"Could not get fundamental features for {ticker}: {e}")
            return {
                'pe_ratio': 0,
                'pb_ratio': 0,
                'roe': 0,
                'debt_to_equity': 0,
                'revenue_growth': 0,
                'eps': 0
            }
    
    def _get_macro_features(self, dates: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
        """Get macro features aligned with dates"""
        try:
            # Ensure dates are unique
            unique_dates = dates.drop_duplicates() if dates.duplicated().any() else dates
            macro_df = pd.DataFrame(index=unique_dates)
            
            # Get latest macro values (simplified - in production, align by date)
            today = datetime.now().date()
            vix = self._get_latest_macro("INDIA_VIX", today)
            crude_oil = self._get_latest_macro("CRUDE_OIL", today)
            usd_inr = self._get_latest_macro("USD_INR", today)
            
            macro_df['vix'] = vix if vix else 0
            macro_df['crude_oil'] = crude_oil if crude_oil else 0
            macro_df['usd_inr'] = usd_inr if usd_inr else 0
            
            # Ensure index is unique
            if macro_df.index.duplicated().any():
                macro_df = macro_df[~macro_df.index.duplicated(keep='last')]
            
            return macro_df
            
        except Exception as e:
            logger.debug(f"Could not get macro features: {e}")
            return None
    
    def _get_latest_macro(self, indicator_name: str, date: datetime.date) -> Optional[float]:
        """Get latest macro indicator value"""
        stmt = select(MacroIndicator).filter_by(
            indicator_name=indicator_name
        ).filter(
            MacroIndicator.date <= date
        ).order_by(MacroIndicator.date.desc())
        latest = self.session.scalars(stmt).first()
        return latest.value if latest else None
    
    def close(self):
        """Close database session"""
        self.session.close()

