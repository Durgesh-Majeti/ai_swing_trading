"""
Inference Engine - Daily prediction generator
"""

import pandas as pd
import numpy as np
from database.models import (
    SessionLocal, AIPredictions, CompanyProfile, Watchlist, FeatureStore
)
from sqlalchemy import select
from loguru import logger
from datetime import datetime, date, timedelta
from ai.model_registry import ModelRegistryManager
from ai.feature_store import FeatureStoreEngine

class InferenceEngine:
    """Generates daily predictions using active model"""
    
    def __init__(self):
        self.session = SessionLocal()
        self.model_registry = ModelRegistryManager()
        self.feature_store = FeatureStoreEngine()
    
    def run_daily_inference(self):
        """Run inference for all watchlist stocks"""
        logger.info("🧠 Starting Daily Inference...")
        
        # Check if we have an active model
        model_obj, model_reg = self.model_registry.load_active_model()
        
        if not model_obj:
            logger.warning("⚠️  No active model found. Skipping inference.")
            return
        
        logger.info(f"Using model: {model_reg.model_name}")
        
        # Get active watchlist
        watchlist = self.session.scalars(
            select(Watchlist).filter_by(is_active=True)
        ).all()
        
        if not watchlist:
            # Fallback to all companies
            companies = self.session.scalars(select(CompanyProfile)).all()
            tickers = [c.ticker for c in companies]
        else:
            tickers = [w.ticker for w in watchlist]
        
        predictions_made = 0
        
        for ticker in tickers:
            try:
                # Ensure features are up to date
                self.feature_store.generate_features(ticker)
                
                # Get feature vector
                feature_vector = self.feature_store.get_feature_vector(ticker)
                
                if feature_vector is None:
                    logger.warning(f"No features available for {ticker}")
                    continue
                
                # Get current price for context
                stmt = select(FeatureStore).filter_by(ticker=ticker).order_by(
                    FeatureStore.date.desc()
                )
                latest_feature = self.session.scalars(stmt).first()
                
                if not latest_feature:
                    continue
                
                # Prepare features for model (convert to numpy array)
                X = feature_vector.values.reshape(1, -1)
                
                # Make prediction
                # Note: This assumes the model has a predict method
                # Adjust based on your actual model interface
                if hasattr(model_obj, 'predict'):
                    predicted_price = model_obj.predict(X)[0]
                elif hasattr(model_obj, 'predict_proba'):
                    # For classification models
                    proba = model_obj.predict_proba(X)[0]
                    # Convert to price prediction (simplified)
                    current_price = self._get_current_price(ticker)
                    if current_price:
                        # Simple heuristic: use probability to adjust price
                        predicted_price = current_price * (1 + (proba[1] - proba[0]) * 0.1)
                    else:
                        continue
                else:
                    logger.warning(f"Model {model_reg.model_name} has no predict method")
                    continue
                
                # Calculate confidence (simplified - can be enhanced)
                confidence = self._calculate_confidence(feature_vector, model_obj)
                
                # Determine direction
                current_price = self._get_current_price(ticker)
                if current_price:
                    direction = "UP" if predicted_price > current_price else "DOWN"
                else:
                    direction = "NEUTRAL"
                
                # Target date (e.g., 30 days from now for swing trading)
                target_date = date.today() + timedelta(days=30)
                
                # Check if prediction already exists for today
                today = date.today()
                stmt = select(AIPredictions).filter_by(
                    ticker=ticker,
                    model_name=model_reg.model_name,
                    target_date=today
                )
                existing = self.session.scalars(stmt).first()
                
                if existing:
                    # Update existing prediction
                    existing.predicted_price = float(predicted_price)
                    existing.confidence_score = float(confidence)
                    existing.direction = direction
                    existing.generated_at = datetime.now()
                else:
                    # Create new prediction
                    prediction = AIPredictions(
                        ticker=ticker,
                        model_name=model_reg.model_name,
                        target_date=target_date,
                        predicted_price=float(predicted_price),
                        confidence_score=float(confidence),
                        direction=direction
                    )
                    self.session.add(prediction)
                
                predictions_made += 1
                logger.info(f"✅ {ticker}: {direction} @ {predicted_price:.2f} (Confidence: {confidence:.2%})")
                
            except Exception as e:
                logger.error(f"❌ Failed to predict for {ticker}: {e}")
                continue
        
        self.session.commit()
        logger.success(f"✅ Inference complete: {predictions_made} predictions generated")
        self.session.close()
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price from latest market data"""
        from database.models import MarketData
        stmt = select(MarketData).filter_by(ticker=ticker).order_by(
            MarketData.date.desc()
        )
        latest = self.session.scalars(stmt).first()
        return latest.close if latest else None
    
    def _calculate_confidence(self, feature_vector: pd.Series, model) -> float:
        """Calculate prediction confidence (simplified)"""
        # Simple heuristic: use feature quality as proxy for confidence
        # Can be enhanced with model-specific confidence measures
        
        # Check for missing or extreme values
        if feature_vector.isna().any():
            return 0.3
        
        # Normalize confidence based on feature stability
        rsi = feature_vector.get('rsi', 50)
        volatility = abs(feature_vector.get('volatility', 0))
        
        # Lower volatility and moderate RSI = higher confidence
        confidence = 0.5  # Base confidence
        
        if 30 < rsi < 70:  # Not extreme
            confidence += 0.2
        
        if volatility < 0.05:  # Low volatility
            confidence += 0.2
        
        return min(confidence, 0.95)  # Cap at 95%

if __name__ == "__main__":
    engine = InferenceEngine()
    engine.run_daily_inference()

