"""
Model Training Script - Train and register ML models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from database.models import SessionLocal, FeatureStore, MarketData
from sqlalchemy import select
from loguru import logger
from datetime import datetime, timedelta
from ai.model_registry import ModelRegistryManager
import pickle

class ModelTrainer:
    """Trains ML models for price prediction"""
    
    def __init__(self):
        self.session = SessionLocal()
        self.registry = ModelRegistryManager()
    
    def prepare_training_data(self, lookback_days: int = 365):
        """Prepare training data from feature store"""
        logger.info("📊 Preparing training data...")
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Get all features
        stmt = select(FeatureStore).filter(FeatureStore.date >= cutoff_date).order_by(FeatureStore.date)
        features = self.session.scalars(stmt).all()
        
        if len(features) < 100:
            logger.warning("Insufficient data for training")
            return None, None
        
        # Convert to DataFrame
        feature_data = []
        target_data = []
        
        for feature in features:
            # Get future price (target)
            future_date = feature.date + timedelta(days=30)  # 30-day prediction
            stmt = select(MarketData).filter_by(
                ticker=feature.ticker
            ).filter(MarketData.date <= future_date).order_by(MarketData.date.desc())
            
            future_price = self.session.scalars(stmt).first()
            
            if not future_price:
                continue
            
            # Feature vector
            feature_vector = [
                feature.log_return or 0,
                feature.rsi or 50,
                feature.macd or 0,
                feature.sma_50 or 0,
                feature.sma_200 or 0,
                feature.atr or 0,
                feature.volatility or 0,
                feature.pe_ratio or 0,
                feature.roe or 0,
                feature.debt_to_equity or 0,
                feature.vix or 0,
                feature.crude_oil or 0,
                feature.usd_inr or 0,
                feature.price_momentum or 0,
                feature.volume_trend or 1
            ]
            
            feature_data.append(feature_vector)
            
            # Target: future price
            target_data.append(future_price.close)
        
        X = np.array(feature_data)
        y = np.array(target_data)
        
        logger.success(f"✅ Prepared {len(X)} samples")
        return X, y
    
    def train_random_forest(self, model_name: str = "RandomForest_Swing_v1"):
        """Train a Random Forest model"""
        logger.info(f"🌳 Training {model_name}...")
        
        X, y = self.prepare_training_data()
        
        if X is None:
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        metrics = {
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
            "n_samples": len(X),
            "n_features": X.shape[1]
        }
        
        logger.success(f"✅ Model trained:")
        logger.info(f"   Train MAE: ₹{train_mae:.2f}")
        logger.info(f"   Test MAE: ₹{test_mae:.2f}")
        logger.info(f"   Test R²: {test_r2:.3f}")
        
        # Register model
        registry_entry = self.registry.register_model(
            model_name=model_name,
            model_type="RandomForest",
            model_object=model,
            performance_metrics=metrics,
            description="Random Forest for 30-day price prediction"
        )
        
        if registry_entry:
            logger.success(f"✅ Model {model_name} registered")
            return model
        
        return None
    
    def activate_model(self, model_name: str):
        """Activate a trained model"""
        success = self.registry.activate_model(model_name)
        if success:
            logger.success(f"✅ Model {model_name} activated")
        return success

if __name__ == "__main__":
    import sys
    
    trainer = ModelTrainer()
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "RandomForest_Swing_v1"
    
    model = trainer.train_random_forest(model_name)
    
    if model:
        # Optionally activate
        activate = input(f"Activate model {model_name}? (y/n): ")
        if activate.lower() == 'y':
            trainer.activate_model(model_name)

