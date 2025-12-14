"""
Signal Generation Models - Train models for signal generation and profit potential
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from database.models import SessionLocal, MarketData, TradeSignal, Portfolio, Order, CompanyProfile
from sqlalchemy import select, and_
from loguru import logger
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
from ai.enhanced_features import EnhancedFeatureEngine
from ai.model_registry import ModelRegistryManager
import pickle
from pathlib import Path


class SignalModelTrainer:
    """Trains models for signal generation and profit potential"""
    
    def __init__(self, index_id: Optional[int] = None):
        self.session = SessionLocal()
        self.registry = ModelRegistryManager()
        self.feature_engine = EnhancedFeatureEngine(index_id=index_id)
        self.index_id = index_id
        self.scaler = StandardScaler()
    
    def prepare_training_data(self, lookback_days: int = 730, prediction_horizon: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Prepare training data with features and targets
        
        Returns:
            X (features), y_signal (BUY/SELL/HOLD), y_return (expected return %), y_target_price (target price)
        """
        logger.info("📊 Preparing training data for signal models...")
        
        # Get all companies (filtered by index if specified)
        if self.index_id:
            from database.models import Index
            index = self.session.scalar(select(Index).filter_by(id=self.index_id))
            if index:
                companies = index.companies
            else:
                companies = []
        else:
            companies = self.session.scalars(select(CompanyProfile)).all()
        
        if not companies:
            logger.warning("No companies found for training")
            return None, None, None, None
        
        feature_data = []
        signal_targets = []
        return_targets = []
        target_price_targets = []
        
        total_companies = len(companies)
        processed = 0
        
        for company in companies:
            ticker = company.ticker
            processed += 1
            
            if processed % 10 == 0:
                logger.info(f"Processing {processed}/{total_companies} companies...")
            
            try:
                # Get historical market data (need enough for features + future prediction)
                # We need: sequence_length (30) + prediction_horizon (30) + buffer for indicators (50) = ~110 days minimum
                # But for training, we want to go back lookback_days
                cutoff_date = datetime.now() - timedelta(days=lookback_days + prediction_horizon + 50)
                stmt = select(MarketData).filter_by(ticker=ticker).filter(
                    MarketData.date >= cutoff_date
                ).order_by(MarketData.date)
                
                market_data = self.session.scalars(stmt).all()
                
                if len(market_data) < 100:  # Need at least 100 days of data
                    continue
                
                # Remove duplicates by date (keep the latest entry for each date)
                from datetime import date
                seen_dates = {}
                for md in market_data:
                    date_key = md.date if isinstance(md.date, date) else md.date.date()
                    # Compare dates properly
                    if date_key not in seen_dates:
                        seen_dates[date_key] = md
                    else:
                        # Keep the one with later timestamp if both are dates
                        existing_date = seen_dates[date_key].date if isinstance(seen_dates[date_key].date, date) else seen_dates[date_key].date.date()
                        current_date = md.date if isinstance(md.date, date) else md.date.date()
                        if current_date >= existing_date:
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
                    df = df.drop_duplicates(subset='date', keep='last')
                
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Ensure index is unique (final check)
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='last')]
                
                # Calculate features using enhanced feature engine
                feature_df = self.feature_engine._calculate_all_features(df, ticker)
                
                if feature_df is None or len(feature_df) < 60:
                    continue
                
                # Drop rows with NaN (from indicator calculations)
                feature_df = feature_df.dropna()
                
                if len(feature_df) < 60:
                    continue
                
                # Create training samples: for each day, use past 30 days as features, 
                # and future price (30 days ahead) as target
                # But we need to stop before the end so we have future data
                sequence_length = 30
                max_start_idx = len(feature_df) - sequence_length - prediction_horizon
                
                if max_start_idx < 0:
                    continue
                
                for i in range(sequence_length, max_start_idx):
                    # Features: use last 30 days
                    feature_window = feature_df.iloc[i-sequence_length:i]
                    
                    # Check if we have future data
                    current_date = feature_df.index[i]
                    future_date = current_date + timedelta(days=prediction_horizon)
                    
                    # Find future price
                    future_idx = None
                    for j in range(i + 1, min(i + prediction_horizon + 20, len(feature_df))):
                        if feature_df.index[j] >= future_date:
                            future_idx = j
                            break
                    
                    if future_idx is None:
                        continue
                    
                    current_price = feature_df.iloc[i]['close']
                    future_price = feature_df.iloc[future_idx]['close']
                    
                    # Flatten feature window
                    feature_vector = feature_window.values.flatten()
                    
                    # Skip if feature vector has NaN or Inf
                    if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
                        continue
                    
                    # Calculate targets
                    price_change_pct = ((future_price - current_price) / current_price) * 100
                    
                    # Signal target (BUY/SELL/HOLD)
                    if price_change_pct > 5:
                        signal = "BUY"
                    elif price_change_pct < -5:
                        signal = "SELL"
                    else:
                        signal = "HOLD"
                    
                    # Return target (percentage)
                    return_pct = price_change_pct
                    
                    # Target price (future price)
                    target_price = future_price
                    
                    feature_data.append(feature_vector)
                    signal_targets.append(signal)
                    return_targets.append(return_pct)
                    target_price_targets.append(target_price)
                    
            except Exception as e:
                logger.debug(f"Error processing {ticker}: {e}")
                continue
        
        if len(feature_data) < 100:
            logger.warning(f"Insufficient training data: {len(feature_data)} samples")
            return None, None, None, None
        
        X = pd.DataFrame(feature_data)
        y_signal = pd.Series(signal_targets)
        y_return = pd.Series(return_targets)
        y_target_price = pd.Series(target_price_targets)
        
        logger.success(f"✅ Prepared {len(X)} training samples with {X.shape[1]} features")
        
        return X, y_signal, y_return, y_target_price
    
    def train_signal_classifier(self, model_name: str = "SignalClassifier_v1") -> Optional[object]:
        """Train classification model for BUY/SELL/HOLD signals"""
        logger.info(f"🎯 Training signal classifier: {model_name}...")
        
        X, y_signal, _, _ = self.prepare_training_data()
        
        if X is None:
            return None
        
        # Split data (time-series aware)
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]  # Use last split
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_signal.iloc[train_idx], y_signal.iloc[test_idx]
        
        # Encode string labels to numeric
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost classifier
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            early_stopping_rounds=20
        )
        
        model.fit(
            X_train_scaled, y_train_encoded,
            eval_set=[(X_test_scaled, y_test_encoded)],
            verbose=False
        )
        
        # Evaluate
        train_pred_encoded = model.predict(X_train_scaled)
        test_pred_encoded = model.predict(X_test_scaled)
        
        # Decode predictions back to string labels for evaluation
        train_pred = label_encoder.inverse_transform(train_pred_encoded)
        test_pred = label_encoder.inverse_transform(test_pred_encoded)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        metrics = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "class_distribution": y_signal.value_counts().to_dict()
        }
        
        logger.success(f"✅ Signal Classifier trained:")
        logger.info(f"   Train Accuracy: {train_acc:.2%}")
        logger.info(f"   Test Accuracy: {test_acc:.2%}")
        logger.info(f"   Test F1-Score: {test_f1:.3f}")
        
        # Register model
        registry_entry = self.registry.register_model(
            model_name=model_name,
            model_type="XGBoostClassifier",
            model_object={"model": model, "scaler": self.scaler, "label_encoder": label_encoder},
            performance_metrics=metrics,
            description="Signal classification model (BUY/SELL/HOLD) with 30-day horizon",
            index_id=self.index_id
        )
        
        if registry_entry:
            logger.success(f"✅ Model {model_name} registered")
            return {"model": model, "scaler": self.scaler}
        
        return None
    
    def train_return_predictor(self, model_name: str = "ReturnPredictor_v1") -> Optional[object]:
        """Train regression model for expected return prediction"""
        logger.info(f"📈 Training return predictor: {model_name}...")
        
        X, _, y_return, _ = self.prepare_training_data()
        
        if X is None:
            return None
        
        # Split data
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_return.iloc[train_idx], y_return.iloc[test_idx]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost regressor
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=20
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        # Directional accuracy
        train_dir_acc = np.mean((y_train > 0) == (train_pred > 0))
        test_dir_acc = np.mean((y_test > 0) == (test_pred > 0))
        
        metrics = {
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
            "train_directional_accuracy": float(train_dir_acc),
            "test_directional_accuracy": float(test_dir_acc),
            "n_samples": len(X),
            "n_features": X.shape[1]
        }
        
        logger.success(f"✅ Return Predictor trained:")
        logger.info(f"   Test MAE: {test_mae:.2f}%")
        logger.info(f"   Test RMSE: {test_rmse:.2f}%")
        logger.info(f"   Test R²: {test_r2:.3f}")
        logger.info(f"   Directional Accuracy: {test_dir_acc:.2%}")
        
        # Register model
        registry_entry = self.registry.register_model(
            model_name=model_name,
            model_type="XGBoostRegressor",
            model_object={"model": model, "scaler": self.scaler},
            performance_metrics=metrics,
            description="Return prediction model (expected % return in 30 days)",
            index_id=self.index_id
        )
        
        if registry_entry:
            logger.success(f"✅ Model {model_name} registered")
            return {"model": model, "scaler": self.scaler}
        
        return None
    
    def train_target_price_predictor(self, model_name: str = "TargetPricePredictor_v1") -> Optional[object]:
        """Train regression model for target price prediction"""
        logger.info(f"🎯 Training target price predictor: {model_name}...")
        
        X, _, _, y_target_price = self.prepare_training_data()
        
        if X is None:
            return None
        
        # Split data
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X))
        train_idx, test_idx = splits[-1]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_target_price.iloc[train_idx], y_target_price.iloc[test_idx]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost regressor
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse',
            early_stopping_rounds=20
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        metrics = {
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
            "n_samples": len(X),
            "n_features": X.shape[1]
        }
        
        logger.success(f"✅ Target Price Predictor trained:")
        logger.info(f"   Test MAE: ₹{test_mae:.2f}")
        logger.info(f"   Test RMSE: ₹{test_rmse:.2f}")
        logger.info(f"   Test R²: {test_r2:.3f}")
        
        # Register model
        registry_entry = self.registry.register_model(
            model_name=model_name,
            model_type="XGBoostRegressor",
            model_object={"model": model, "scaler": self.scaler},
            performance_metrics=metrics,
            description="Target price prediction model (30-day target price)",
            index_id=self.index_id
        )
        
        if registry_entry:
            logger.success(f"✅ Model {model_name} registered")
            return {"model": model, "scaler": self.scaler}
        
        return None
    
    def train_all_models(self, index_name: str = ""):
        """
        Train all signal generation models
        
        Args:
            index_name: Normalized index name (e.g., "NIFTY50") to include in model names.
                       If empty, models will be generic (not index-specific).
        """
        suffix = f"_{index_name}" if index_name else ""
        
        index_info = f" for index {index_name}" if index_name else ""
        logger.info(f"🚀 Training all signal generation models{index_info}...")
        
        # Train signal classifier
        signal_model_name = f"SignalClassifier{suffix}_v1"
        logger.info(f"Training {signal_model_name}...")
        signal_model = self.train_signal_classifier(signal_model_name)
        
        # Train return predictor
        return_model_name = f"ReturnPredictor{suffix}_v1"
        logger.info(f"Training {return_model_name}...")
        return_model = self.train_return_predictor(return_model_name)
        
        # Train target price predictor
        target_model_name = f"TargetPricePredictor{suffix}_v1"
        logger.info(f"Training {target_model_name}...")
        target_model = self.train_target_price_predictor(target_model_name)
        
        logger.success("✅ All signal models trained!")
        
        return {
            "signal_classifier": signal_model,
            "return_predictor": return_model,
            "target_price_predictor": target_model
        }
    
    def close(self):
        """Close database sessions"""
        self.session.close()
        self.feature_engine.close()


if __name__ == "__main__":
    trainer = SignalModelTrainer()
    trainer.train_all_models()
    trainer.close()

