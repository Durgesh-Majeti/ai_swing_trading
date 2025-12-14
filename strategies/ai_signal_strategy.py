"""
AI Signal Strategy - Uses trained ML models to generate signals with profit potential
"""

from strategies.base import BaseStrategy
from database.models import AIPredictions, Index
from typing import Optional, Dict, Any
from loguru import logger
from ai.model_registry import ModelRegistryManager
from ai.enhanced_features import EnhancedFeatureEngine
from ai.profit_calculator import ProfitPotentialCalculator
from sqlalchemy import select
import numpy as np


def normalize_index_name(index_name: str) -> str:
    """Normalize index name for model lookup"""
    if not index_name:
        return ""
    return index_name.replace(" ", "").replace("_", "").upper()


class AISignalStrategy(BaseStrategy):
    """AI-powered strategy using signal generation models"""
    
    def __init__(self, index_id: Optional[int] = None, index_name: Optional[str] = None):
        """
        Initialize AI Signal Strategy
        
        Args:
            index_id: Optional index ID to load index-specific models
            index_name: Optional index name (alternative to index_id)
        """
        super().__init__("AI_Signal_Strategy")
        self.model_registry = ModelRegistryManager()
        self.feature_engine = EnhancedFeatureEngine(index_id=index_id)
        self.profit_calculator = ProfitPotentialCalculator()
        self.index_id = index_id
        self.index_name = index_name
        
        # Determine normalized index name for model lookup
        self.normalized_index_name = ""
        if index_id:
            from database.models import SessionLocal
            session = SessionLocal()
            try:
                index = session.scalar(select(Index).filter_by(id=index_id))
                if index:
                    self.index_name = index.name
                    self.normalized_index_name = normalize_index_name(index.name)
            finally:
                session.close()
        elif index_name:
            self.normalized_index_name = normalize_index_name(index_name)
        
        # Load models
        self.signal_model = None
        self.return_model = None
        self.target_price_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load signal generation models (index-specific if available, else generic)"""
        try:
            # Determine model names - try index-specific first, then generic
            if self.normalized_index_name:
                signal_model_name = f"SignalClassifier_{self.normalized_index_name}_v1"
                return_model_name = f"ReturnPredictor_{self.normalized_index_name}_v1"
                target_model_name = f"TargetPricePredictor_{self.normalized_index_name}_v1"
            else:
                signal_model_name = "SignalClassifier_v1"
                return_model_name = "ReturnPredictor_v1"
                target_model_name = "TargetPricePredictor_v1"
            
            # Try to load signal classifier (index-specific first, then generic fallback)
            signal_reg = self.model_registry.get_model(signal_model_name)
            if not signal_reg and self.normalized_index_name:
                # Fallback to generic model
                signal_reg = self.model_registry.get_model("SignalClassifier_v1")
            
            if signal_reg:
                import pickle
                with open(signal_reg.file_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.signal_model = model_data.get("model")
                    self.signal_scaler = model_data.get("scaler")
                    self.signal_label_encoder = model_data.get("label_encoder")
                logger.info(f"✅ Loaded Signal Classifier model: {signal_reg.model_name}")
            
            # Try to load return predictor
            return_reg = self.model_registry.get_model(return_model_name)
            if not return_reg and self.normalized_index_name:
                # Fallback to generic model
                return_reg = self.model_registry.get_model("ReturnPredictor_v1")
            
            if return_reg:
                import pickle
                with open(return_reg.file_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.return_model = model_data.get("model")
                    self.return_scaler = model_data.get("scaler")
                logger.info(f"✅ Loaded Return Predictor model: {return_reg.model_name}")
            
            # Try to load target price predictor
            target_reg = self.model_registry.get_model(target_model_name)
            if not target_reg and self.normalized_index_name:
                # Fallback to generic model
                target_reg = self.model_registry.get_model("TargetPricePredictor_v1")
            
            if target_reg:
                import pickle
                with open(target_reg.file_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.target_price_model = model_data.get("model")
                    self.target_scaler = model_data.get("scaler")
                logger.info(f"✅ Loaded Target Price Predictor model: {target_reg.model_name}")
            
            if not (self.signal_model or self.return_model or self.target_price_model):
                logger.warning("⚠️  No signal models loaded. Strategy will use fallback logic.")
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def generate_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Generate signal using AI models"""
        try:
            # Check if models are loaded
            if not (self.signal_model or self.return_model or self.target_price_model):
                logger.warning(f"⚠️  No models loaded for {ticker}. Cannot generate AI signal.")
                return None
            
            current_price = self.get_latest_price(ticker)
            if not current_price:
                logger.debug(f"No price data available for {ticker}")
                return None
            
            # Get features
            feature_sequence = self.feature_engine.get_sequence_features(ticker, sequence_length=30)
            if feature_sequence is None or len(feature_sequence) == 0:
                logger.debug(f"No features available for {ticker}")
                return None
            
            if len(feature_sequence) < 30:
                logger.debug(f"Insufficient features for {ticker}: {len(feature_sequence)} < 30")
                return None
            
            # Prepare feature vector (flatten last 30 days)
            feature_vector = feature_sequence.values.flatten().reshape(1, -1)
            
            # Log feature vector info for debugging
            logger.debug(f"{ticker}: Feature vector shape: {feature_vector.shape}, Features: {feature_sequence.shape[1]}")
            
            # Predict signal
            predicted_signal = "HOLD"
            confidence_score = 0.5
            predicted_return_pct = 0.0
            predicted_target_price = current_price
            
            if self.signal_model and self.signal_scaler:
                try:
                    # Scale features
                    feature_scaled = self.signal_scaler.transform(feature_vector)
                    
                    # Check feature dimension match
                    expected_features = self.signal_scaler.n_features_in_ if hasattr(self.signal_scaler, 'n_features_in_') else None
                    if expected_features and feature_vector.shape[1] != expected_features:
                        logger.warning(f"{ticker}: Feature dimension mismatch! Expected {expected_features}, got {feature_vector.shape[1]}")
                        return None
                    
                    # Predict signal (model returns encoded labels)
                    signal_pred_encoded = self.signal_model.predict(feature_scaled)[0]
                    signal_proba = self.signal_model.predict_proba(feature_scaled)[0]
                    
                    # Decode signal back to string (BUY/SELL/HOLD)
                    if hasattr(self, 'signal_label_encoder') and self.signal_label_encoder:
                        predicted_signal = self.signal_label_encoder.inverse_transform([signal_pred_encoded])[0]
                    else:
                        # Fallback if label_encoder not available
                        predicted_signal = signal_pred_encoded
                    
                    confidence_score = float(np.max(signal_proba))
                except Exception as e:
                    logger.debug(f"Error predicting signal for {ticker}: {e}")
            
            # Predict return
            if self.return_model and self.return_scaler:
                try:
                    feature_scaled = self.return_scaler.transform(feature_vector)
                    predicted_return_pct = float(self.return_model.predict(feature_scaled)[0])
                except Exception as e:
                    logger.debug(f"Error predicting return for {ticker}: {e}")
            
            # Predict target price
            if self.target_price_model and self.target_scaler:
                try:
                    feature_scaled = self.target_scaler.transform(feature_vector)
                    predicted_target_price = float(self.target_price_model.predict(feature_scaled)[0])
                except Exception as e:
                    logger.debug(f"Error predicting target price for {ticker}: {e}")
            
            # Only generate signal if not HOLD
            if predicted_signal == "HOLD":
                logger.debug(f"{ticker}: Signal is HOLD, skipping")
                return None
            
            # Calculate profit potential
            profit_metrics = self.profit_calculator.calculate_profit_metrics(
                current_price=current_price,
                predicted_signal=predicted_signal,
                predicted_return_pct=predicted_return_pct,
                predicted_target_price=predicted_target_price,
                confidence_score=confidence_score
            )
            
            # Filter by minimum profit score
            if profit_metrics["profit_score"] < 40:  # Minimum threshold
                logger.debug(f"{ticker}: Profit score {profit_metrics['profit_score']:.1f} < 40, filtering out")
                return None
            
            # Calculate position size
            quantity = self.profit_calculator.calculate_position_size(
                capital=1000000,  # Default capital (can be made configurable)
                risk_amount=profit_metrics["risk_amount"]
            )
            
            if quantity == 0:
                return None
            
            # Build reasoning
            reasoning_parts = [
                f"AI Signal: {predicted_signal}",
                f"Confidence: {confidence_score:.0%}",
                f"Expected Return: {predicted_return_pct:.2f}%",
                f"Profit Score: {profit_metrics['profit_score']:.1f}",
                f"Risk-Reward: {profit_metrics['risk_reward_ratio']:.2f}:1"
            ]
            
            return {
                "signal": predicted_signal,
                "entry_price": current_price,
                "stop_loss": profit_metrics["stop_loss_price"],
                "target_price": profit_metrics["target_price"],
                "quantity": quantity,
                "reasoning": " | ".join(reasoning_parts),
                "profit_potential_pct": profit_metrics["profit_potential_pct"],
                "profit_score": profit_metrics["profit_score"],
                "risk_reward_ratio": profit_metrics["risk_reward_ratio"],
                "confidence_score": confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error generating AI signal for {ticker}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def close(self):
        """Close database sessions"""
        super().close()
        self.feature_engine.close()

