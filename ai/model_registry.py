"""
Model Registry - Version control for AI models
"""

import pickle
import json
from pathlib import Path
from database.models import SessionLocal, ModelRegistry
from sqlalchemy import select
from loguru import logger
from datetime import datetime

class ModelRegistryManager:
    """Manages model versioning and activation"""
    
    def __init__(self, models_dir: str = "models"):
        self.session = SessionLocal()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def register_model(self, model_name: str, model_type: str, model_object, 
                      performance_metrics: dict = None, description: str = "", index_id: int = None):
        """Register a new model in the registry"""
        try:
            # Save model to disk
            model_path = self.models_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_object, f)
            
            # Extract version from name (e.g., "LSTM_Nifty_v2" -> "v2")
            version = model_name.split('_v')[-1] if '_v' in model_name else "1"
            
            # Create registry entry
            registry_entry = ModelRegistry(
                model_name=model_name,
                version=version,
                model_type=model_type,
                file_path=str(model_path),
                index_id=index_id,  # Store index_id for index-specific filtering
                is_active=False,  # New models are inactive by default (composite strategy doesn't need activation)
                trained_on_date=datetime.now().date(),
                performance_metrics=json.dumps(performance_metrics) if performance_metrics else None,
                description=description
            )
            
            self.session.add(registry_entry)
            self.session.commit()
            
            logger.success(f"✅ Model {model_name} registered")
            return registry_entry
            
        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
            self.session.rollback()
            return None
    
    def activate_model(self, model_name: str):
        """Activate a model (deactivates all others)"""
        try:
            # Deactivate all models
            all_models = self.session.scalars(select(ModelRegistry)).all()
            for model in all_models:
                model.is_active = False
            
            # Activate specified model
            stmt = select(ModelRegistry).filter_by(model_name=model_name)
            model = self.session.scalars(stmt).first()
            
            if not model:
                logger.error(f"Model {model_name} not found")
                return False
            
            model.is_active = True
            self.session.commit()
            
            logger.success(f"✅ Model {model_name} activated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to activate model: {e}")
            self.session.rollback()
            return False
    
    def get_active_model(self):
        """Get the currently active model"""
        stmt = select(ModelRegistry).filter_by(is_active=True)
        model_reg = self.session.scalars(stmt).first()
        return model_reg
    
    def load_active_model(self):
        """Load the active model object from disk"""
        model_reg = self.get_active_model()
        
        if not model_reg:
            logger.warning("No active model found")
            return None
        
        try:
            with open(model_reg.file_path, 'rb') as f:
                model = pickle.load(f)
            return model, model_reg
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def list_models(self):
        """List all registered models"""
        models = self.session.scalars(select(ModelRegistry)).all()
        return models
    
    def get_model_info(self, model_name: str):
        """Get information about a specific model"""
        stmt = select(ModelRegistry).filter_by(model_name=model_name)
        return self.session.scalars(stmt).first()
    
    def get_model(self, model_name: str):
        """Get a model by name (returns registry entry)"""
        stmt = select(ModelRegistry).filter_by(model_name=model_name).order_by(
            ModelRegistry.trained_on_date.desc()
        )
        return self.session.scalars(stmt).first()

