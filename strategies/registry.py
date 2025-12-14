"""
Strategy Registry - Auto-discovers and loads strategies
"""

import importlib
import inspect
from pathlib import Path
from strategies.base import BaseStrategy
from loguru import logger
from typing import List, Dict

class StrategyRegistry:
    """Manages strategy discovery and loading"""
    
    def __init__(self, strategies_dir: str = "strategies"):
        self.strategies_dir = Path(strategies_dir)
        self.strategies: Dict[str, BaseStrategy] = {}
        self._discover_strategies()
    
    def _discover_strategies(self):
        """Auto-discover all strategy classes"""
        logger.info("🔍 Discovering strategies...")
        
        # Get all Python files in strategies directory
        strategy_files = list(self.strategies_dir.glob("*.py"))
        strategy_files = [f for f in strategy_files if f.name != "__init__.py" and f.name != "base.py" and f.name != "registry.py"]
        
        for file_path in strategy_files:
            try:
                module_name = f"strategies.{file_path.stem}"
                module = importlib.import_module(module_name)
                
                # Find all classes that inherit from BaseStrategy
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy and 
                        obj.__module__ == module_name):
                        
                        # Instantiate strategy
                        strategy = obj()
                        self.strategies[strategy.name] = strategy
                        logger.success(f"✅ Loaded strategy: {strategy.name}")
                        
            except Exception as e:
                logger.warning(f"⚠️  Failed to load strategy from {file_path}: {e}")
        
        logger.info(f"📊 Total strategies loaded: {len(self.strategies)}")
    
    def get_strategy(self, name: str) -> BaseStrategy:
        """Get a strategy by name"""
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all available strategy names"""
        return list(self.strategies.keys())
    
    def run_all_strategies(self, ticker: str):
        """Run all strategies for a ticker and return signals"""
        signals = []
        
        for name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(ticker)
                if signal:
                    signal['strategy_name'] = name
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error running strategy {name} for {ticker}: {e}")
        
        return signals

