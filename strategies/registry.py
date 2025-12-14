"""
Strategy Registry - Auto-discovers and loads strategies
"""

import importlib
import inspect
from pathlib import Path
from strategies.base import BaseStrategy
from loguru import logger
from typing import List, Dict, Optional

class StrategyRegistry:
    """Manages strategy discovery and loading"""
    
    def __init__(self, strategies_dir: str = "strategies", index_id: Optional[int] = None, index_name: Optional[str] = None):
        """
        Initialize Strategy Registry
        
        Args:
            strategies_dir: Directory containing strategy files
            index_id: Optional index ID to pass to index-aware strategies
            index_name: Optional index name to pass to index-aware strategies
        """
        self.strategies_dir = Path(strategies_dir)
        self.strategies: Dict[str, BaseStrategy] = {}
        self.index_id = index_id
        self.index_name = index_name
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
                        
                        # Check if strategy accepts index_id or index_name
                        sig = inspect.signature(obj.__init__)
                        params = sig.parameters
                        
                        # Instantiate strategy with index info if supported
                        if 'index_id' in params or 'index_name' in params:
                            kwargs = {}
                            if 'index_id' in params:
                                kwargs['index_id'] = self.index_id
                            if 'index_name' in params:
                                kwargs['index_name'] = self.index_name
                            strategy = obj(**kwargs)
                        else:
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

