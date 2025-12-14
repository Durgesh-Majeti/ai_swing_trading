"""
Base Strategy Class - All strategies inherit from this
"""

from abc import ABC, abstractmethod
from database.models import SessionLocal, TradeSignal, MarketData, TechnicalIndicators, AIPredictions, MacroIndicator
from sqlalchemy import select, func
from loguru import logger
from datetime import datetime
from typing import Optional, Dict, Any

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.session = SessionLocal()
    
    @abstractmethod
    def generate_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Generate a trade signal for a ticker
        
        Returns:
            Dict with keys: signal ("BUY"/"SELL"/"HOLD"), entry_price, stop_loss, target_price, quantity, reasoning
            Or None if no signal
        """
        pass
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get latest closing price"""
        stmt = select(MarketData).filter_by(ticker=ticker).order_by(MarketData.date.desc())
        latest = self.session.scalars(stmt).first()
        return latest.close if latest else None
    
    def get_latest_indicators(self, ticker: str) -> Optional[TechnicalIndicators]:
        """Get latest technical indicators"""
        stmt = select(TechnicalIndicators).filter_by(ticker=ticker).order_by(TechnicalIndicators.date.desc())
        return self.session.scalars(stmt).first()
    
    def get_latest_prediction(self, ticker: str) -> Optional[AIPredictions]:
        """Get latest AI prediction"""
        stmt = select(AIPredictions).filter_by(ticker=ticker).order_by(AIPredictions.generated_at.desc())
        return self.session.scalars(stmt).first()
    
    def get_latest_macro(self, indicator_name: str) -> Optional[float]:
        """Get latest macro indicator value"""
        stmt = select(MacroIndicator).filter_by(indicator_name=indicator_name).order_by(
            MacroIndicator.date.desc()
        )
        latest = self.session.scalars(stmt).first()
        return latest.value if latest else None
    
    def calculate_position_size(self, ticker: str, risk_per_trade: float = 0.02) -> int:
        """
        Calculate position size based on risk management
        risk_per_trade: Percentage of capital to risk per trade (default 2%)
        """
        # Simplified position sizing
        # In production, this would consider available capital, portfolio value, etc.
        base_quantity = 10  # Base position size
        return base_quantity
    
    def close(self):
        """Close database session"""
        self.session.close()

