"""
Risk Management Layer - Pre-trade checks and position sizing
"""

from database.models import SessionLocal, Portfolio, Order, MacroIndicator, TradeSignal
from sqlalchemy import select, func
from loguru import logger
from typing import Dict, Optional, Tuple

class RiskManager:
    """Manages risk checks before order execution"""
    
    def __init__(self, max_capital_per_trade: float = 0.10, max_sector_exposure: float = 0.20):
        self.session = SessionLocal()
        self.max_capital_per_trade = max_capital_per_trade  # 10% max per trade
        self.max_sector_exposure = max_sector_exposure  # 20% max per sector
    
    def validate_signal(self, signal: TradeSignal) -> Tuple[bool, str]:
        """
        Validate a trade signal against risk rules
        
        Returns:
            (is_valid, reason)
        """
        try:
            # 1. Capital Check
            if not self._check_capital(signal):
                return False, "Insufficient capital"
            
            # 2. Exposure Check
            exposure_check, reason = self._check_sector_exposure(signal)
            if not exposure_check:
                return False, f"Sector exposure limit: {reason}"
            
            # 3. Volatility Check
            if not self._check_market_volatility():
                return False, "Market volatility too high (VIX > 25)"
            
            # 4. Existing Position Check
            if not self._check_existing_position(signal):
                return False, "Already have position in this stock"
            
            # 5. Position Size Validation
            if not self._validate_position_size(signal):
                return False, "Position size too large"
            
            return True, "All checks passed"
            
        except Exception as e:
            logger.error(f"Error in risk validation: {e}")
            return False, f"Validation error: {e}"
    
    def _check_capital(self, signal: TradeSignal) -> bool:
        """Check if we have enough capital"""
        # Simplified: In production, this would check actual account balance
        # For now, we assume we have sufficient capital
        return True
    
    def _check_sector_exposure(self, signal: TradeSignal) -> Tuple[bool, str]:
        """Check sector exposure limits"""
        # Get company sector
        from database.models import CompanyProfile
        stmt = select(CompanyProfile).filter_by(ticker=signal.ticker)
        company = self.session.scalars(stmt).first()
        
        if not company or not company.sector:
            return True, ""  # Can't check without sector info
        
        # Calculate current sector exposure
        stmt = select(Portfolio).join(CompanyProfile).filter_by(sector=company.sector)
        sector_positions = self.session.scalars(stmt).all()
        
        if not sector_positions:
            return True, ""
        
        # Calculate total portfolio value (simplified)
        total_value = sum(
            pos.current_price * pos.quantity for pos in sector_positions
        )
        
        # Calculate new position value
        new_position_value = signal.entry_price * signal.quantity
        
        # In production, compare against total portfolio value
        # For now, just check if we're adding too much to one sector
        if len(sector_positions) >= 5:  # Already have 5 positions in this sector
            return False, f"Too many positions in {company.sector} sector"
        
        return True, ""
    
    def _check_market_volatility(self) -> bool:
        """Check if market volatility is acceptable"""
        stmt = select(MacroIndicator).filter_by(indicator_name="INDIA_VIX").order_by(
            MacroIndicator.date.desc()
        )
        latest_vix = self.session.scalars(stmt).first()
        
        if latest_vix and latest_vix.value > 25:
            return False  # Too volatile
        
        return True
    
    def _check_existing_position(self, signal: TradeSignal) -> bool:
        """Check if we already have a position"""
        stmt = select(Portfolio).filter_by(ticker=signal.ticker)
        existing = self.session.scalars(stmt).first()
        
        if existing and existing.quantity > 0:
            # For SELL signals, having a position is OK
            if signal.signal == "SELL":
                return True
            # For BUY signals, having a position means we're trying to add
            # This could be allowed or not based on strategy
            return False  # Don't allow duplicate positions for now
        
        return True
    
    def _validate_position_size(self, signal: TradeSignal) -> bool:
        """Validate position size is reasonable"""
        position_value = signal.entry_price * signal.quantity
        
        # In production, compare against total capital
        # For now, just check reasonable limits
        if position_value > 1000000:  # 10 lakh limit
            return False
        
        if signal.quantity <= 0:
            return False
        
        return True
    
    def calculate_safe_position_size(self, signal: TradeSignal, available_capital: float) -> int:
        """Calculate safe position size based on risk"""
        # Risk-based position sizing
        risk_amount = available_capital * self.max_capital_per_trade
        
        # Calculate position size based on stop loss distance
        price_diff = abs(signal.entry_price - signal.stop_loss)
        if price_diff == 0:
            return signal.quantity
        
        risk_per_share = price_diff
        safe_quantity = int(risk_amount / risk_per_share)
        
        # Don't exceed original quantity
        return min(safe_quantity, signal.quantity)
    
    def close(self):
        """Close database session"""
        self.session.close()

