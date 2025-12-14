"""
Technical Analysis Strategy - RSI + MACD + Moving Averages
"""

from strategies.base import BaseStrategy
from database.models import TechnicalIndicators
from typing import Optional, Dict, Any
from loguru import logger

class TechnicalStrategy(BaseStrategy):
    """Technical analysis based strategy"""
    
    def __init__(self):
        super().__init__("Technical_RSI_MACD")
    
    def generate_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Generate signal based on technical indicators"""
        try:
            indicators = self.get_latest_indicators(ticker)
            current_price = self.get_latest_price(ticker)
            
            if not indicators or not current_price:
                return None
            
            # Strategy Logic: Buy when RSI < 30 (oversold) and MACD bullish
            # Sell when RSI > 70 (overbought) and MACD bearish
            
            signal = None
            reasoning_parts = []
            
            # RSI Analysis
            if indicators.rsi_14:
                if indicators.rsi_14 < 30:
                    signal = "BUY"
                    reasoning_parts.append(f"RSI oversold ({indicators.rsi_14:.2f})")
                elif indicators.rsi_14 > 70:
                    signal = "SELL"
                    reasoning_parts.append(f"RSI overbought ({indicators.rsi_14:.2f})")
            
            # MACD Analysis
            if indicators.macd and indicators.macd_signal:
                if indicators.macd > indicators.macd_signal:
                    if signal == "BUY":
                        reasoning_parts.append("MACD bullish crossover")
                    elif signal is None:
                        signal = "BUY"
                        reasoning_parts.append("MACD bullish")
                elif indicators.macd < indicators.macd_signal:
                    if signal == "SELL":
                        reasoning_parts.append("MACD bearish crossover")
                    elif signal is None:
                        signal = "SELL"
                        reasoning_parts.append("MACD bearish")
            
            # Moving Average Analysis
            if indicators.sma_50 and indicators.sma_200:
                if current_price > indicators.sma_50 > indicators.sma_200:
                    if signal == "BUY":
                        reasoning_parts.append("Price above both SMAs (bullish trend)")
                elif current_price < indicators.sma_50 < indicators.sma_200:
                    if signal == "SELL":
                        reasoning_parts.append("Price below both SMAs (bearish trend)")
            
            if signal is None:
                return None
            
            # Calculate stop loss and target
            if indicators.atr:
                atr_multiplier = 2.0
                stop_loss_pct = (indicators.atr * atr_multiplier) / current_price
            else:
                stop_loss_pct = 0.05  # Default 5% stop loss
            
            if signal == "BUY":
                stop_loss = current_price * (1 - stop_loss_pct)
                target_price = current_price * (1 + stop_loss_pct * 2)  # 2:1 risk/reward
            else:  # SELL
                stop_loss = current_price * (1 + stop_loss_pct)
                target_price = current_price * (1 - stop_loss_pct * 2)
            
            quantity = self.calculate_position_size(ticker)
            
            return {
                "signal": signal,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "quantity": quantity,
                "reasoning": " | ".join(reasoning_parts)
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {ticker}: {e}")
            return None

