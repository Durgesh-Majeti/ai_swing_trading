"""
Hybrid Strategy - Combines Technical, Fundamental, and AI predictions
"""

from strategies.base import BaseStrategy
from database.models import AIPredictions, FundamentalData
from typing import Optional, Dict, Any
from loguru import logger

class HybridStrategy(BaseStrategy):
    """Hybrid strategy combining multiple signals"""
    
    def __init__(self):
        super().__init__("Hybrid_AI_Technical_Fundamental")
    
    def generate_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Generate signal using hybrid approach"""
        try:
            current_price = self.get_latest_price(ticker)
            if not current_price:
                return None
            
            # Get all inputs
            indicators = self.get_latest_indicators(ticker)
            ai_prediction = self.get_latest_prediction(ticker)
            vix = self.get_latest_macro("INDIA_VIX")
            
            # Get fundamentals
            from database.models import FundamentalData
            from sqlalchemy import select
            stmt = select(FundamentalData).filter_by(ticker=ticker).order_by(
                FundamentalData.report_date.desc()
            )
            fundamental = self.session.scalars(stmt).first()
            
            # Scoring system
            buy_score = 0
            sell_score = 0
            reasoning_parts = []
            
            # Check if AI predictions are available
            has_ai = ai_prediction is not None
            
            # 1. AI Prediction (weight: 40% if available, otherwise redistributed)
            if ai_prediction:
                if ai_prediction.direction == "UP" and ai_prediction.confidence_score > 0.7:
                    buy_score += 40
                    reasoning_parts.append(f"AI: {ai_prediction.direction} (Confidence: {ai_prediction.confidence_score:.0%})")
                elif ai_prediction.direction == "DOWN" and ai_prediction.confidence_score > 0.7:
                    sell_score += 40
                    reasoning_parts.append(f"AI: {ai_prediction.direction} (Confidence: {ai_prediction.confidence_score:.0%})")
            else:
                # No AI available - note in reasoning
                reasoning_parts.append("AI: Not available (using technical/fundamental only)")
            
            # 2. Technical Indicators (weight: 30% normally, 50% if no AI)
            tech_weight = 50 if not has_ai else 30
            if indicators:
                if indicators.rsi_14:
                    if indicators.rsi_14 < 35:
                        buy_score += (tech_weight // 2)
                        reasoning_parts.append(f"RSI oversold ({indicators.rsi_14:.1f})")
                    elif indicators.rsi_14 > 65:
                        sell_score += (tech_weight // 2)
                        reasoning_parts.append(f"RSI overbought ({indicators.rsi_14:.1f})")
                
                if indicators.macd and indicators.macd_signal:
                    if indicators.macd > indicators.macd_signal:
                        buy_score += (tech_weight // 2)
                    else:
                        sell_score += (tech_weight // 2)
            
            # 3. Fundamentals (weight: 20% normally, 30% if no AI)
            fund_weight = 30 if not has_ai else 20
            if fundamental:
                if fundamental.pe_ratio and fundamental.pe_ratio < 20:  # Undervalued
                    buy_score += (fund_weight // 2)
                    reasoning_parts.append(f"Low P/E ({fundamental.pe_ratio:.1f})")
                elif fundamental.pe_ratio and fundamental.pe_ratio > 40:  # Overvalued
                    sell_score += (fund_weight // 2)
                    reasoning_parts.append(f"High P/E ({fundamental.pe_ratio:.1f})")
                
                if fundamental.roe and fundamental.roe > 15:  # Good profitability
                    buy_score += (fund_weight // 2)
                    reasoning_parts.append(f"Strong ROE ({fundamental.roe:.1f}%)")
            
            # 4. Market Sentiment (weight: 10% - unchanged)
            if vix:
                if vix < 15:  # Low volatility, good for entry
                    buy_score += 5
                    reasoning_parts.append(f"Low VIX ({vix:.1f})")
                elif vix > 25:  # High volatility, risky
                    sell_score += 5
                    reasoning_parts.append(f"High VIX ({vix:.1f})")
            
            # Determine signal (adjust threshold if no AI - lower threshold since max score is lower)
            signal = None
            threshold = 50 if not has_ai else 60  # Lower threshold when AI unavailable
            if buy_score >= threshold:
                signal = "BUY"
            elif sell_score >= threshold:
                signal = "SELL"
            
            if signal is None:
                return None
            
            # Calculate stop loss and target
            if indicators and indicators.atr:
                atr_multiplier = 2.0
                stop_loss_pct = (indicators.atr * atr_multiplier) / current_price
            else:
                stop_loss_pct = 0.05
            
            if signal == "BUY":
                stop_loss = current_price * (1 - stop_loss_pct)
                target_price = current_price * (1 + stop_loss_pct * 2)
            else:
                stop_loss = current_price * (1 + stop_loss_pct)
                target_price = current_price * (1 - stop_loss_pct * 2)
            
            quantity = self.calculate_position_size(ticker)
            
            return {
                "signal": signal,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "quantity": quantity,
                "reasoning": " | ".join(reasoning_parts) if reasoning_parts else "Hybrid analysis"
            }
            
        except Exception as e:
            logger.error(f"Error generating hybrid signal for {ticker}: {e}")
            return None

