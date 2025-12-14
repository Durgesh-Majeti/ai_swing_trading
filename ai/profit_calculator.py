"""
Profit Potential Calculator - Calculates profit potential from model predictions
"""

import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger


class ProfitPotentialCalculator:
    """Calculates profit potential metrics from model predictions"""
    
    @staticmethod
    def calculate_profit_metrics(
        current_price: float,
        predicted_signal: str,
        predicted_return_pct: float,
        predicted_target_price: float,
        confidence_score: float = 0.5,
        stop_loss_pct: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate comprehensive profit potential metrics
        
        Args:
            current_price: Current stock price
            predicted_signal: BUY/SELL/HOLD
            predicted_return_pct: Predicted return percentage
            predicted_target_price: Predicted target price
            confidence_score: Model confidence (0-1)
            stop_loss_pct: Stop loss percentage (default 5%)
        
        Returns:
            Dictionary with profit metrics
        """
        if predicted_signal == "HOLD":
            return {
                "profit_potential_pct": 0.0,
                "profit_potential_absolute": 0.0,
                "risk_reward_ratio": 0.0,
                "expected_profit": 0.0,
                "stop_loss_price": current_price,
                "target_price": current_price,
                "risk_amount": 0.0,
                "reward_amount": 0.0,
                "profit_score": 0.0
            }
        
        # Calculate stop loss
        if predicted_signal == "BUY":
            stop_loss_price = current_price * (1 - stop_loss_pct)
            target_price = max(predicted_target_price, current_price * (1 + abs(predicted_return_pct) / 100))
        else:  # SELL
            stop_loss_price = current_price * (1 + stop_loss_pct)
            target_price = min(predicted_target_price, current_price * (1 - abs(predicted_return_pct) / 100))
        
        # Calculate profit/loss amounts
        if predicted_signal == "BUY":
            reward_amount = target_price - current_price
            risk_amount = current_price - stop_loss_price
        else:  # SELL (short)
            reward_amount = current_price - target_price
            risk_amount = stop_loss_price - current_price
        
        # Risk-reward ratio
        if risk_amount > 0:
            risk_reward_ratio = reward_amount / risk_amount
        else:
            risk_reward_ratio = 0.0
        
        # Profit potential (percentage)
        profit_potential_pct = (reward_amount / current_price) * 100
        
        # Expected profit (weighted by confidence)
        expected_profit = reward_amount * confidence_score
        
        # Profit score (composite metric)
        # Combines profit potential, risk-reward ratio, and confidence
        profit_score = (
            (profit_potential_pct / 100) * 0.4 +  # 40% weight on profit potential
            (min(risk_reward_ratio / 3, 1.0)) * 0.4 +  # 40% weight on risk-reward (capped at 3:1)
            confidence_score * 0.2  # 20% weight on confidence
        ) * 100  # Scale to 0-100
        
        return {
            "profit_potential_pct": float(profit_potential_pct),
            "profit_potential_absolute": float(reward_amount),
            "risk_reward_ratio": float(risk_reward_ratio),
            "expected_profit": float(expected_profit),
            "stop_loss_price": float(stop_loss_price),
            "target_price": float(target_price),
            "risk_amount": float(risk_amount),
            "reward_amount": float(reward_amount),
            "profit_score": float(profit_score)
        }
    
    @staticmethod
    def calculate_position_size(
        capital: float,
        risk_amount: float,
        max_risk_per_trade: float = 0.02  # 2% of capital per trade
    ) -> int:
        """
        Calculate position size based on risk
        
        Args:
            capital: Available capital
            risk_amount: Risk per share (current_price - stop_loss)
            max_risk_per_trade: Maximum risk as fraction of capital
        
        Returns:
            Number of shares to buy
        """
        if risk_amount <= 0:
            return 0
        
        max_risk_dollars = capital * max_risk_per_trade
        position_size = int(max_risk_dollars / risk_amount)
        
        return max(0, position_size)
    
    @staticmethod
    def prioritize_signals(
        signals: list,
        min_profit_score: float = 50.0,
        min_risk_reward: float = 1.5
    ) -> list:
        """
        Prioritize signals based on profit potential
        
        Args:
            signals: List of signal dictionaries with profit metrics
            min_profit_score: Minimum profit score to include
            min_risk_reward: Minimum risk-reward ratio
        
        Returns:
            Sorted list of signals (best first)
        """
        filtered = [
            s for s in signals
            if s.get("profit_score", 0) >= min_profit_score
            and s.get("risk_reward_ratio", 0) >= min_risk_reward
        ]
        
        # Sort by profit score (descending)
        sorted_signals = sorted(
            filtered,
            key=lambda x: x.get("profit_score", 0),
            reverse=True
        )
        
        return sorted_signals

