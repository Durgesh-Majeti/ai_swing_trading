# Signal Generation Models Guide

## Overview

This guide explains how to use the AI/ML models that generate trading signals with profit potential. These models are designed to output signals (BUY/SELL/HOLD) along with profit potential metrics that can be used to build trading logic.

## Model Architecture

### 1. Signal Classifier Model
- **Type**: XGBoost Classifier
- **Output**: BUY, SELL, or HOLD signal
- **Confidence**: Probability scores for each class
- **Purpose**: Determines whether to enter a trade and in which direction

### 2. Return Predictor Model
- **Type**: XGBoost Regressor
- **Output**: Expected return percentage (e.g., +5.2% or -3.1%)
- **Purpose**: Predicts how much profit/loss is expected

### 3. Target Price Predictor Model
- **Type**: XGBoost Regressor
- **Output**: Predicted target price (absolute value)
- **Purpose**: Predicts the target price for the trade

## Features Used

The models use **enhanced features** including:

### Technical Features
- Price data (OHLCV)
- Technical indicators (RSI, MACD, SMA, EMA, ATR, Bollinger Bands)
- Lag features (price/volume from previous days)
- Rolling statistics (mean, std, max, min over windows)
- Momentum indicators
- Volatility measures
- Pattern detection (breakouts, support/resistance)

### Fundamental Features
- PE Ratio, PB Ratio
- ROE, EPS
- Debt-to-Equity
- Revenue Growth

### Macro Features
- VIX (Volatility Index)
- Crude Oil prices
- USD/INR exchange rate

**Total Features**: 2490+ features per sample (including sequence features)

## Training the Models

### Basic Training (All Indices)
```bash
python train_signal_models.py
```

### Index-Specific Training
```bash
python train_signal_models.py NIFTY_50
python train_signal_models.py NIFTY_MIDCAP_100
```

This will train three models:
1. `SignalClassifier_NIFTY_50_v1` (index-specific naming)
2. `ReturnPredictor_NIFTY_50_v1`
3. `TargetPricePredictor_NIFTY_50_v1`

**Note**: Models are automatically associated with their index and only visible in the dashboard for that specific index.

## Using the Models

### Automatic Usage (Recommended)

The **AI_Signal_Strategy** automatically uses these models:

1. **Strategy Registration**: The strategy is auto-discovered by the Strategy Registry
2. **Model Loading**: Models are loaded automatically when the strategy is instantiated
3. **Signal Generation**: The strategy generates signals using the models

### Manual Usage

```python
from ai.signal_models import SignalModelTrainer
from ai.model_registry import ModelRegistryManager
from ai.enhanced_features import EnhancedFeatureEngine
from ai.profit_calculator import ProfitPotentialCalculator

# Load models
registry = ModelRegistryManager()
signal_reg = registry.get_model("SignalClassifier_v1")
return_reg = registry.get_model("ReturnPredictor_v1")
target_reg = registry.get_model("TargetPricePredictor_v1")

# Get features
feature_engine = EnhancedFeatureEngine()
features = feature_engine.get_sequence_features("RELIANCE", sequence_length=30)
feature_vector = features.values.flatten().reshape(1, -1)

# Make predictions
import pickle
with open(signal_reg.file_path, 'rb') as f:
    signal_model_data = pickle.load(f)
    signal_model = signal_model_data["model"]
    signal_scaler = signal_model_data["scaler"]

# Scale and predict
feature_scaled = signal_scaler.transform(feature_vector)
predicted_signal = signal_model.predict(feature_scaled)[0]
confidence = signal_model.predict_proba(feature_scaled)[0].max()
```

## Profit Potential Calculation

The `ProfitPotentialCalculator` combines model outputs to calculate:

1. **Profit Potential %**: Expected profit as percentage
2. **Risk-Reward Ratio**: Reward amount / Risk amount
3. **Profit Score**: Composite score (0-100) combining:
   - Profit potential (40%)
   - Risk-reward ratio (40%)
   - Confidence score (20%)
4. **Stop Loss Price**: Calculated based on ATR or default 5%
5. **Target Price**: From model prediction or calculated from return %

### Example

```python
from ai.profit_calculator import ProfitPotentialCalculator

calculator = ProfitPotentialCalculator()

metrics = calculator.calculate_profit_metrics(
    current_price=2500.0,
    predicted_signal="BUY",
    predicted_return_pct=8.5,
    predicted_target_price=2712.5,
    confidence_score=0.75,
    stop_loss_pct=0.05
)

print(f"Profit Potential: {metrics['profit_potential_pct']:.2f}%")
print(f"Risk-Reward: {metrics['risk_reward_ratio']:.2f}:1")
print(f"Profit Score: {metrics['profit_score']:.1f}")
```

## Signal Output Format

The AI Signal Strategy returns signals in this format:

```python
{
    "signal": "BUY",  # or "SELL" or None (HOLD)
    "entry_price": 2500.0,
    "stop_loss": 2375.0,
    "target_price": 2712.5,
    "quantity": 20,
    "reasoning": "AI Signal: BUY | Confidence: 75% | Expected Return: 8.50% | Profit Score: 72.3 | Risk-Reward: 1.70:1",
    "profit_potential_pct": 8.5,
    "profit_score": 72.3,
    "risk_reward_ratio": 1.70,
    "confidence_score": 0.75
}
```

## Building Trading Logic

You can use these outputs to build custom trading logic:

### Example 1: Filter by Profit Score
```python
if signal and signal.get("profit_score", 0) >= 60:
    # High-quality signal
    execute_trade(signal)
```

### Example 2: Filter by Risk-Reward
```python
if signal and signal.get("risk_reward_ratio", 0) >= 2.0:
    # Good risk-reward ratio
    execute_trade(signal)
```

### Example 3: Prioritize Signals
```python
from ai.profit_calculator import ProfitPotentialCalculator

signals = [signal1, signal2, signal3]
prioritized = ProfitPotentialCalculator.prioritize_signals(
    signals,
    min_profit_score=50.0,
    min_risk_reward=1.5
)

# Execute top signals first
for signal in prioritized[:5]:  # Top 5
    execute_trade(signal)
```

## Model Performance

### Evaluation Metrics

**Signal Classifier**:
- Accuracy: % of correct signal predictions
- F1-Score: Weighted average of precision and recall
- Class Distribution: Balance of BUY/SELL/HOLD predictions

**Return Predictor**:
- MAE: Mean Absolute Error (in percentage points)
- RMSE: Root Mean Squared Error
- R²: Coefficient of determination
- Directional Accuracy: % of times direction (up/down) is correct

**Target Price Predictor**:
- MAE: Mean Absolute Error (in rupees)
- RMSE: Root Mean Squared Error
- R²: Coefficient of determination

### Viewing Performance

```python
from ai.model_registry import ModelRegistryManager
import json

registry = ModelRegistryManager()
model_info = registry.get_model_info("SignalClassifier_v1")

if model_info and model_info.performance_metrics:
    metrics = json.loads(model_info.performance_metrics)
    print(f"Test Accuracy: {metrics.get('test_accuracy', 0):.2%}")
    print(f"Test F1-Score: {metrics.get('test_f1', 0):.3f}")
```

## Best Practices

1. **Train Index-Specific Models**: Different indices have different characteristics
2. **Regular Retraining**: Retrain models monthly or quarterly with new data
3. **Monitor Performance**: Track model accuracy and adjust thresholds
4. **Combine with Other Strategies**: Use AI signals alongside technical/fundamental analysis
5. **Risk Management**: Always use stop-loss and position sizing
6. **Backtest**: Test strategies on historical data before live trading

## Troubleshooting

### Models Not Found
- Ensure models are trained: `python train_signal_models.py NIFTY_50`
- Check model registry: Models should be in `models/` directory
- Verify model names match in strategy and registry
- Ensure you're viewing the correct index in the dashboard (models are index-specific)

### Low Signal Quality
- Retrain models with more data (at least 5 years recommended)
- Adjust profit score threshold in strategy
- Check feature quality (missing data, outliers)
- Ensure sufficient historical data is loaded for feature generation

### Performance Issues
- Feature engineering is computationally expensive
- Consider caching features
- Use index-specific models to reduce data size

### Duplicate Index Errors
- **Fixed in v0.3.0**: All modules now handle duplicate dates properly
- If you encounter "cannot reindex on an axis with duplicate labels":
  - Ensure you're using the latest version
  - Check database for duplicate market data entries
  - Run ETL sync to refresh data

## Next Steps

1. Train models for your target indices
2. Activate models in the model registry
3. Run the AI Signal Strategy via Strategy Engine
4. Monitor signal quality and adjust thresholds
5. Integrate with execution engine for automated trading

---

**Document Version**: 1.1  
**Last Updated**: 2025-12-14

