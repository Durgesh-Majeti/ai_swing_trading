# Historical Data Loading Guide

## Overview

The system now supports parameterized loading of historical data for any index and time period. This is essential for:
- Training AI models with sufficient historical data
- Backtesting different AI models on specific time periods
- Testing strategies on different market conditions

## Features

✅ **Parameterized Index**: Load data for any NSE index (NIFTY_50, NIFTY_100, etc.)  
✅ **Flexible Time Periods**: Load by years or specific date ranges  
✅ **Force Refresh**: Option to re-download all data  
✅ **Feature Generation**: Automatic feature calculation after data loading  

## Usage

### Method 1: Using the Utility Function

```python
from utils.load_historical_data import load_historical_data

# Load 5 years of Nifty 50 data
result = load_historical_data(
    index_name="NIFTY_50",
    years=5.0,
    force_refresh=False,
    generate_features=True
)

# Load specific date range for backtesting
result = load_historical_data(
    index_name="NIFTY_50",
    start_date="2019-01-01",
    end_date="2024-01-01",
    force_refresh=False,
    generate_features=True
)
```

### Method 2: Using ETL Module Directly

```python
from engine.etl import ETLModule

# Initialize with index name
etl = ETLModule(index_name="NIFTY_50")

# Load 5 years of data
etl.run_full_sync(years=5.0, force_refresh=False)

# Or use date range
from datetime import datetime
etl.run_full_sync(
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2024, 1, 1),
    force_refresh=False
)
```

### Method 3: Command Line

```bash
# Load 5 years of Nifty 50 data
uv run python -m utils.load_historical_data NIFTY_50 5.0

# Load with force refresh
uv run python -m utils.load_historical_data NIFTY_50 5.0 --force

# Load specific date range (requires code modification or use Python)
```

## Parameters

### `index_name` (str, required)
- Name of the index (e.g., "NIFTY_50", "NIFTY_100", "NIFTY_MIDCAP_100")
- Must match the index name in the database

### `years` (float, optional)
- Number of years of historical data to fetch
- Examples: `5.0` for 5 years, `2.5` for 2.5 years
- If not provided, defaults to 1 year
- Ignored if `start_date` and `end_date` are provided

### `start_date` (datetime/str, optional)
- Start date for data range
- Format: `datetime(2019, 1, 1)` or `"2019-01-01"`
- If provided with `end_date`, overrides `years` parameter

### `end_date` (datetime/str, optional)
- End date for data range
- Format: `datetime(2024, 1, 1)` or `"2024-01-01"`
- Defaults to today if not provided

### `force_refresh` (bool, default: False)
- If `True`, re-downloads all data even if it already exists in database
- If `False`, only downloads new data (incremental update)

### `generate_features` (bool, default: True)
- If `True`, automatically generates features after loading data
- If `False`, only loads raw market data and technical indicators

## Examples for Backtesting

### Example 1: Load Training Data (5 years)
```python
from utils.load_historical_data import load_historical_data

# Load 5 years of Nifty 50 data for model training
result = load_historical_data(
    index_name="NIFTY_50",
    years=5.0,
    force_refresh=False,
    generate_features=True
)

if result["success"]:
    print(f"Loaded {result['total_records']} records")
    print(f"Date range: {result['date_ranges']}")
```

### Example 2: Load Test Data (Specific Period)
```python
# Load data for backtesting a specific period
result = load_historical_data(
    index_name="NIFTY_50",
    start_date="2023-01-01",
    end_date="2024-01-01",
    force_refresh=False,
    generate_features=True
)
```

### Example 3: Load Different Index
```python
# Load Nifty Midcap 100 data
result = load_historical_data(
    index_name="NIFTY_MIDCAP_100",
    years=3.0,
    force_refresh=False,
    generate_features=True
)
```

### Example 4: Train-Test Split for Backtesting
```python
from datetime import datetime

# Load training data (2019-2022)
train_result = load_historical_data(
    index_name="NIFTY_50",
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2022, 12, 31),
    generate_features=True
)

# Load test data (2023-2024)
test_result = load_historical_data(
    index_name="NIFTY_50",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    generate_features=True
)
```

## Integration with Model Training

After loading historical data, you can train models:

```python
from utils.load_historical_data import load_historical_data
from ai.signal_models import SignalModelTrainer
from database.models import SessionLocal, Index
from sqlalchemy import select

# 1. Load 5 years of data
load_historical_data(
    index_name="NIFTY_50",
    years=5.0,
    generate_features=True
)

# 2. Train models for Nifty 50
session = SessionLocal()
index = session.scalar(select(Index).filter_by(name="NIFTY_50"))
trainer = SignalModelTrainer(index_id=index.id)
trainer.train_all_models(index_suffix="NIFTY_50")
trainer.close()
session.close()
```

## Integration with Backtesting

```python
from utils.load_historical_data import load_historical_data
from backtesting.engine import BacktestEngine

# Load historical data for backtesting period
load_historical_data(
    index_name="NIFTY_50",
    start_date="2023-01-01",
    end_date="2024-01-01",
    generate_features=True
)

# Run backtest
engine = BacktestEngine()
results = engine.run_backtest(
    strategy_name="AI_Signal_Strategy",
    ticker="RELIANCE",
    start_date="2023-01-01",
    end_date="2024-01-01"
)
```

## Available Indices

Common indices you can use:
- `NIFTY_50` - Nifty 50
- `NIFTY_100` - Nifty 100
- `NIFTY_200` - Nifty 200
- `NIFTY_500` - Nifty 500
- `NIFTY_MIDCAP_100` - Nifty Midcap 100
- `NIFTY_SMALLCAP_100` - Nifty Smallcap 100
- `NIFTY_BANK` - Nifty Bank
- `NIFTY_IT` - Nifty IT
- And many more...

See `engine/loaders/nse_index_discovery.py` for the complete list.

## Best Practices

1. **Training Data**: Load 5+ years for robust model training
2. **Test Data**: Use recent 1-2 years for validation
3. **Backtesting**: Use specific date ranges to test different market conditions
4. **Incremental Updates**: Use `force_refresh=False` for daily updates
5. **Feature Generation**: Always generate features after loading new data

## Troubleshooting

### Index Not Found
```
Error: Index NIFTY_50 not found
```
**Solution**: Ensure the index exists in the database. Run index discovery/sync first.

### Insufficient Data
```
Warning: No data for TICKER
```
**Solution**: Check if the ticker symbol is correct and data is available on Yahoo Finance.

### Date Range Issues
```
Error: start_date must be before end_date
```
**Solution**: Ensure start_date < end_date

## Performance Notes

- Loading 5 years of data for Nifty 50 (~50 companies) takes approximately 5-10 minutes
- Feature generation adds additional time (~2-5 minutes)
- Use `force_refresh=False` for faster incremental updates
- Consider loading data in batches for large indices (Nifty 500+)

---

**Document Version**: 1.0  
**Last Updated**: 2024

