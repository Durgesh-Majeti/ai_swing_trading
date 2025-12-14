# Nifty 50 AI Swing Trader

A fully automated, modular trading system for Nifty 50 stocks using a hybrid approach of traditional financial analysis and modern Artificial Intelligence.

## 🎯 Core Philosophy

**Decoupled Architecture**: The system is built as a set of independent "workers" that never communicate directly with each other. Instead, they synchronize through a central **Database Hub**. This ensures that if the AI module crashes, the Execution module can still manage existing trades safely.

**AI Independence**: The system is designed to function completely independently of AI models. If no AI model is available or if the AI module fails, the system automatically falls back to technical and fundamental analysis. Strategies adjust their scoring weights dynamically, and all workflows continue to operate normally.

## 🏗️ System Architecture: The "Hub-and-Spoke" Model

The entire project revolves around a central **SQL Database** which acts as the "Source of Truth." Every other module—Data Collection, AI, Strategy, and Execution—is a spoke connected to this hub.

### Database Zones

- **Watchlist Zone**: Defines what to track (e.g., RELIANCE.NS, TCS.NS)
- **Data Zone**: Stores raw Market Data, Financial Reports, and Macro Indicators
- **Intelligence Zone**: Stores trained AI Models and their daily Predictions
- **Operations Zone**: Stores generated Trade Signals, Orders, and Portfolio status

## 📦 Module Breakdown

### A. The ETL Module (Data Ingestion)
**Role**: The "Hunter-Gatherer" - Wakes up at specific times to collect data.

**Capabilities**:
- **Market Data**: Fetches daily OHLCV data for all watchlist stocks
- **Fundamentals**: Fetches quarterly results (P/E, ROE, Debt/Equity)
- **Macro-Economics**: Monitors India VIX, Crude Oil prices, and USD/INR rates
- **Sanitization**: Cleans data (handling missing values, stock splits) before writing to database

**Location**: `engine/etl.py`

### B. The AI & Machine Learning Module (The Brain)
**Role**: The "Forecaster" - Uses historical data to predict future probability.

**Key Components**:
- **Feature Store** (`ai/feature_store.py`): Transforms raw prices into ML-ready features
- **Model Registry** (`ai/model_registry.py`): Version control for AI models
- **Inference Engine** (`ai/inference.py`): Daily prediction generator

**Location**: `ai/`

### C. The Strategy Engine (The Decision Maker)
**Role**: The "Judge" - Weighs evidence from multiple sources to make Buy/Sell decisions.

**Design - "The Registry Pattern"**:
- Strategies are "Plug-and-Play" - Drop a new strategy file into the folder, and the system automatically recognizes it
- Hybrid Logic: Combines Technical, Fundamental, and AI signals
- Output: Generates Trade Signals (Buy/Sell, Stop Loss, Target) with status "NEW"

**Location**: `strategies/`

**Available Strategies**:
- `TechnicalStrategy`: RSI + MACD + Moving Averages
- `HybridStrategy`: Combines Technical, Fundamental, and AI predictions

### D. The Execution Engine (The Trader)
**Role**: The "Gatekeeper" - Executes Strategy's signals with safety prioritization.

**Risk Management Layer**:
- Capital Check: Verifies sufficient cash
- Exposure Check: Limits sector exposure (max 20% per sector)
- Volatility Check: Blocks trades when VIX > 25
- Position Size Validation: Ensures reasonable position sizes

**Order Lifecycle**: SUBMITTED → FILLED → CLOSED
- Supports both **Paper Trading** (Simulated) and **Live Trading** modes

**Location**: `execution/`

### E. The Dashboard (The Monitor)
**Role**: The "Eyes" - Visual interface for monitoring and control.

**Features**:
- View current Portfolio and P&L
- Inspect generated Signals and AI Predictions
- Manually override or cancel signals
- Add/Remove stocks from Watchlist
- Monitor AI Models and activate/deactivate them

**Location**: `dashboard.py`

## 🔄 Daily Operational Workflow

The system functions autonomously day after day:

1. **Market Close (15:30)**: ETL Module triggers
   - Downloads today's price data for all Nifty 50 stocks
   - Updates Market Data tables
   - Fetches macro indicators (VIX, Crude, USD/INR)

2. **Evening Analysis (17:00)**:
   - **AI Engine** activates: Reads new data, processes through Feature Store, runs predictions
   - **Strategy Engine** activates: Reviews market data and AI predictions, generates "NEW" Trade Signals

3. **Pre-Market (09:00 Next Day)**: Execution Engine wakes up
   - Reads "NEW" signals
   - Applies Risk Management rules
   - Places orders (Paper or Live)

4. **Anytime**: Open the Dashboard
   - View performance graphs
   - Check active positions
   - Review AI predictions and signals

## ✅ System Status

After completing the initial run, your system will have:
- ✅ Database initialized with all tables
- ✅ 51 Nifty 50 stocks synced
- ✅ Watchlist populated
- ✅ 1 year of historical market data
- ✅ Technical indicators calculated
- ✅ Macro indicators (VIX, Crude, USD/INR) updated
- ✅ ML features generated for all stocks

**Next Steps**: Train a model and start generating predictions!

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- `uv` package manager (recommended) or `pip`

### Installation

1. **Clone or navigate to the project directory**

2. **Install dependencies using uv** (recommended):
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Quick Start - Initialize the system**:
   ```bash
   # Using uv (recommended)
   uv run python init_db.py
   uv run python -c "from engine.loaders.profile_loader import sync_nifty_companies; sync_nifty_companies()"
   uv run python -m utils.watchlist_init
   uv run python -m engine.etl
   uv run python -m ai.feature_store
   ```

   Or use the quick start script:
   ```bash
   uv run python quick_start.py
   ```

### Initial Run

After installation, run the initial setup to populate the database:

```bash
# 1. Initialize database
uv run python init_db.py

# 2. Sync Nifty 50 companies
uv run python -c "from engine.loaders.profile_loader import sync_nifty_companies; sync_nifty_companies()"

# 3. Initialize watchlist
uv run python -m utils.watchlist_init

# 4. Run ETL to fetch market data and calculate indicators
uv run python -m engine.etl

# 5. Generate ML features
uv run python -m ai.feature_store
```

This will:
- Create all database tables
- Sync 51 Nifty 50 stocks
- Add all stocks to watchlist
- Fetch 1 year of historical market data
- Calculate technical indicators (RSI, MACD, SMAs, ATR)
- Fetch macro indicators (India VIX, Crude Oil, USD/INR)
- Generate ML-ready features for all stocks

### Running the System

#### Manual Workflows

Using `uv` (recommended):
```bash
# Run ETL (data collection)
uv run python orchestrator.py etl

# Run AI Inference
uv run python orchestrator.py inference

# Run Strategy Engine
uv run python orchestrator.py strategy

# Run Execution Engine
uv run python orchestrator.py execute

# Run full workflow
uv run python orchestrator.py all
```

Or using standard Python:
```bash
python orchestrator.py etl
python orchestrator.py inference
python orchestrator.py strategy
python orchestrator.py execute
python orchestrator.py all
```

#### Automated Scheduler

```bash
# Start the automated scheduler (runs workflows at scheduled times)
uv run python orchestrator.py schedule
```

#### Dashboard

```bash
uv run streamlit run dashboard.py
```

Then open your browser to `http://localhost:8501`

### Training a Model

1. **Generate features for all stocks** (if not done during initial setup):
   ```bash
   uv run python -m ai.feature_store
   ```

2. **Train a model**:
   ```bash
   uv run python ai/train_model.py RandomForest_Swing_v1
   ```

3. **Activate the model** (when prompted, or manually):
   ```python
   from ai.model_registry import ModelRegistryManager
   registry = ModelRegistryManager()
   registry.activate_model("RandomForest_Swing_v1")
   ```

### Complete Workflow Example

After initial setup, here's a typical workflow:

```bash
# 1. Daily ETL (fetch latest market data)
uv run python -m engine.etl

# 2. Generate/update features
uv run python -m ai.feature_store

# 3. Run AI inference (requires trained model)
uv run python -m ai.inference

# 4. Generate trade signals
uv run python -m strategies.engine

# 5. Execute trades (paper trading mode)
uv run python -m execution.executor

# 6. View results in dashboard
uv run streamlit run dashboard.py
```

## 📁 Project Structure

```
.
├── ai/                      # AI & ML Module
│   ├── __init__.py
│   ├── feature_store.py     # Feature generation
│   ├── model_registry.py    # Model version control
│   ├── inference.py         # Daily predictions
│   └── train_model.py       # Model training
├── automation/              # Scheduling
│   ├── __init__.py
│   └── scheduler.py         # Daily workflow automation
├── database/                # Database models
│   ├── __init__.py
│   └── models.py            # SQLAlchemy models
├── engine/                  # ETL Module
│   ├── __init__.py
│   ├── etl.py              # Main ETL orchestrator
│   ├── ingestion.py
│   ├── runner.py
│   └── loaders/
│       ├── price_loader.py
│       └── profile_loader.py
├── execution/               # Execution Engine
│   ├── __init__.py
│   ├── executor.py         # Order execution
│   └── risk_manager.py     # Risk checks
├── strategies/              # Strategy Engine
│   ├── __init__.py
│   ├── base.py             # Base strategy class
│   ├── technical.py        # Technical analysis strategy
│   ├── hybrid.py           # Hybrid strategy
│   ├── registry.py         # Strategy discovery
│   └── engine.py           # Strategy orchestrator
├── models/                  # Saved ML models (created at runtime)
├── dashboard.py             # Streamlit dashboard
├── orchestrator.py          # Main entry point
├── init_db.py              # Database initialization
├── main.py                 # Legacy entry point
├── pyproject.toml          # Dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Database

The system uses SQLite by default (`stock_data.db`). To switch to PostgreSQL:

1. Update `DATABASE_URL` in `database/models.py`:
   ```python
   DATABASE_URL = "postgresql://user:password@localhost/trading_db"
   ```

2. Install PostgreSQL adapter:
   ```bash
   uv add psycopg2-binary
   ```

### Trading Mode

Change trading mode in `execution/executor.py`:
- `mode="PAPER"` for paper trading (simulated)
- `mode="LIVE"` for live trading (requires broker integration)

## 📊 Adding New Strategies

1. Create a new file in `strategies/` (e.g., `strategies/momentum.py`)
2. Inherit from `BaseStrategy`:
   ```python
   from strategies.base import BaseStrategy
   
   class MomentumStrategy(BaseStrategy):
       def __init__(self):
           super().__init__("Momentum")
       
       def generate_signal(self, ticker: str):
           # Your strategy logic here
           return {
               "signal": "BUY",
               "entry_price": 100.0,
               "stop_loss": 95.0,
               "target_price": 110.0,
               "quantity": 10,
               "reasoning": "Momentum breakout"
           }
   ```
3. The system will automatically discover and load it!

## 🛡️ Risk Management

The Execution Engine includes multiple risk checks:

- **Capital Check**: Verifies sufficient funds
- **Sector Exposure**: Limits exposure per sector (default: 20%)
- **Volatility Check**: Blocks trades when VIX > 25
- **Position Sizing**: Calculates safe position sizes based on stop loss distance
- **Duplicate Prevention**: Prevents multiple positions in the same stock

## 📈 Monitoring

Use the Streamlit Dashboard to:
- Monitor portfolio performance
- Review trade signals
- Inspect AI predictions
- Manage watchlist
- Control model activation

## 🔐 Security Notes

- **Paper Trading**: Always test with paper trading first
- **API Keys**: Never commit broker API keys to version control
- **Database**: Keep `stock_data.db` secure (contains trading history)

## 🐛 Troubleshooting

### Database Issues
- If tables don't exist, run: `python init_db.py`
- If data is stale, run: `python orchestrator.py etl`

### Model Issues
- If no predictions are generated, ensure:
  1. Features are generated: `python -m ai.feature_store`
  2. A model is trained and activated
  3. Sufficient historical data exists

### Strategy Issues
- Check that strategy files inherit from `BaseStrategy`
- Verify strategy is in `strategies/` directory
- Check logs for specific errors

## 📝 License

This project is for educational and research purposes. Use at your own risk.

## 📋 Changelog

All version updates and changes are documented in [CHANGELOG.md](CHANGELOG.md).

The changelog follows the [Keep a Changelog](https://keepachangelog.com/) format and includes:
- New features and enhancements
- Bug fixes and corrections
- Breaking changes
- Technical improvements
- Documentation updates

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows the decoupled architecture principle
- All modules communicate only through the database
- New strategies inherit from `BaseStrategy`
- Risk management is prioritized

## 📧 Support

For issues or questions, please check the logs first (using `loguru`). The system provides detailed logging at each step.

---

**Happy Trading! 📈🤖**

