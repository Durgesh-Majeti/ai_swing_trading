# Indian Stock Analysis - Multi-Index AI Swing Trader

A fully automated, modular trading system for Indian stocks across multiple NSE indices (Nifty 50, Nifty 100, Nifty 500, Sectoral indices, and more) using a hybrid approach of traditional financial analysis and modern Artificial Intelligence.

## 🎯 Core Philosophy

**Decoupled Architecture**: The system is built as a set of independent "workers" that never communicate directly with each other. Instead, they synchronize through a central **Database Hub**. This ensures that if the AI module crashes, the Execution module can still manage existing trades safely.

**AI Independence**: The system is designed to function completely independently of AI models. If no AI model is available or if the AI module fails, the system automatically falls back to technical and fundamental analysis. Strategies adjust their scoring weights dynamically, and all workflows continue to operate normally.

## 🏗️ System Architecture: The "Hub-and-Spoke" Model

The entire project revolves around a central **SQL Database** which acts as the "Source of Truth." Every other module—Data Collection, AI, Strategy, and Execution—is a spoke connected to this hub.

### Database Zones

- **Index Zone**: Manages multiple stock indices (Nifty 50, Nifty 100, Sectoral, etc.)
- **Watchlist Zone**: Index-specific watchlists defining what to track (e.g., RELIANCE.NS, TCS.NS)
- **Data Zone**: Stores raw Market Data, Financial Reports, and Macro Indicators
- **Intelligence Zone**: Stores trained AI Models and their daily Predictions
- **Operations Zone**: Stores generated Trade Signals, Orders, and Portfolio status
- **Strategy Zone**: Index-specific strategy documentation and metadata

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
- **Index-Specific Strategies**: Each strategy can be configured for specific indices (Nifty 50, Nifty 100, Sectoral, etc.)
- Hybrid Logic: Combines Technical, Fundamental, and AI signals
- Output: Generates Trade Signals (Buy/Sell, Stop Loss, Target) with status "NEW"

**Location**: `strategies/`

**Available Strategies**:
- `TechnicalStrategy`: RSI + MACD + Moving Averages
- `HybridStrategy`: Combines Technical, Fundamental, and AI predictions

**Strategy Documentation**: Each strategy can have detailed documentation including:
- How it works
- Entry/Exit conditions
- Risk management approach
- Recommended timeframe
- Risk level assessment

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
- **Index Management**: Create and sync companies for all NSE indices (54+ indices supported)
- **Index Selector**: Filter all views by selected index
- **Strategies Page**: View and manage index-specific strategies with detailed documentation
- View current Portfolio and P&L
- Inspect generated Signals and AI Predictions
- Manually override or cancel signals
- Add/Remove stocks from Index-specific Watchlists
- Monitor AI Models and activate/deactivate them
- **Backtesting**: Test strategies on historical data with comprehensive performance metrics

**Location**: `dashboard.py`

## 🔄 Daily Operational Workflow

The system functions autonomously day after day:

1. **Market Close (15:30)**: ETL Module triggers
   - Downloads today's price data for all watchlist stocks (across all indices)
   - Updates Market Data tables
   - Fetches macro indicators (VIX, Crude, USD/INR)

2. **Evening Analysis (17:00)**:
   - **AI Engine** activates: Reads new data, processes through Feature Store, runs predictions
   - **Strategy Engine** activates: Runs index-specific strategies, reviews market data and AI predictions, generates "NEW" Trade Signals

3. **Pre-Market (09:00 Next Day)**: Execution Engine wakes up
   - Reads "NEW" signals (filtered by active index if specified)
   - Applies Risk Management rules
   - Places orders (Paper or Live)

4. **Anytime**: Open the Dashboard
   - View performance graphs
   - Check active positions
   - Review AI predictions and signals

## ✅ System Status

After completing the initial run, your system will have:
- ✅ Database initialized with all tables
- ✅ 54+ NSE indices available (Nifty 50, 100, 500, Sectoral, Thematic, Strategy indices)
- ✅ Companies synced for selected indices
- ✅ Index-specific watchlists populated
- ✅ 1 year of historical market data
- ✅ Technical indicators calculated
- ✅ Macro indicators (VIX, Crude, USD/INR) updated
- ✅ ML features generated for all stocks
- ✅ Index-specific strategies configured

**Next Steps**: 
1. Create all indices: Use "Create All NSE Indices" in dashboard or run `uv run python -m utils.discover_nse_indices`
2. Sync companies for your target indices
3. Create index-specific strategies
4. Train a model and start generating predictions!

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
   
   # Create all NSE indices (54+ indices)
   uv run python -m utils.discover_nse_indices
   
   # Sync companies for specific index (e.g., NIFTY_50)
   uv run python -c "from engine.loaders.profile_loader import sync_index_companies; sync_index_companies('NIFTY_50')"
   
   # Or sync all available indices
   uv run python -m utils.sync_all_indices
   
   # Initialize watchlist for an index
   uv run python -m utils.watchlist_init NIFTY_50
   
   # Run ETL and generate features
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

# 2. Create all NSE indices (54+ indices)
uv run python -m utils.discover_nse_indices

# 3. Sync companies for indices (example: NIFTY_50)
uv run python -c "from engine.loaders.profile_loader import sync_index_companies; sync_index_companies('NIFTY_50')"

# Or sync all available indices at once
uv run python -m utils.sync_all_indices

# 4. Initialize watchlist for an index
uv run python -m utils.watchlist_init NIFTY_50

# 5. Run ETL to fetch market data and calculate indicators
uv run python -m engine.etl

# 6. Generate ML features
uv run python -m ai.feature_store
```

This will:
- Create all database tables
- Create 54+ NSE indices in database
- Sync companies for selected indices (28 confirmed available on NSE)
- Add stocks to index-specific watchlists
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
│       ├── profile_loader.py  # Company and index sync
│       └── nse_index_discovery.py  # NSE index discovery
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
│   └── engine.py           # Strategy orchestrator (index-aware)
├── backtesting/            # Backtesting Module
│   ├── __init__.py
│   ├── engine.py           # Backtesting engine
│   ├── models.py           # Backtest result models
│   └── runner.py           # CLI for backtesting
├── migrations/             # Database migrations
│   ├── __init__.py
│   ├── add_index_support.py
│   ├── add_quantity_to_trade_signals.py
│   └── add_missing_trade_signal_columns.py
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── watchlist_init.py   # Watchlist initialization
│   ├── sync_all_indices.py # Index sync utility
│   └── discover_nse_indices.py  # Index discovery
├── migrations/             # Database migrations
│   ├── __init__.py
│   ├── add_index_support.py
│   ├── add_quantity_to_trade_signals.py
│   └── add_missing_trade_signal_columns.py
├── models/                  # Saved ML models (created at runtime)
├── dashboard.py             # Streamlit dashboard
├── orchestrator.py          # Main entry point
├── init_db.py              # Database initialization
├── quick_start.py          # Quick setup script
├── start_dashboard.py      # Dashboard starter
├── main.py                 # Legacy entry point
├── pyproject.toml          # Dependencies
├── CHANGELOG.md            # Version history
└── README.md               # This file
```

## 🗄️ Database Structure

The system uses a **central SQLite database** (`stock_data.db`) as the "Source of Truth" following the Hub-and-Spoke architecture. All modules communicate through this database, ensuring complete decoupling.

### Database Zones

The database is organized into logical zones:

- **Index Zone**: Stock indices and company-index relationships
- **Watchlist Zone**: Index-specific active stock tracking
- **Data Zone**: Raw market data, fundamentals, technical indicators, macro indicators
- **Intelligence Zone**: AI models, predictions, and ML features
- **Operations Zone**: Trade signals, orders, portfolio positions, backtest results
- **Strategy Zone**: Index-specific strategy documentation and metadata

### Complete Table Schema

#### **Index Zone**

**`indices`** - Stock indices (Nifty 50, Nifty 100, etc.)
- `id` (PK, Integer): Unique identifier
- `name` (String, Unique, Indexed): Internal name (e.g., "NIFTY_50", "NIFTY_100")
- `display_name` (String): Human-readable name (e.g., "Nifty 50")
- `description` (Text): Index description
- `is_active` (Boolean): Whether index is active
- `created_at` (DateTime): Creation timestamp

**`company_index_mapping`** - Many-to-many relationship between companies and indices
- `ticker` (PK, FK → `company_profiles.ticker`): Company ticker
- `index_id` (PK, FK → `indices.id`): Index identifier

#### **Master Data Zone**

**`company_profiles`** - Master company information (The Hub)
- `ticker` (PK, String, Indexed): Stock ticker (e.g., "RELIANCE.NS")
- `name` (String): Company name
- `sector` (String): Business sector
- `industry` (String): Industry classification
- `exchange` (String): Stock exchange
- `currency` (String): Trading currency
- `description` (Text): Company description

**Relationships:**
- One-to-many: `market_data`, `fundamental_data`, `sentiment_data`, `ai_predictions`, `watchlist`, `orders`, `portfolio`
- Many-to-many: `indices` (via `company_index_mapping`)

#### **Data Zone**

**`market_data`** - Historical OHLCV price data
- `id` (PK, Integer, Indexed)
- `ticker` (FK → `company_profiles.ticker`, Indexed)
- `date` (DateTime, Indexed): Trading date
- `open` (Float): Opening price
- `high` (Float): High price
- `low` (Float): Low price
- `close` (Float): Closing price
- `volume` (Integer): Trading volume

**`fundamental_data`** - Financial metrics and fundamentals
- `id` (PK, Integer)
- `ticker` (FK → `company_profiles.ticker`)
- `report_date` (Date): Financial report date
- `pe_ratio` (Float): Price-to-Earnings ratio
- `pb_ratio` (Float): Price-to-Book ratio
- `market_cap` (Float): Market capitalization
- `roe` (Float): Return on Equity
- `eps` (Float): Earnings per Share
- `revenue_growth` (Float): Revenue growth percentage
- `debt_to_equity` (Float): Debt-to-Equity ratio

**`technical_indicators`** - Calculated technical indicators
- `id` (PK, Integer)
- `ticker` (FK → `company_profiles.ticker`)
- `date` (DateTime): Calculation date
- `rsi_14` (Float): Relative Strength Index (14-period)
- `macd` (Float): MACD line
- `macd_signal` (Float): MACD signal line
- `sma_50` (Float): 50-day Simple Moving Average
- `sma_200` (Float): 200-day Simple Moving Average
- `beta` (Float): Beta coefficient
- `atr` (Float): Average True Range

**`sentiment_data`** - News and sentiment analysis
- `id` (PK, Integer)
- `ticker` (FK → `company_profiles.ticker`)
- `date` (DateTime): Sentiment date
- `source` (String): Data source
- `headline` (Text): News headline
- `sentiment_score` (Float): Sentiment score (-1 to 1)
- `magnitude` (Float): Sentiment magnitude

**`macro_indicators`** - External economic indicators
- `id` (PK, Integer)
- `date` (DateTime, Indexed): Indicator date
- `indicator_name` (String, Indexed): Name (e.g., "INDIA_VIX", "CRUDE_OIL", "USD_INR")
- `value` (Float): Indicator value
- `unit` (String): Unit of measurement

#### **Watchlist Zone**

**`watchlist`** - Index-specific active stock tracking
- `id` (PK, Integer)
- `ticker` (FK → `company_profiles.ticker`, Indexed)
- `index_id` (FK → `indices.id`, Indexed): **Index isolation**
- `added_at` (DateTime): When stock was added
- `is_active` (Boolean): Whether entry is active
- `notes` (Text): User notes

**Note:** Same ticker can appear in multiple index watchlists.

#### **Intelligence Zone**

**`model_registry`** - AI model version control
- `id` (PK, Integer)
- `model_name` (String, Unique, Indexed): Model identifier
- `version` (String): Model version
- `model_type` (String): Type (e.g., "RandomForest", "XGBoost", "LSTM")
- `file_path` (String): Path to saved model file
- `is_active` (Boolean): Whether model is currently active
- `created_at` (DateTime): Creation timestamp
- `trained_on_date` (Date): Training date
- `performance_metrics` (JSON): Model performance metrics
- `description` (Text): Model description

**`feature_store`** - ML-ready features for model training/inference
- `id` (PK, Integer)
- `ticker` (FK → `company_profiles.ticker`, Indexed)
- `date` (DateTime, Indexed): Feature date

**Technical Features:**
- `log_return` (Float): Logarithmic return
- `rsi` (Float): RSI value
- `macd` (Float): MACD value
- `sma_50` (Float): 50-day SMA
- `sma_200` (Float): 200-day SMA
- `atr` (Float): Average True Range
- `volatility` (Float): Price volatility
- `price_momentum` (Float): Price momentum
- `volume_trend` (Float): Volume trend

**Fundamental Features:**
- `pe_ratio` (Float): P/E ratio
- `roe` (Float): Return on Equity
- `debt_to_equity` (Float): Debt-to-Equity ratio

**Macro Features:**
- `vix` (Float): India VIX value
- `crude_oil` (Float): Crude oil price
- `usd_inr` (Float): USD/INR exchange rate

**`ai_predictions`** - AI model price predictions
- `id` (PK, Integer)
- `ticker` (FK → `company_profiles.ticker`)
- `index_id` (FK → `indices.id`, Nullable, Indexed): **Index isolation**
- `generated_at` (DateTime): Prediction timestamp
- `model_name` (String): Model that generated prediction
- `target_date` (Date): Target prediction date
- `predicted_price` (Float): Predicted price
- `confidence_score` (Float): Prediction confidence (0-1)
- `direction` (String): Predicted direction ("UP", "DOWN", "NEUTRAL")

#### **Operations Zone**

**`trade_signals`** - Strategy-generated trade signals
- `id` (PK, Integer)
- `created_at` (DateTime, Indexed): Signal creation time
- `ticker` (FK → `company_profiles.ticker`, Indexed)
- `index_id` (FK → `indices.id`, Nullable, Indexed): **Index isolation**
- `strategy_name` (String): Strategy that generated signal
- `signal` (String): Signal type ("BUY", "SELL", "HOLD")
- `status` (String): Signal status ("NEW", "PROCESSED", "REJECTED", "CANCELLED")
- `entry_price` (Float): Recommended entry price
- `stop_loss` (Float): Stop loss price
- `target_price` (Float): Target price
- `quantity` (Integer): Recommended quantity
- `reasoning` (Text): Signal reasoning/explanation
- `priority` (Integer): Signal priority (1-10, higher = more important)

**`orders`** - Order lifecycle management
- `id` (PK, Integer)
- `signal_id` (FK → `trade_signals.id`, Nullable): Source signal
- `ticker` (FK → `company_profiles.ticker`, Indexed)
- `index_id` (FK → `indices.id`, Nullable, Indexed): **Index isolation**
- `order_type` (String): Order type ("MARKET", "LIMIT", "SL", "SL-M")
- `side` (String): Order side ("BUY", "SELL")
- `quantity` (Integer): Order quantity
- `price` (Float): Limit/trigger price
- `status` (String): Order status ("SUBMITTED", "FILLED", "REJECTED", "CANCELLED", "CLOSED")
- `created_at` (DateTime, Indexed): Order creation time
- `filled_at` (DateTime, Nullable): Fill timestamp
- `filled_price` (Float, Nullable): Actual fill price
- `stop_loss` (Float, Nullable): Stop loss price
- `target_price` (Float, Nullable): Target price
- `mode` (String): Trading mode ("PAPER", "LIVE")
- `broker_order_id` (String, Nullable): Broker order ID (for live trading)

**`portfolio`** - Current positions and P&L tracking
- `id` (PK, Integer)
- `ticker` (FK → `company_profiles.ticker`, Indexed)
- `index_id` (FK → `indices.id`, Nullable, Indexed): **Index isolation**
- `quantity` (Integer): Current position quantity
- `avg_entry_price` (Float): Average entry price
- `current_price` (Float): Current market price
- `last_updated` (DateTime): Last price update time
- `unrealized_pnl` (Float): Unrealized profit/loss
- `realized_pnl` (Float): Realized profit/loss
- `stop_loss` (Float, Nullable): Stop loss price
- `target_price` (Float, Nullable): Target price
- `entry_date` (DateTime): Position entry date
- `order_id` (FK → `orders.id`, Nullable): Source order

**Note:** Same ticker can have positions in different indices (isolated by `index_id`).

**`backtest_runs`** - Backtest execution metadata
- `id` (PK, Integer)
- `created_at` (DateTime): Backtest creation time
- `strategy_name` (String, Indexed): Strategy tested
- `ticker` (FK → `company_profiles.ticker`, Indexed): Stock tested
- `start_date` (DateTime): Backtest start date
- `end_date` (DateTime): Backtest end date
- `initial_capital` (Float): Starting capital
- `position_size_pct` (Float): Position size percentage
- `total_trades` (Integer): Total trades executed
- `winning_trades` (Integer): Number of winning trades
- `losing_trades` (Integer): Number of losing trades
- `net_profit` (Float): Net profit/loss
- `total_profit` (Float): Total profit from winning trades
- `total_loss` (Float): Total loss from losing trades
- `win_rate` (Float): Win rate percentage
- `profit_factor` (Float): Profit factor (profit/loss ratio)
- `max_drawdown` (Float): Maximum drawdown percentage
- `avg_win` (Float): Average win amount
- `avg_loss` (Float): Average loss amount
- `final_capital` (Float): Final capital after backtest
- `notes` (Text): Backtest notes

**`backtest_trades`** - Individual trades from backtest
- `id` (PK, Integer)
- `backtest_run_id` (FK → `backtest_runs.id`, Indexed): Parent backtest
- `entry_date` (DateTime): Trade entry date
- `exit_date` (DateTime): Trade exit date
- `entry_price` (Float): Entry price
- `exit_price` (Float): Exit price
- `quantity` (Integer): Trade quantity
- `side` (String): Trade side ("BUY", "SELL")
- `pnl` (Float): Profit/loss for this trade
- `pnl_pct` (Float): Profit/loss percentage
- `exit_reason` (String): Exit reason ("STOP_LOSS", "TARGET", "END_DATE")

#### **Strategy Zone**

**`strategy_metadata`** - Index-specific strategy documentation
- `id` (PK, Integer)
- `strategy_name` (String, Indexed): Strategy identifier (must match class name)
- `index_id` (FK → `indices.id`, Indexed): **Index-specific strategy**
- `display_name` (String): Human-readable strategy name
- `description` (Text): Strategy description
- `category` (String): Strategy category ("Technical", "Fundamental", "Hybrid", "AI-Based", etc.)
- `parameters` (JSON, Nullable): Strategy-specific parameters
- `recommended_timeframe` (String): Recommended timeframe ("Day Trading", "Swing Trading", etc.)
- `risk_level` (String): Risk level ("Low", "Medium", "High")
- `how_it_works` (Text): How the strategy works
- `entry_conditions` (Text): Entry conditions
- `exit_conditions` (Text): Exit conditions
- `risk_management` (Text): Risk management approach
- `created_at` (DateTime): Creation timestamp
- `updated_at` (DateTime): Last update timestamp
- `is_active` (Boolean): Whether strategy is active
- `author` (String, Nullable): Strategy author
- `version` (String): Strategy version

**Note:** Same strategy can have different documentation per index.

### Index Isolation

**Complete data isolation** is achieved through `index_id` foreign keys in operational tables:

- **`portfolio`**: Positions are isolated per index (same ticker can have different positions in different indices)
- **`trade_signals`**: Signals are isolated per index
- **`orders`**: Orders are isolated per index
- **`ai_predictions`**: Predictions are isolated per index
- **`watchlist`**: Watchlists are index-specific
- **`strategy_metadata`**: Strategy documentation is index-specific

This ensures that when you select an index in the dashboard, **all data shown is completely isolated** to that index only.

### Relationships Diagram

```
indices (1) ──< (M) company_index_mapping (M) >── (1) company_profiles
   │                                                      │
   │                                                      ├──< (M) market_data
   │                                                      ├──< (M) fundamental_data
   │                                                      ├──< (M) sentiment_data
   │                                                      ├──< (M) feature_store
   │                                                      ├──< (M) ai_predictions
   │                                                      ├──< (M) watchlist
   │                                                      ├──< (M) orders
   │                                                      └──< (M) portfolio
   │
   ├──< (M) watchlist
   ├──< (M) strategy_metadata
   ├──< (M) ai_predictions (via index_id)
   ├──< (M) trade_signals (via index_id)
   ├──< (M) orders (via index_id)
   └──< (M) portfolio (via index_id)

trade_signals (1) ──< (M) orders
orders (1) ──< (M) portfolio
backtest_runs (1) ──< (M) backtest_trades
```

### Database Indexes

For performance optimization, the following columns are indexed:

- **Primary Keys**: All `id` columns
- **Foreign Keys**: All foreign key columns (`ticker`, `index_id`, `signal_id`, etc.)
- **Frequently Queried**: `date`, `created_at`, `status`, `is_active`
- **Lookup Fields**: `name`, `model_name`, `strategy_name`, `indicator_name`

### Database Migrations

The system includes a migration framework for schema updates:

- `migrations/add_index_support.py` - Adds index support (indices table, company_index_mapping, index_id columns)
- `migrations/add_index_isolation.py` - Adds index_id to operational tables for complete isolation
- `migrations/add_quantity_to_trade_signals.py` - Adds quantity column to trade_signals
- `migrations/add_missing_trade_signal_columns.py` - Adds priority and reasoning columns
- `migrations/add_strategy_metadata_table.py` - Creates strategy_metadata table
- `migrations/check_schema.py` - Utility to check schema differences

**Running Migrations:**
```bash
# Run a specific migration
uv run python -m migrations.add_index_isolation

# Check for schema differences
uv run python -m migrations.check_schema
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

**Note:** When switching databases, ensure all migrations are run to create the schema.

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
4. **Add Strategy Documentation**: Use the "Strategies" page in the dashboard to add detailed documentation for your strategy, including:
   - How it works
   - Entry/Exit conditions
   - Risk management approach
   - Recommended timeframe
   - Assign it to specific indices

## 📈 Multi-Index Support

The system supports **54+ NSE indices** including:

### Benchmark Indices
- Nifty 50, 100, 200, 500
- Nifty Total Market

### Market Cap Based
- Nifty LargeMidcap 250
- Nifty Midcap 50, 100, 150, 250
- Nifty Smallcap 50, 100, 250
- Nifty Microcap 250

### Sectoral Indices
- Nifty Auto, Bank, Energy, FMCG, Healthcare, IT, Media, Metal, Pharma
- Nifty PSU Bank, Private Bank, Realty
- Nifty Consumer Durables, Oil & Gas, Infrastructure, Commodities

### Thematic Indices
- Nifty CPSE, MNC, Next 50, PSE
- Nifty India Consumption, Digital, Manufacturing
- Nifty Quality 30, Shariah 25
- Nifty Tata Group, Mahindra Group, Aditya Birla Group

### Strategy Indices
- Nifty Alpha 50, High Beta 50, Low Volatility 50, Momentum 50
- Nifty Quality Low Volatility 30
- Nifty 50/100/500 Equal Weight

### Managing Indices

**Create All Indices**:
```bash
uv run python -m utils.discover_nse_indices
```

**Sync Companies for Indices**:
```bash
# Sync single index
uv run python -c "from engine.loaders.profile_loader import sync_index_companies; sync_index_companies('NIFTY_50')"

# Sync all available indices
uv run python -m utils.sync_all_indices
```

**Via Dashboard**:
- Go to Control Center → Index Management
- Click "Create All NSE Indices" to create all 54+ indices
- Select indices and click "Sync Selected Indices" to fetch companies
- Use "Sync All Indices" to sync all available indices at once

### Index-Specific Strategies

Strategies can be configured for specific indices:
- Each strategy can have different parameters per index
- Strategy documentation is index-specific
- Strategy Engine can run strategies for selected index only
- Watchlists are index-specific

## 🛡️ Risk Management

The Execution Engine includes multiple risk checks:

- **Capital Check**: Verifies sufficient funds
- **Sector Exposure**: Limits exposure per sector (default: 20%)
- **Volatility Check**: Blocks trades when VIX > 25
- **Position Sizing**: Calculates safe position sizes based on stop loss distance
- **Duplicate Prevention**: Prevents multiple positions in the same stock

## 📈 Monitoring

Use the Streamlit Dashboard to:
- **Manage Indices**: Create and sync companies for all NSE indices
- **View Strategies**: Browse index-specific strategies with detailed documentation
- Monitor portfolio performance
- Review trade signals (filtered by index)
- Inspect AI predictions
- Manage index-specific watchlists
- Control model activation
- **Backtest Strategies**: Test strategies on historical data with comprehensive metrics

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

