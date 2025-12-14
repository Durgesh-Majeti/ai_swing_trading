# Changelog

All notable changes to the Nifty 50 AI Swing Trader project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Backtesting Module** (2025-12-14)
  - Complete backtesting engine for strategy performance evaluation
  - Historical data simulation with realistic trade execution
  - Performance metrics calculation (win rate, profit factor, drawdown, etc.)
  - Support for testing individual strategies or comparing all strategies
  - Backtest results storage in database
  - Dashboard integration with backtesting UI
  - Trade-by-trade analysis and visualization
  - P&L charts and performance breakdowns
  - Historical backtest results viewing

### Fixed
- **Database Migration Issue** (2025-12-14)
  - Added migration script to add missing `quantity` column to `trade_signals` table
  - Created migration utilities for future schema updates
  - Updated `init_db.py` with better table counting and migration notes

### Planned
- Live trading broker integration
- Advanced backtesting framework
- Real-time portfolio rebalancing
- Email/SMS notifications for trades
- Multi-timeframe analysis support

## [0.1.0] - 2025-12-14

### Added

#### Core System
- Initial project setup with `uv` package manager support
- Complete hub-and-spoke architecture implementation
- Central SQLite database as communication hub
- Decoupled module design for independent operation

#### Database Models (`database/models.py`)
- `CompanyProfile` table - Master company information
- `MarketData` table - OHLCV price data storage
- `FundamentalData` table - Financial metrics (P/E, ROE, Debt/Equity)
- `TechnicalIndicators` table - Calculated technical indicators
- `SentimentData` table - News and sentiment analysis (structure)
- `AIPredictions` table - AI model predictions storage
- `Watchlist` table - Active stock tracking
- `MacroIndicator` table - External economic indicators (VIX, Crude, USD/INR)
- `ModelRegistry` table - AI model version control
- `FeatureStore` table - ML-ready feature storage
- `TradeSignal` table - Strategy-generated signals
- `Order` table - Order lifecycle management
- `Portfolio` table - Current positions and P&L tracking

#### ETL Module (`engine/etl.py`)
- Market data synchronization for all watchlist stocks
- Macro indicator fetching (India VIX, Crude Oil, USD/INR)
- Technical indicator calculation (RSI, MACD, SMAs, ATR)
- Data sanitization and missing value handling
- Incremental data updates (only fetches new data)
- Company profile loader from NSE official source
- Price history loader with yfinance integration

#### AI & Machine Learning Module (`ai/`)
- **Feature Store** (`ai/feature_store.py`)
  - Raw data transformation to ML features
  - Technical feature generation (log returns, RSI, MACD, volatility)
  - Fundamental feature integration
  - Macro feature inclusion
  - Feature vector generation for model inference
  
- **Model Registry** (`ai/model_registry.py`)
  - Model version control system
  - Model activation/deactivation
  - Model metadata storage (performance metrics, training dates)
  - Model file management
  
- **Inference Engine** (`ai/inference.py`)
  - Daily prediction generation
  - Active model loading
  - Confidence score calculation
  - Prediction direction determination
  
- **Model Training** (`ai/train_model.py`)
  - Random Forest model trainer
  - Training data preparation from feature store
  - Model evaluation metrics (MAE, R²)
  - Automatic model registration

#### Strategy Engine (`strategies/`)
- **Base Strategy Class** (`strategies/base.py`)
  - Abstract base class for all strategies
  - Common utility methods (price, indicators, predictions access)
  - Position sizing calculation
  
- **Technical Strategy** (`strategies/technical.py`)
  - RSI-based oversold/overbought detection
  - MACD crossover signals
  - Moving average trend analysis
  - ATR-based stop loss calculation
  
- **Hybrid Strategy** (`strategies/hybrid.py`)
  - Multi-source signal combination
  - Weighted scoring system (Technical, Fundamental, AI, Macro)
  - Dynamic weight redistribution when AI unavailable
  - Adaptive threshold adjustment
  
- **Strategy Registry** (`strategies/registry.py`)
  - Automatic strategy discovery
  - Plug-and-play strategy loading
  - Strategy execution orchestration
  
- **Strategy Engine** (`strategies/engine.py`)
  - Daily strategy analysis runner
  - Signal generation and storage
  - Duplicate signal prevention

#### Execution Engine (`execution/`)
- **Risk Manager** (`execution/risk_manager.py`)
  - Capital availability checks
  - Sector exposure limits (20% max per sector)
  - Market volatility checks (VIX > 25 blocking)
  - Position size validation
  - Duplicate position prevention
  
- **Executor** (`execution/executor.py`)
  - Trade signal processing
  - Order lifecycle management (SUBMITTED → FILLED → CLOSED)
  - Paper trading mode (simulated execution)
  - Live trading mode structure (ready for broker integration)
  - Portfolio position tracking
  - Real-time P&L calculation
  - Price updates for existing positions

#### Dashboard (`dashboard.py`)
- **Dashboard Page**
  - Key metrics overview (Positions, Signals, P&L, VIX)
  - Recent activity display
  - Order history table
  
- **Control Center Page** (NEW)
  - System status dashboard
  - Individual workflow controls (ETL, Features, Training, Inference, Strategy, Execution)
  - Full workflow runner with progress tracking
  - Model training interface
  - Real-time status updates
  - Error handling with user-friendly messages
  
- **Portfolio Page**
  - Active positions display
  - P&L visualization with charts
  - Stop loss and target tracking
  
- **Signals Page**
  - Trade signal filtering (status, type)
  - Signal approval/rejection controls
  - Signal reasoning display
  
- **AI Predictions Page**
  - Active model information
  - Prediction history
  - Confidence score distribution charts
  
- **Watchlist Page**
  - Add/remove stocks from watchlist
  - Watchlist management interface
  
- **Models Page**
  - Model registry display
  - Model activation controls
  
- **Settings Page**
  - System information
  - Database statistics
  - Latest data timestamps

#### Automation (`automation/scheduler.py`)
- Daily workflow scheduler
- Market close workflow (15:30) - ETL data collection
- Evening analysis workflow (17:00) - AI inference + Strategy generation
- Pre-market workflow (09:00) - Signal execution
- Manual workflow triggers
- Error handling and logging

#### Utilities
- **Watchlist Initialization** (`utils/watchlist_init.py`)
  - Automatic watchlist population from company database
  
- **Quick Start Script** (`quick_start.py`)
  - One-command system initialization
  - Step-by-step setup automation
  
- **Dashboard Starter** (`start_dashboard.py`)
  - Helper script for dashboard launch

#### Documentation
- Comprehensive README.md with:
  - System architecture explanation
  - Installation instructions
  - Usage examples
  - Module breakdown
  - Configuration options
  - Troubleshooting guide
  
- Project structure documentation
- API and workflow documentation

### Changed

#### AI Independence (2025-12-14)
- **Scheduler** (`automation/scheduler.py`)
  - Made AI inference optional in evening workflow
  - Added graceful error handling for missing models
  - Workflow continues even if inference fails
  
- **Hybrid Strategy** (`strategies/hybrid.py`)
  - Dynamic weight redistribution when AI unavailable
  - Technical weight: 30% → 50% (without AI)
  - Fundamental weight: 20% → 30% (without AI)
  - Signal threshold: 60 → 50 (without AI)
  - Added reasoning notes for AI unavailability
  
- **Dashboard** (`dashboard.py`)
  - AI Inference section checks for active model
  - Button disabled when no model available
  - Warning messages instead of errors
  - Full workflow continues even if inference fails
  - Added resilience information banner

#### Database Model Fixes
- Fixed `CompanyProfile.id` AttributeError
  - Changed to use `CompanyProfile.ticker` (actual primary key)
  - Updated in Control Center and Settings pages

### Fixed

#### Bug Fixes
- **AttributeError: CompanyProfile has no attribute 'id'** (2025-12-14)
  - Fixed database queries using incorrect primary key
  - Updated all count queries to use `ticker` instead of `id`
  
- **Module import issues**
  - Fixed relative imports in utility scripts
  - Added proper module path handling

### Technical Details

#### Dependencies
- Python 3.12+
- SQLAlchemy 2.0+ for database ORM
- Streamlit for dashboard UI
- pandas, pandas-ta for data processing
- scikit-learn, xgboost for ML models
- yfinance for market data
- plotly for visualizations
- loguru for logging
- schedule for automation

#### Database
- SQLite by default (production-ready for PostgreSQL)
- All tables with proper relationships
- Indexed columns for performance
- Foreign key constraints

#### Architecture Decisions
- Decoupled modules communicate only through database
- No direct inter-module communication
- Fail-safe design (modules can crash independently)
- AI module is optional enhancement, not requirement

## Version History

- **0.1.0** (2025-12-14) - Initial release
  - Complete trading system implementation
  - All core modules functional
  - Dashboard with Control Center
  - AI independence feature
  - Full documentation

---

## How to Read This Changelog

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features that will be removed
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

## Contributing

When making changes, please update this changelog:
1. Add entries under "Unreleased" section
2. Use present tense ("Add feature" not "Added feature")
3. Group related changes
4. Include issue/PR numbers when applicable
5. Move to version section when releasing

