# Changelog

All notable changes to the Indian Stock Analysis - Multi-Index AI Swing Trader project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Live trading broker integration
- Advanced backtesting framework with walk-forward analysis
- Real-time portfolio rebalancing
- Email/SMS notifications for trades
- Multi-timeframe analysis support
- Index performance comparison tools
- Sector rotation strategies

## [0.3.0] - 2025-12-14

### Added
- **AI Signal Generation Models** (2025-12-14)
  - Signal Classifier model for BUY/SELL/HOLD predictions
  - Return Predictor model for expected percentage return
  - Target Price Predictor model for future price targets
  - Index-specific model training (e.g., SignalClassifier_NIFTY50_v1)
  - Composite AI Signal Strategy that uses all three models together
  - Model training script (`train_signal_models.py`) with index support
  - Profit potential calculator for risk-reward analysis
  - Enhanced feature engineering with 2490+ features per sample

- **Index-Specific Model Management** (2025-12-14)
  - Added `index_id` to ModelRegistry table for model isolation
  - Models are now filtered by index in dashboard
  - Migration script to populate index_id for existing models
  - Models only visible for their specific index

- **Historical Data Loading Utility** (2025-12-14)
  - Parameterized data loading with `years`, `start_date`, `end_date` support
  - Index-specific data loading (`utils/load_historical_data.py`)
  - Convenience function for Nifty 50 5-year data loading
  - Supports flexible backtesting data preparation

- **Enhanced Feature Engineering** (2025-12-14)
  - Advanced feature calculation with lag features, rolling statistics
  - Cross-asset features and pattern detection
  - Support for sequence features for time-series models
  - Comprehensive feature set (2490 features per sample)

### Fixed
- **Duplicate Index Error Fixes** (2025-12-14)
  - Fixed "cannot reindex on an axis with duplicate labels" error in all modules
  - Added duplicate date handling in ETL market data sync
  - Fixed duplicate handling in technical indicators calculation
  - Fixed duplicate handling in feature store generation
  - Fixed duplicate handling in enhanced features for AI strategies
  - Fixed duplicate handling in backtesting engine
  - All modules now properly handle duplicate dates from database

- **AI Signal Strategy Fixes** (2025-12-14)
  - Fixed Bollinger Bands unpacking error (returns DataFrame, not tuple)
  - Fixed XGBoost API compatibility (moved early_stopping_rounds to constructor)
  - Added LabelEncoder for string to numeric label conversion
  - Fixed feature dimension validation
  - Enhanced error logging and diagnostics

- **Backtesting Improvements** (2025-12-14)
  - Fixed backtesting engine to pass index information to strategies
  - Updated backtesting to use strategy's generate_signal method
  - Fixed AI Signal Strategy to work in backtesting context
  - Added proper error handling for feature generation during backtesting

- **ETL Module Fixes** (2025-12-14)
  - Fixed ambiguous truth value error in market data sync
  - Added multi-index column handling for yfinance data
  - Fixed Series to scalar conversion for price/volume values
  - Improved date comparison logic

### Changed
- **Model Registry** (2025-12-14)
  - `register_model()` now accepts `index_id` parameter
  - Models are stored with index association
  - Dashboard Models page filters by selected index

- **AI Signal Strategy** (2025-12-14)
  - Automatically uses all three models (SignalClassifier, ReturnPredictor, TargetPricePredictor)
  - No individual model activation needed - composite strategy approach
  - Loads index-specific models automatically
  - Falls back to generic models if index-specific not available

- **Dashboard Updates** (2025-12-14)
  - Models page now shows index-specific models only
  - Added note about composite strategy not needing activation
  - Enhanced ETL page with individual data zone refresh controls

## [0.2.0] - 2025-12-14

### Added
- **Complete Index Isolation** (2025-12-14)
  - Added `index_id` to Portfolio, TradeSignal, Order, AIPredictions tables
  - Complete data isolation between indices
  - Index initialization/reset utility (`utils/index_reset.py`)
  - All operational data now properly isolated per index
  - Migration script for index isolation (`migrations/add_index_isolation.py`)

- **Workflow Synchronization with Index** (2025-12-14)
  - ETL Module now accepts `index_id` and filters by index
  - Feature Store now accepts `index_id` and filters by index
  - AI Inference now accepts `index_id` and filters by index
  - Execution Engine now filters signals by `index_id`
  - All workflow controls in dashboard synced with selected index
  - Full workflow runner respects selected index

- **Multi-Index Support** (2025-12-14)
  - Complete support for 54+ NSE indices (Nifty 50, 100, 500, Sectoral, Thematic, Strategy indices)
  - Index model in database with company-index relationships
  - Index-specific watchlists (stocks can belong to multiple indices)
  - Index-specific strategies (strategies can be configured per index)
  - Index selector in dashboard sidebar (filters all views by selected index)
  - Index management section in Control Center
  - Comprehensive NSE index discovery utility (`utils/discover_nse_indices.py`)
  - Index sync functionality for all available indices
  - Support for 28 confirmed available indices on NSE

- **Index Management Features** (2025-12-14)
  - "Create All NSE Indices" button in dashboard (creates 54+ indices)
  - Multi-select index sync (sync companies for selected indices)
  - "Sync All Indices" bulk operation
  - Index-specific company assignment
  - Progress tracking for index sync operations
  - Results display showing new companies and assignments per index

- **Strategies Page** (2025-12-14)
  - New "Strategies" page in dashboard
  - View all available strategies with detailed documentation
  - Add/Edit strategy documentation including:
    - How it works
    - Entry/Exit conditions
    - Risk management approach
    - Recommended timeframe
    - Risk level assessment
    - Index assignment
  - Auto-populate section for discovered strategies without documentation
  - Index-filtered strategy view

- **NSE Index Discovery** (2025-12-14)
  - Comprehensive index list with 54+ indices
  - Automatic URL generation for NSE CSV files
  - Index availability checking (tests which indices have CSV files on NSE)
  - Support for Benchmark, Market Cap, Sectoral, Thematic, and Strategy indices
  - Dynamic URL generation for unknown index patterns

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
  - Comprehensive metrics explanation in dashboard

### Changed
- **Profile Loader** (`engine/loaders/profile_loader.py`)
  - Renamed `sync_nifty_companies()` to `sync_index_companies()` for multi-index support
  - Added comprehensive index URL mapping (54+ indices)
  - Added `sync_all_indices()` function for bulk operations
  - Dynamic URL generation for indices not in mapping
  - Improved error handling and logging

- **Watchlist System** (`utils/watchlist_init.py`)
  - Updated to support index-specific watchlists
  - `initialize_watchlist()` now accepts `index_name` parameter
  - Watchlists are now per-index (same stock can be in multiple index watchlists)

- **Strategy Engine** (`strategies/engine.py`)
  - Added index filtering support
  - Can run strategies for specific index only
  - Filters watchlist and strategies by selected index
  - Accepts optional `index_name` parameter

- **Dashboard** (`dashboard.py`)
  - Added index selector in sidebar (filters all pages)
  - Updated Watchlist page to show index-specific watchlists
  - Added "Strategies" page with strategy documentation management
  - Added "Index Management" section in Control Center
  - Updated all pages to respect selected index filter
  - Strategy Engine controls now use selected index

- **Database Models** (`database/models.py`)
  - Added `Index` model for stock indices
  - Added `company_index_mapping` junction table (many-to-many relationship)
  - Updated `Watchlist` to include `index_id` (index-specific watchlists)
  - Updated `StrategyMetadata` to include `index_id` (index-specific strategies)
  - Added relationships between companies and indices

### Fixed
- **Backtesting Capital Tracking Bug** (2025-12-14)
  - Fixed incorrect net profit calculation in backtesting engine
  - Capital was incorrectly deducted when opening positions
  - Now properly tracks `capital_used` and returns it along with P&L on position close
  - Net profit now correctly reflects actual trading performance

- **Database Migration Issues** (2025-12-14)
  - Added migration script to add missing `quantity` column to `trade_signals` table
  - Added migration script to add `priority` and `reasoning` columns
  - Created migration for index support (`migrations/add_index_support.py`)
  - Created migration utilities for future schema updates
  - Updated `init_db.py` with better table counting and migration notes

### Technical Details

#### Multi-Index Architecture
- Companies can belong to multiple indices (many-to-many relationship)
- Watchlists are index-specific (same stock can be tracked in multiple index watchlists)
- Strategies can be configured per index with different parameters
- Strategy Engine can filter by index for targeted analysis
- All dashboard views respect index selection

#### Index Sync System
- Automatic discovery of available NSE indices
- URL pattern matching for NSE CSV files
- Availability checking (tests which indices have CSV files)
- Bulk sync operations with progress tracking
- Error handling for unavailable indices

#### Supported Indices
- **28 confirmed available** indices with CSV files on NSE
- **54+ total indices** defined in system (some may have different URL patterns)
- Categories: Benchmark, Market Cap, Sectoral, Thematic, Strategy
- Easy to add more indices by updating `nse_index_discovery.py`

### Planned
- Live trading broker integration
- Advanced backtesting framework with walk-forward analysis
- Real-time portfolio rebalancing
- Email/SMS notifications for trades
- Multi-timeframe analysis support
- Index performance comparison tools
- Sector rotation strategies

## [0.1.0] - 2025-12-14

### Release Notes
This is the initial stable release of the Nifty 50 AI Swing Trader system. The system is production-ready with all core modules functional, including:

- Complete ETL pipeline for data collection
- AI/ML module with model training and inference
- Strategy engine with plug-and-play architecture
- Execution engine with comprehensive risk management
- Full-featured dashboard with Control Center
- Automated scheduling system
- AI-independent operation (works with or without AI models)

**Key Features:**
- Decoupled architecture for maximum resilience
- Hub-and-spoke design with database as central communication hub
- AI models are optional enhancements, not requirements
- Paper trading mode for safe testing
- Comprehensive documentation and changelog

**Breaking Changes:** None (initial release)

**Migration Guide:** N/A (initial release)

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

- **0.2.0** (2025-12-14) - Multi-Index Support & Complete Isolation
  - Complete index isolation with index_id in all operational tables
  - Workflow synchronization with selected index
  - Index initialization/reset functionality
  - 54+ NSE indices support
  - Index-specific strategies and watchlists
  - Comprehensive index management features
  - All dashboard views filtered by index

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

