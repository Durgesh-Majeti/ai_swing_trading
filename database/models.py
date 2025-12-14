from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Date, Boolean, JSON
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

# ==========================================
# 0. DATABASE CONNECTION SETUP (Crucial Part)
# ==========================================
# This creates the file 'stock_data.db' in your project folder
DATABASE_URL = "sqlite:///stock_data.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)
Base = declarative_base()

# ==========================================
# 1. MASTER TABLE (The Hub)
# ==========================================
class CompanyProfile(Base):
    __tablename__ = "company_profiles"
    
    ticker = Column(String, primary_key=True, index=True)
    name = Column(String)
    sector = Column(String)
    industry = Column(String)
    exchange = Column(String)
    currency = Column(String)
    description = Column(Text)
    
    # Relationships
    market_data = relationship("MarketData", back_populates="company")
    fundamentals = relationship("FundamentalData", back_populates="company")
    sentiment = relationship("SentimentData", back_populates="company")
    predictions = relationship("AIPredictions", back_populates="company")
    watchlist_entries = relationship("Watchlist", back_populates="company")
    orders = relationship("Order", back_populates="company")
    portfolio_positions = relationship("Portfolio", back_populates="company")

# ==========================================
# 2. TECHNICAL ANALYSIS (Price History)
# ==========================================
class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"), index=True)
    date = Column(DateTime, index=True)
    
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    
    company = relationship("CompanyProfile", back_populates="market_data")

# ==========================================
# 3. FUNDAMENTAL ANALYSIS (Health Card)
# ==========================================
class FundamentalData(Base):
    __tablename__ = "fundamental_data"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"))
    report_date = Column(Date)
    
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    market_cap = Column(Float)
    roe = Column(Float)
    eps = Column(Float)
    revenue_growth = Column(Float)
    debt_to_equity = Column(Float)
    
    company = relationship("CompanyProfile", back_populates="fundamentals")

# ==========================================
# 4. QUANTITATIVE ANALYSIS (Math & Stats)
# ==========================================
class TechnicalIndicators(Base):
    __tablename__ = "technical_indicators"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"))
    date = Column(DateTime)
    
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    beta = Column(Float)
    atr = Column(Float)

# ==========================================
# 5. SENTIMENT ANALYSIS (News & Social)
# ==========================================
class SentimentData(Base):
    __tablename__ = "sentiment_data"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"))
    date = Column(DateTime)
    
    source = Column(String)
    headline = Column(Text)
    sentiment_score = Column(Float)
    magnitude = Column(Float)
    
    company = relationship("CompanyProfile", back_populates="sentiment")

# ==========================================
# 6. AI / ML PREDICTIONS (The Future)
# ==========================================
class AIPredictions(Base):
    __tablename__ = "ai_predictions"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"))
    generated_at = Column(DateTime, default=datetime.now)
    
    model_name = Column(String)
    target_date = Column(Date)
    predicted_price = Column(Float)
    confidence_score = Column(Float)
    direction = Column(String)
    
    company = relationship("CompanyProfile", back_populates="predictions")

# ==========================================
# 7. WATCHLIST ZONE (What to Track)
# ==========================================
class Watchlist(Base):
    __tablename__ = "watchlist"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"), unique=True, index=True)
    added_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)
    notes = Column(Text)
    
    company = relationship("CompanyProfile", back_populates="watchlist_entries")

# ==========================================
# 8. MACRO INDICATORS (External Factors)
# ==========================================
class MacroIndicator(Base):
    __tablename__ = "macro_indicators"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, index=True)
    indicator_name = Column(String, index=True)  # e.g., "INDIA_VIX", "CRUDE_OIL", "USD_INR"
    value = Column(Float)
    unit = Column(String)  # e.g., "points", "USD/barrel", "INR"
    
    __table_args__ = ({"sqlite_autoincrement": True},)

# ==========================================
# 9. MODEL REGISTRY (AI Version Control)
# ==========================================
class ModelRegistry(Base):
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String, unique=True, index=True)  # e.g., "LSTM_Nifty_v2"
    version = Column(String)
    model_type = Column(String)  # e.g., "LSTM", "XGBoost", "RandomForest"
    file_path = Column(String)  # Path to saved model file
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    trained_on_date = Column(Date)
    performance_metrics = Column(JSON)  # Store metrics like accuracy, MAE, etc.
    description = Column(Text)

# ==========================================
# 10. FEATURE STORE (ML Features)
# ==========================================
class FeatureStore(Base):
    __tablename__ = "feature_store"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"), index=True)
    date = Column(DateTime, index=True)
    
    # Technical Features
    log_return = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    atr = Column(Float)
    volatility = Column(Float)
    
    # Fundamental Features
    pe_ratio = Column(Float)
    roe = Column(Float)
    debt_to_equity = Column(Float)
    
    # Macro Features
    vix = Column(Float)
    crude_oil = Column(Float)
    usd_inr = Column(Float)
    
    # Additional computed features
    price_momentum = Column(Float)
    volume_trend = Column(Float)

# ==========================================
# 11. TRADE SIGNALS (Strategy Output)
# ==========================================
class TradeSignal(Base):
    __tablename__ = "trade_signals"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"), index=True)
    
    strategy_name = Column(String)
    signal = Column(String)  # "BUY", "SELL", "HOLD"
    status = Column(String, default="NEW")  # "NEW", "PROCESSED", "REJECTED", "CANCELLED"
    
    entry_price = Column(Float)
    stop_loss = Column(Float)
    target_price = Column(Float)
    quantity = Column(Integer)
    
    reasoning = Column(Text)  # Why this signal was generated
    priority = Column(Integer, default=5)  # 1-10, higher = more important

# ==========================================
# 12. ORDERS (Execution Zone)
# ==========================================
class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("trade_signals.id"), nullable=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"), index=True)
    
    order_type = Column(String)  # "MARKET", "LIMIT", "SL", "SL-M"
    side = Column(String)  # "BUY", "SELL"
    quantity = Column(Integer)
    price = Column(Float)  # Limit price or trigger price
    
    status = Column(String, default="SUBMITTED")  # "SUBMITTED", "FILLED", "REJECTED", "CANCELLED", "CLOSED"
    
    created_at = Column(DateTime, default=datetime.now, index=True)
    filled_at = Column(DateTime, nullable=True)
    filled_price = Column(Float, nullable=True)
    
    stop_loss = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    
    mode = Column(String, default="PAPER")  # "PAPER" or "LIVE"
    broker_order_id = Column(String, nullable=True)  # If live trading
    
    company = relationship("CompanyProfile", back_populates="orders")

# ==========================================
# 13. PORTFOLIO (Current Positions)
# ==========================================
class Portfolio(Base):
    __tablename__ = "portfolio"
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"), unique=True, index=True)
    
    quantity = Column(Integer)
    avg_entry_price = Column(Float)
    current_price = Column(Float)
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # P&L Tracking
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    
    # Risk Management
    stop_loss = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    
    # Metadata
    entry_date = Column(DateTime, default=datetime.now)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)
    
    company = relationship("CompanyProfile", back_populates="portfolio_positions")

# ==========================================
# 14. BACKTEST RESULTS (Performance Analysis)
# ==========================================
class BacktestRun(Base):
    """Stores backtest run metadata"""
    __tablename__ = "backtest_runs"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.now)
    strategy_name = Column(String, index=True)
    ticker = Column(String, ForeignKey("company_profiles.ticker"), index=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    initial_capital = Column(Float)
    position_size_pct = Column(Float)
    
    # Results
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    net_profit = Column(Float)
    total_profit = Column(Float)
    total_loss = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    final_capital = Column(Float)
    
    # Metadata
    notes = Column(Text)

class BacktestTrade(Base):
    """Stores individual trades from backtest"""
    __tablename__ = "backtest_trades"
    
    id = Column(Integer, primary_key=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id"), index=True)
    
    entry_date = Column(DateTime)
    exit_date = Column(DateTime)
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Integer)
    side = Column(String)  # BUY or SELL
    pnl = Column(Float)
    pnl_pct = Column(Float)
    exit_reason = Column(String)  # STOP_LOSS, TARGET, END_DATE