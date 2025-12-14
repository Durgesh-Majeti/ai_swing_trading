"""
Dashboard - The Monitor
Streamlit interface for viewing and controlling the trading system
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.models import (
    SessionLocal, Portfolio, TradeSignal, Order, AIPredictions,
    Watchlist, CompanyProfile, MarketData, MacroIndicator, ModelRegistry
)
from sqlalchemy import select, func, desc
from datetime import datetime, timedelta
import sys

# Page config
st.set_page_config(
    page_title="Nifty 50 AI Swing Trader",
    page_icon="📈",
    layout="wide"
)

def get_session():
    """Get database session"""
    return SessionLocal()

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Dashboard", "Portfolio", "Signals", "AI Predictions", "Watchlist", "Models", "Settings"]
)

# Main Dashboard
if page == "Dashboard":
    st.title("📈 Nifty 50 AI Swing Trader - Dashboard")
    
    session = get_session()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_positions = session.scalar(select(func.count(Portfolio.id)))
        st.metric("Active Positions", total_positions)
    
    with col2:
        total_signals = session.scalar(select(func.count(TradeSignal.id)).filter_by(status="NEW"))
        st.metric("New Signals", total_signals)
    
    with col3:
        # Calculate total P&L
        positions = session.scalars(select(Portfolio)).all()
        total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in positions)
        st.metric("Total P&L", f"₹{total_pnl:,.2f}")
    
    with col4:
        # Get latest VIX
        vix = session.scalar(
            select(MacroIndicator.value).filter_by(indicator_name="INDIA_VIX")
            .order_by(desc(MacroIndicator.date))
        )
        st.metric("India VIX", f"{vix:.2f}" if vix else "N/A")
    
    session.close()
    
    # Recent Activity
    st.subheader("📋 Recent Activity")
    
    session = get_session()
    
    # Recent Orders
    recent_orders = session.scalars(
        select(Order).order_by(desc(Order.created_at)).limit(10)
    ).all()
    
    if recent_orders:
        orders_data = [{
            "Time": o.created_at.strftime("%Y-%m-%d %H:%M"),
            "Ticker": o.ticker,
            "Side": o.side,
            "Quantity": o.quantity,
            "Price": f"₹{o.filled_price or o.price:.2f}",
            "Status": o.status
        } for o in recent_orders]
        
        st.dataframe(pd.DataFrame(orders_data), use_container_width=True)
    else:
        st.info("No recent orders")
    
    session.close()

elif page == "Portfolio":
    st.title("💼 Portfolio")
    
    session = get_session()
    
    positions = session.scalars(select(Portfolio)).all()
    
    if positions:
        portfolio_data = []
        for pos in positions:
            company = session.scalar(select(CompanyProfile).filter_by(ticker=pos.ticker))
            portfolio_data.append({
                "Ticker": pos.ticker,
                "Company": company.name if company else pos.ticker,
                "Quantity": pos.quantity,
                "Avg Entry": f"₹{pos.avg_entry_price:.2f}",
                "Current Price": f"₹{pos.current_price:.2f}",
                "Unrealized P&L": f"₹{pos.unrealized_pnl:.2f}",
                "Realized P&L": f"₹{pos.realized_pnl:.2f}",
                "Total P&L": f"₹{pos.unrealized_pnl + pos.realized_pnl:.2f}",
                "Stop Loss": f"₹{pos.stop_loss:.2f}" if pos.stop_loss else "N/A",
                "Target": f"₹{pos.target_price:.2f}" if pos.target_price else "N/A"
            })
        
        df = pd.DataFrame(portfolio_data)
        st.dataframe(df, use_container_width=True)
        
        # P&L Chart
        if len(positions) > 0:
            pnl_data = pd.DataFrame([{
                "Ticker": p.ticker,
                "P&L": p.unrealized_pnl + p.realized_pnl
            } for p in positions])
            
            fig = px.bar(pnl_data, x="Ticker", y="P&L", title="Portfolio P&L by Stock")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active positions")
    
    session.close()

elif page == "Signals":
    st.title("📊 Trade Signals")
    
    session = get_session()
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Status", ["ALL", "NEW", "PROCESSED", "REJECTED"])
    with col2:
        signal_filter = st.selectbox("Signal Type", ["ALL", "BUY", "SELL"])
    
    # Build query
    query = select(TradeSignal)
    if status_filter != "ALL":
        query = query.filter_by(status=status_filter)
    if signal_filter != "ALL":
        query = query.filter_by(signal=signal_filter)
    
    signals = session.scalars(query.order_by(desc(TradeSignal.created_at))).all()
    
    if signals:
        signals_data = [{
            "ID": s.id,
            "Time": s.created_at.strftime("%Y-%m-%d %H:%M"),
            "Ticker": s.ticker,
            "Strategy": s.strategy_name,
            "Signal": s.signal,
            "Entry": f"₹{s.entry_price:.2f}",
            "Stop Loss": f"₹{s.stop_loss:.2f}",
            "Target": f"₹{s.target_price:.2f}",
            "Quantity": s.quantity,
            "Status": s.status,
            "Reasoning": s.reasoning[:50] + "..." if s.reasoning and len(s.reasoning) > 50 else (s.reasoning or "")
        } for s in signals]
        
        df = pd.DataFrame(signals_data)
        st.dataframe(df, use_container_width=True)
        
        # Action buttons
        st.subheader("Actions")
        signal_id = st.number_input("Signal ID to process", min_value=1, value=1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve Signal"):
                signal = session.scalar(select(TradeSignal).filter_by(id=signal_id))
                if signal:
                    signal.status = "PROCESSED"
                    session.commit()
                    st.success(f"Signal {signal_id} approved")
                    st.rerun()
        
        with col2:
            if st.button("❌ Reject Signal"):
                signal = session.scalar(select(TradeSignal).filter_by(id=signal_id))
                if signal:
                    signal.status = "REJECTED"
                    session.commit()
                    st.success(f"Signal {signal_id} rejected")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Refresh"):
                st.rerun()
    else:
        st.info("No signals found")
    
    session.close()

elif page == "AI Predictions":
    st.title("🧠 AI Predictions")
    
    session = get_session()
    
    # Get active model
    active_model = session.scalar(select(ModelRegistry).filter_by(is_active=True))
    if active_model:
        st.info(f"Active Model: **{active_model.model_name}** ({active_model.model_type})")
    else:
        st.warning("No active model found")
    
    # Recent predictions
    predictions = session.scalars(
        select(AIPredictions).order_by(desc(AIPredictions.generated_at)).limit(50)
    ).all()
    
    if predictions:
        pred_data = [{
            "Ticker": p.ticker,
            "Predicted Price": f"₹{p.predicted_price:.2f}",
            "Direction": p.direction,
            "Confidence": f"{p.confidence_score:.1%}",
            "Target Date": p.target_date.strftime("%Y-%m-%d"),
            "Generated": p.generated_at.strftime("%Y-%m-%d %H:%M")
        } for p in predictions]
        
        df = pd.DataFrame(pred_data)
        st.dataframe(df, use_container_width=True)
        
        # Confidence distribution
        conf_data = pd.DataFrame([{
            "Confidence": p.confidence_score,
            "Ticker": p.ticker
        } for p in predictions])
        
        fig = px.histogram(conf_data, x="Confidence", title="Confidence Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions available")
    
    session.close()

elif page == "Watchlist":
    st.title("👀 Watchlist")
    
    session = get_session()
    
    # Current watchlist
    watchlist = session.scalars(select(Watchlist).filter_by(is_active=True)).all()
    
    if watchlist:
        watchlist_data = [{
            "Ticker": w.ticker,
            "Added": w.added_at.strftime("%Y-%m-%d"),
            "Notes": w.notes or ""
        } for w in watchlist]
        
        st.dataframe(pd.DataFrame(watchlist_data), use_container_width=True)
    else:
        st.info("Watchlist is empty")
    
    # Add to watchlist
    st.subheader("Add Stock to Watchlist")
    col1, col2 = st.columns(2)
    
    with col1:
        ticker_input = st.text_input("Ticker (e.g., RELIANCE.NS)", "")
    with col2:
        notes_input = st.text_input("Notes (optional)", "")
    
    if st.button("➕ Add to Watchlist"):
        if ticker_input:
            # Check if company exists
            company = session.scalar(select(CompanyProfile).filter_by(ticker=ticker_input))
            if not company:
                st.error(f"Company {ticker_input} not found. Please sync companies first.")
            else:
                # Check if already in watchlist
                existing = session.scalar(select(Watchlist).filter_by(ticker=ticker_input))
                if existing:
                    existing.is_active = True
                    existing.notes = notes_input
                    st.info(f"{ticker_input} already in watchlist - updated")
                else:
                    from database.models import Watchlist
                    watchlist_entry = Watchlist(
                        ticker=ticker_input,
                        notes=notes_input
                    )
                    session.add(watchlist_entry)
                    st.success(f"{ticker_input} added to watchlist")
                
                session.commit()
                st.rerun()
    
    # Remove from watchlist
    st.subheader("Remove from Watchlist")
    remove_ticker = st.selectbox("Select ticker to remove", [w.ticker for w in watchlist] if watchlist else [])
    
    if st.button("➖ Remove from Watchlist"):
        if remove_ticker:
            entry = session.scalar(select(Watchlist).filter_by(ticker=remove_ticker))
            if entry:
                entry.is_active = False
                session.commit()
                st.success(f"{remove_ticker} removed from watchlist")
                st.rerun()
    
    session.close()

elif page == "Models":
    st.title("🤖 AI Models")
    
    session = get_session()
    
    models = session.scalars(select(ModelRegistry)).all()
    
    if models:
        models_data = [{
            "Name": m.model_name,
            "Type": m.model_type,
            "Version": m.version,
            "Active": "✅" if m.is_active else "❌",
            "Created": m.created_at.strftime("%Y-%m-%d"),
            "Description": m.description or ""
        } for m in models]
        
        df = pd.DataFrame(models_data)
        st.dataframe(df, use_container_width=True)
        
        # Activate model
        st.subheader("Activate Model")
        model_names = [m.model_name for m in models]
        selected_model = st.selectbox("Select model to activate", model_names)
        
        if st.button("✅ Activate Model"):
            from ai.model_registry import ModelRegistryManager
            registry = ModelRegistryManager()
            if registry.activate_model(selected_model):
                st.success(f"Model {selected_model} activated")
                st.rerun()
    else:
        st.info("No models registered")
    
    session.close()

elif page == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("System Status")
    
    session = get_session()
    
    # Database info
    total_companies = session.scalar(select(func.count(CompanyProfile.id)))
    total_market_data = session.scalar(select(func.count(MarketData.id)))
    
    st.metric("Total Companies", total_companies)
    st.metric("Total Market Data Records", total_market_data)
    
    # Manual triggers
    st.subheader("Manual Triggers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run ETL"):
            st.info("ETL triggered (check logs)")
            # In production, this would trigger the ETL module
    
    with col2:
        if st.button("🧠 Run AI Inference"):
            st.info("AI Inference triggered (check logs)")
            # In production, this would trigger the inference engine
    
    with col3:
        if st.button("⚖️ Run Strategy Engine"):
            st.info("Strategy Engine triggered (check logs)")
            # In production, this would trigger the strategy engine
    
    session.close()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Nifty 50 AI Swing Trader**")
st.sidebar.markdown("Version 0.1.0")

