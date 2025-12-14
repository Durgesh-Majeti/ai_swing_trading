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
    Watchlist, CompanyProfile, MarketData, MacroIndicator, ModelRegistry,
    BacktestRun, BacktestTrade, StrategyMetadata, Index
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

def get_index_tickers(index: Index = None):
    """
    Get list of tickers for a given index.
    If index is None, returns None (meaning no filter).
    
    Args:
        index: Index object or None
        
    Returns:
        List of ticker strings or None (for no filter)
    """
    if not index:
        return None
    
    session = get_session()
    try:
        # Query companies directly using the relationship
        # First, get the index from database to ensure it's attached to session
        db_index = session.scalar(select(Index).filter_by(id=index.id))
        if not db_index:
            return []
        
        # Get companies through the relationship
        companies = db_index.companies
        tickers = [c.ticker for c in companies]
        return tickers if tickers else []
    except Exception as e:
        # Fallback: query using the junction table
        from database.models import company_index_mapping
        try:
            tickers = session.scalars(
                select(company_index_mapping.c.ticker)
                .where(company_index_mapping.c.index_id == index.id)
            ).all()
            return list(tickers) if tickers else []
        except Exception:
            return []
    finally:
        session.close()

# Sidebar Navigation
st.sidebar.title("📊 Navigation")

# Index Selector
session = get_session()
indices = session.scalars(select(Index).filter_by(is_active=True).order_by(Index.name)).all()
index_names = {idx.display_name: idx.name for idx in indices}
index_display_names = list(index_names.keys())

if index_display_names:
    selected_index_display = st.sidebar.selectbox(
        "📈 Select Index",
        ["All Indices"] + index_display_names,
        key="index_selector"
    )
    selected_index_name = index_names.get(selected_index_display) if selected_index_display != "All Indices" else None
    selected_index = session.scalar(select(Index).filter_by(name=selected_index_name)) if selected_index_name else None
    
    # Index Initialization/Reset Option
    if selected_index:
        st.sidebar.divider()
        st.sidebar.subheader(f"🔧 {selected_index.display_name}")
        
        with st.sidebar.expander("⚠️ Initialize Index", expanded=False):
            st.warning("**Danger Zone**: This will delete all operational data for this index!")
            st.markdown("""
            **Will be deleted:**
            - All portfolio positions
            - All trade signals
            - All orders
            - All AI predictions
            - Watchlist entries
            - Strategy metadata
            """)
            
            if st.button("🗑️ Reset Index", key="reset_index_btn", type="secondary", use_container_width=True):
                try:
                    from utils.index_reset import reset_index
                    with st.spinner(f"Resetting {selected_index.display_name}..."):
                        result = reset_index(selected_index_name, keep_watchlist=False, keep_strategies=False)
                        st.success(f"✅ Index reset complete!")
                        st.json(result)
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to reset index: {str(e)}")
else:
    selected_index_display = "All Indices"
    selected_index_name = None
    selected_index = None
    st.sidebar.warning("No indices found. Run migration to create indices.")

session.close()

page = st.sidebar.selectbox(
    "Choose a page",
    ["Dashboard", "Control Center", "Backtesting", "Strategies", "Portfolio", "Signals", "AI Predictions", "Watchlist", "Models", "Settings"]
)

# Main Dashboard
if page == "Dashboard":
    if selected_index:
        st.title(f"📈 {selected_index.display_name} - Dashboard")
        st.info(f"📊 Showing data for: **{selected_index.display_name}**")
    else:
        st.title("📈 Multi-Index AI Swing Trader - Dashboard")
        st.info("📊 Showing data for: **All Indices**")
    
    session = get_session()
    
    # Key Metrics - Filter by index_id for complete isolation
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Filter positions by index_id
        if selected_index:
            total_positions = session.scalar(
                select(func.count(Portfolio.id)).where(Portfolio.index_id == selected_index.id)
            )
        else:
            total_positions = session.scalar(select(func.count(Portfolio.id)))
        st.metric("Active Positions", total_positions or 0)
    
    with col2:
        # Filter signals by index_id
        if selected_index:
            total_signals = session.scalar(
                select(func.count(TradeSignal.id))
                .where(TradeSignal.status == "NEW")
                .where(TradeSignal.index_id == selected_index.id)
            )
        else:
            total_signals = session.scalar(select(func.count(TradeSignal.id)).filter_by(status="NEW"))
        st.metric("New Signals", total_signals or 0)
    
    with col3:
        # Calculate total P&L filtered by index_id
        if selected_index:
            positions = session.scalars(
                select(Portfolio).where(Portfolio.index_id == selected_index.id)
            ).all()
        else:
            positions = session.scalars(select(Portfolio)).all()
        total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in positions)
        st.metric("Total P&L", f"₹{total_pnl:,.2f}")
    
    with col4:
        # Get latest VIX (not index-specific)
        vix = session.scalar(
            select(MacroIndicator.value).filter_by(indicator_name="INDIA_VIX")
            .order_by(desc(MacroIndicator.date))
        )
        st.metric("India VIX", f"{vix:.2f}" if vix else "N/A")
    
    session.close()
    
    # Recent Activity
    st.subheader("📋 Recent Activity")
    
    session = get_session()
    
    # Recent Orders filtered by index_id
    if selected_index:
        recent_orders = session.scalars(
            select(Order)
            .where(Order.index_id == selected_index.id)
            .order_by(desc(Order.created_at))
            .limit(10)
        ).all()
    else:
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
        st.info("No recent orders" + (f" for {selected_index.display_name}" if selected_index else ""))
    
    session.close()

elif page == "Portfolio":
    if selected_index:
        st.title(f"💼 {selected_index.display_name} Portfolio")
        st.info(f"📊 Showing portfolio for: **{selected_index.display_name}**")
    else:
        st.title("💼 Portfolio")
        st.info("📊 Showing portfolio for: **All Indices**")
    
    session = get_session()
    
    # Filter positions by index_id for complete isolation
    if selected_index:
        positions = session.scalars(
            select(Portfolio).where(Portfolio.index_id == selected_index.id)
        ).all()
    else:
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
        st.info("No active positions" + (f" for {selected_index.display_name}" if selected_index else ""))
    
    session.close()

elif page == "Signals":
    if selected_index:
        st.title(f"📊 {selected_index.display_name} Trade Signals")
        st.info(f"📊 Showing signals for: **{selected_index.display_name}**")
    else:
        st.title("📊 Trade Signals")
        st.info("📊 Showing signals for: **All Indices**")
    
    session = get_session()
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Status", ["ALL", "NEW", "PROCESSED", "REJECTED"])
    with col2:
        signal_filter = st.selectbox("Signal Type", ["ALL", "BUY", "SELL"])
    
    # Build query - Filter by index_id for complete isolation
    query = select(TradeSignal)
    if status_filter != "ALL":
        query = query.filter_by(status=status_filter)
    if signal_filter != "ALL":
        query = query.filter_by(signal=signal_filter)
    
    # Filter by index_id
    if selected_index:
        query = query.where(TradeSignal.index_id == selected_index.id)
    
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
    if selected_index:
        st.title(f"🧠 {selected_index.display_name} AI Predictions")
        st.info(f"📊 Showing predictions for: **{selected_index.display_name}**")
    else:
        st.title("🧠 AI Predictions")
        st.info("📊 Showing predictions for: **All Indices**")
    
    session = get_session()
    
    # Get active model
    active_model = session.scalar(select(ModelRegistry).filter_by(is_active=True))
    if active_model:
        st.info(f"Active Model: **{active_model.model_name}** ({active_model.model_type})")
    else:
        st.warning("No active model found")
    
    # Recent predictions filtered by index_id for complete isolation
    query = select(AIPredictions).order_by(desc(AIPredictions.generated_at)).limit(50)
    if selected_index:
        query = query.where(AIPredictions.index_id == selected_index.id)
    
    predictions = session.scalars(query).all()
    
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
    
    # Filter by selected index
    if selected_index:
        st.info(f"📈 Showing watchlist for: **{selected_index.display_name}**")
        watchlist = session.scalars(
            select(Watchlist).filter_by(is_active=True, index_id=selected_index.id)
        ).all()
    else:
        watchlist = session.scalars(select(Watchlist).filter_by(is_active=True)).all()
    
    if watchlist:
        watchlist_data = [{
            "Ticker": w.ticker,
            "Index": w.index.display_name if w.index else "N/A",
            "Added": w.added_at.strftime("%Y-%m-%d"),
            "Notes": w.notes or ""
        } for w in watchlist]
        
        st.dataframe(pd.DataFrame(watchlist_data), use_container_width=True)
    else:
        st.info("Watchlist is empty" + (f" for {selected_index.display_name}" if selected_index else ""))
    
    # Add to watchlist
    st.subheader("Add Stock to Watchlist")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker_input = st.text_input("Ticker (e.g., RELIANCE.NS)", "")
    with col2:
        # Index selector for adding
        add_indices = session.scalars(select(Index).filter_by(is_active=True).order_by(Index.name)).all()
        index_options = {idx.display_name: idx for idx in add_indices}
        selected_add_index_display = st.selectbox(
            "Index",
            list(index_options.keys()),
            key="add_watchlist_index"
        )
        selected_add_index = index_options[selected_add_index_display]
    with col3:
        notes_input = st.text_input("Notes (optional)", "")
    
    if st.button("➕ Add to Watchlist"):
        if ticker_input:
            # Check if company exists
            company = session.scalar(select(CompanyProfile).filter_by(ticker=ticker_input))
            if not company:
                st.error(f"Company {ticker_input} not found. Please sync companies first.")
            else:
                # Check if already in watchlist for this index
                existing = session.scalar(
                    select(Watchlist).filter_by(ticker=ticker_input, index_id=selected_add_index.id)
                )
                if existing:
                    existing.is_active = True
                    existing.notes = notes_input
                    st.info(f"{ticker_input} already in watchlist for {selected_add_index.display_name} - updated")
                else:
                    watchlist_entry = Watchlist(
                        ticker=ticker_input,
                        index_id=selected_add_index.id,
                        notes=notes_input
                    )
                    session.add(watchlist_entry)
                    st.success(f"{ticker_input} added to {selected_add_index.display_name} watchlist")
                
                session.commit()
                st.rerun()
    
    # Remove from watchlist
    if watchlist:
        st.subheader("Remove from Watchlist")
        remove_options = [f"{w.ticker} ({w.index.display_name if w.index else 'N/A'})" for w in watchlist]
        remove_selection = st.selectbox("Select ticker to remove", remove_options)
        
        if st.button("➖ Remove from Watchlist"):
            if remove_selection:
                ticker_to_remove = remove_selection.split(" (")[0]
                entry = session.scalar(
                    select(Watchlist).filter_by(ticker=ticker_to_remove, is_active=True)
                )
                if entry:
                    entry.is_active = False
                    session.commit()
                    st.success(f"{ticker_to_remove} removed from watchlist")
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

elif page == "Control Center":
    if selected_index:
        st.title(f"🎮 {selected_index.display_name} Control Center")
        st.info(f"📊 Showing metrics and controls for: **{selected_index.display_name}**")
    else:
        st.title("🎮 Control Center")
        st.info("📊 Showing metrics and controls for: **All Indices**")
    
    st.markdown("**Control and execute all trading system workflows from here**")
    
    # Info banner about AI independence
    st.info("💡 **System Resilience**: This system is designed to work independently of AI models. If no AI model is available, strategies will use technical and fundamental analysis only. All workflows will continue to function.")
    
    session = get_session()
    
    # System Status
    st.subheader("📊 System Status")
    
    # Get tickers for selected index (for company count)
    index_tickers = get_index_tickers(selected_index)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Filter companies by index
        if index_tickers:
            total_companies = len(index_tickers)
        else:
            total_companies = session.scalar(select(func.count(CompanyProfile.ticker)))
        st.metric("Companies", total_companies or 0)
    
    with col2:
        # Filter market data by index tickers (market data doesn't have index_id)
        if index_tickers:
            total_market_data = session.scalar(
                select(func.count(MarketData.id)).where(MarketData.ticker.in_(index_tickers))
            )
        else:
            total_market_data = session.scalar(select(func.count(MarketData.id)))
        st.metric("Market Records", total_market_data or 0)
    
    with col3:
        # Filter watchlist by index
        if selected_index:
            watchlist_count = session.scalar(
                select(func.count(Watchlist.id)).filter_by(is_active=True, index_id=selected_index.id)
            )
        else:
            watchlist_count = session.scalar(select(func.count(Watchlist.id)).filter_by(is_active=True))
        st.metric("Watchlist", watchlist_count or 0)
    
    with col4:
        active_model = session.scalar(select(func.count(ModelRegistry.id)).filter_by(is_active=True))
        st.metric("Active Models", active_model or 0)
    
    session.close()
    
    st.divider()
    
    # Index Management
    st.subheader("📈 Index Management")
    
    # Create/Discover Indices
    with st.expander("➕ Create All NSE Indices", expanded=False):
        st.markdown("**Create all known NSE indices in the database**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("This will create all known NSE indices (50+ indices) in the database. You can then sync companies for each index.")
        with col2:
            if st.button("➕ Create All Indices", key="create_indices_btn", use_container_width=True):
                with st.spinner("Creating all NSE indices in database..."):
                    try:
                        from utils.discover_nse_indices import create_all_indices_in_db
                        created, updated = create_all_indices_in_db()
                        st.success(f"✅ Created {created} new indices, updated {updated} existing indices")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to create indices: {str(e)}")
    
    with st.expander("🔄 Sync Companies for Indices", expanded=True):
        st.markdown("**Update company lists for each index from NSE**")
        
        # Get all indices
        indices = session.scalars(select(Index).filter_by(is_active=True).order_by(Index.name)).all()
        
        if indices:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info("Sync companies from NSE for selected indices. This will fetch the latest company lists and assign them to their respective indices.")
                
                # Multi-select for indices
                index_options = {idx.display_name: idx.name for idx in indices}
                selected_indices_display = st.multiselect(
                    "Select Indices to Sync",
                    list(index_options.keys()),
                    default=list(index_options.keys())[:3],  # Default to first 3
                    key="sync_indices_select"
                )
            
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("🔄 Sync Selected Indices", key="sync_indices_btn", use_container_width=True):
                    if selected_indices_display:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results = {}
                        
                        total = len(selected_indices_display)
                        for i, index_display in enumerate(selected_indices_display):
                            index_name = index_options[index_display]
                            status_text.text(f"Syncing {index_display}... ({i+1}/{total})")
                            
                            try:
                                from engine.loaders.profile_loader import sync_index_companies
                                new_count, assigned_count = sync_index_companies(index_name)
                                results[index_display] = {
                                    'new': new_count,
                                    'assigned': assigned_count,
                                    'success': True
                                }
                            except Exception as e:
                                results[index_display] = {
                                    'new': 0,
                                    'assigned': 0,
                                    'success': False,
                                    'error': str(e)
                                }
                            
                            progress_bar.progress((i + 1) / total)
                        
                        # Show results
                        status_text.text("✅ Sync Complete!")
                        st.success("✅ Index sync completed!")
                        
                        # Results table
                        results_data = []
                        for index_display, result in results.items():
                            if result['success']:
                                results_data.append({
                                    "Index": index_display,
                                    "New Companies": result['new'],
                                    "Assigned": result['assigned'],
                                    "Status": "✅ Success"
                                })
                            else:
                                results_data.append({
                                    "Index": index_display,
                                    "New Companies": 0,
                                    "Assigned": 0,
                                    "Status": f"❌ Error: {result.get('error', 'Unknown')}"
                                })
                        
                        st.dataframe(pd.DataFrame(results_data), use_container_width=True)
                        st.rerun()
                    else:
                        st.warning("Please select at least one index to sync.")
            
            # Quick sync all button
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("💡 **Quick Action**: Sync all indices at once (this may take a few minutes)")
            with col2:
                if st.button("🚀 Sync All Indices", key="sync_all_btn", use_container_width=True):
                    with st.spinner("Syncing all indices from NSE... This may take several minutes."):
                        try:
                            from engine.loaders.profile_loader import sync_all_indices
                            results = sync_all_indices()
                            
                            # Display summary
                            summary_data = []
                            for index_name, result in results.items():
                                summary_data.append({
                                    "Index": index_name,
                                    "New Companies": result['new_companies'],
                                    "Assigned": result['assigned_companies'],
                                    "Status": "✅ Success" if result['success'] else "❌ Failed"
                                })
                            
                            st.success("✅ All indices synced successfully!")
                            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to sync indices: {str(e)}")
        else:
            st.warning("No indices found. Please run migration to create indices first.")
    
    st.divider()
    
    # Workflow Controls
    st.subheader("🔄 Workflow Controls")
    
    # Show index context
    if selected_index:
        st.info(f"📊 **Running workflows for: {selected_index.display_name}** - All operations will be limited to this index only.")
    else:
        st.warning("⚠️ **No index selected** - Workflows will run for **All Indices**. Select an index from the sidebar to limit operations to a specific index.")
    
    st.divider()
    
    # ETL Section
    with st.expander("📥 ETL - Data Collection", expanded=True):
        st.markdown("**Fetch market data, macro indicators, and calculate technical indicators**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            index_info = f" for **{selected_index.display_name}**" if selected_index else " for **All Indices**"
            st.info(f"This will sync market data for watchlist stocks{index_info}, fetch macro indicators (VIX, Crude, USD/INR), and calculate technical indicators.")
        with col2:
            if st.button("🔄 Run ETL", key="etl_btn", use_container_width=True):
                with st.spinner("Running ETL Pipeline... This may take a few minutes."):
                    try:
                        from engine.etl import ETLModule
                        index_id = selected_index.id if selected_index else None
                        etl = ETLModule(index_id=index_id)
                        etl.run_full_sync()
                        st.success("✅ ETL Pipeline completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ ETL failed: {str(e)}")
    
    # Feature Generation
    with st.expander("🔧 Feature Generation"):
        st.markdown("**Generate ML-ready features for stocks**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            index_info = f" for **{selected_index.display_name}**" if selected_index else " for **All Indices**"
            st.info(f"Transforms raw market data into machine learning features (RSI, MACD, momentum, etc.){index_info}")
        with col2:
            if st.button("🔧 Generate Features", key="features_btn", use_container_width=True):
                with st.spinner("Generating features..."):
                    try:
                        from ai.feature_store import FeatureStoreEngine
                        index_id = selected_index.id if selected_index else None
                        engine = FeatureStoreEngine(index_id=index_id)
                        engine.generate_all_features()
                        st.success("✅ Features generated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Feature generation failed: {str(e)}")
    
    # Model Training
    with st.expander("🤖 Model Training"):
        st.markdown("**Train a new AI model**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            model_name = st.text_input("Model Name", value="RandomForest_Swing_v1", key="model_name")
            auto_activate = st.checkbox("Activate after training", value=False)
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("🎓 Train Model", key="train_btn", use_container_width=True):
                with st.spinner(f"Training model {model_name}... This may take several minutes."):
                    try:
                        from ai.train_model import ModelTrainer
                        trainer = ModelTrainer()
                        model = trainer.train_random_forest(model_name)
                        if model:
                            if auto_activate:
                                trainer.activate_model(model_name)
                                st.success(f"✅ Model {model_name} trained and activated!")
                            else:
                                st.success(f"✅ Model {model_name} trained successfully!")
                            st.rerun()
                        else:
                            st.warning("⚠️ Model training failed. Check logs for details.")
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
    
    # AI Inference
    with st.expander("🧠 AI Inference"):
        st.markdown("**Generate predictions using the active AI model**")
        
        # Check if model is available
        session_check = get_session()
        active_model = session_check.scalar(select(ModelRegistry).filter_by(is_active=True))
        session_check.close()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            index_info = f" for **{selected_index.display_name}**" if selected_index else " for **All Indices**"
            if active_model:
                st.info(f"Runs the active model to generate price predictions for watchlist stocks{index_info}")
            else:
                st.warning("⚠️ No active AI model found. Train and activate a model first, or the system will work with technical/fundamental analysis only.")
        with col2:
            if st.button("🧠 Run Inference", key="inference_btn", use_container_width=True, disabled=not active_model):
                with st.spinner("Running AI inference... Generating predictions..."):
                    try:
                        from ai.inference import InferenceEngine
                        index_id = selected_index.id if selected_index else None
                        engine = InferenceEngine(index_id=index_id)
                        engine.run_daily_inference()
                        st.success("✅ AI Inference completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.warning(f"⚠️ Inference skipped: {str(e)}. System will continue with technical/fundamental analysis.")
    
    # Strategy Engine
    with st.expander("⚖️ Strategy Engine"):
        st.markdown("**Generate trade signals using all active strategies**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            index_info = f" for **{selected_index.display_name}**" if selected_index else " for **All Indices**"
            st.info(f"Runs all registered strategies to generate BUY/SELL signals based on technical, fundamental, and AI analysis{index_info}")
        with col2:
            if st.button("⚖️ Generate Signals", key="strategy_btn", use_container_width=True):
                with st.spinner("Running strategy engine... Analyzing stocks..."):
                    try:
                        from strategies.engine import StrategyEngine
                        # Use selected index if available
                        index_name = selected_index_name if selected_index_name else None
                        engine = StrategyEngine(index_name=index_name)
                        engine.run_daily_analysis(index_name=index_name)
                        st.success("✅ Strategy Engine completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Strategy engine failed: {str(e)}")
    
    # Execution Engine
    with st.expander("💼 Execution Engine"):
        st.markdown("**Execute trade signals with risk management**")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            mode = st.radio("Trading Mode", ["PAPER", "LIVE"], key="exec_mode", horizontal=True)
            st.warning("⚠️ LIVE mode will execute real trades. Use with caution!")
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("💼 Execute Trades", key="execute_btn", use_container_width=True):
                with st.spinner("Processing trade signals..."):
                    try:
                        from execution.executor import ExecutionEngine
                        index_id = selected_index.id if selected_index else None
                        engine = ExecutionEngine(mode=mode)
                        engine.process_new_signals(index_id=index_id)
                        engine.update_portfolio_prices()
                        engine.close()
                        st.success(f"✅ Execution completed in {mode} mode!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Execution failed: {str(e)}")
    
    # Full Workflow
    st.divider()
    st.subheader("🚀 Full Workflow")
    st.markdown("**Run the complete trading workflow in sequence**")
    
    workflow_steps = st.multiselect(
        "Select workflow steps:",
        ["ETL", "Features", "Inference", "Strategy", "Execution"],
        default=["ETL", "Features", "Inference", "Strategy", "Execution"]
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        index_info = f" for **{selected_index.display_name}**" if selected_index else " for **All Indices**"
        st.info(f"This will run all selected steps in sequence{index_info}. Note: Inference is optional - the system will work with technical/fundamental analysis if no AI model is available.")
    with col2:
        if st.button("🚀 Run Full Workflow", key="full_workflow_btn", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps_completed = 0
            total_steps = len(workflow_steps)
            
            try:
                # Get index_id for all workflows
                index_id = selected_index.id if selected_index else None
                
                # ETL
                if "ETL" in workflow_steps:
                    status_text.text("Step 1/{}: Running ETL...".format(total_steps))
                    from engine.etl import ETLModule
                    etl = ETLModule(index_id=index_id)
                    etl.run_full_sync()
                    steps_completed += 1
                    progress_bar.progress(steps_completed / total_steps)
                
                # Features
                if "Features" in workflow_steps:
                    status_text.text("Step {}/{}: Generating features...".format(steps_completed + 1, total_steps))
                    from ai.feature_store import FeatureStoreEngine
                    engine = FeatureStoreEngine(index_id=index_id)
                    engine.generate_all_features()
                    steps_completed += 1
                    progress_bar.progress(steps_completed / total_steps)
                
                # Inference (optional - continues even if no model)
                if "Inference" in workflow_steps:
                    status_text.text("Step {}/{}: Running AI inference...".format(steps_completed + 1, total_steps))
                    try:
                        from ai.inference import InferenceEngine
                        engine = InferenceEngine(index_id=index_id)
                        engine.run_daily_inference()
                    except Exception as e:
                        st.warning(f"⚠️ AI Inference skipped (no model available): {str(e)}")
                        st.info("Continuing with remaining steps...")
                    steps_completed += 1
                    progress_bar.progress(steps_completed / total_steps)
                
                # Strategy
                if "Strategy" in workflow_steps:
                    status_text.text("Step {}/{}: Generating signals...".format(steps_completed + 1, total_steps))
                    from strategies.engine import StrategyEngine
                    # Use selected index if available
                    index_name = selected_index_name if selected_index_name else None
                    engine = StrategyEngine(index_name=index_name)
                    engine.run_daily_analysis(index_name=index_name)
                    steps_completed += 1
                    progress_bar.progress(steps_completed / total_steps)
                
                # Execution
                if "Execution" in workflow_steps:
                    status_text.text("Step {}/{}: Executing trades...".format(steps_completed + 1, total_steps))
                    from execution.executor import ExecutionEngine
                    engine = ExecutionEngine(mode="PAPER")
                    engine.process_new_signals(index_id=index_id)
                    engine.update_portfolio_prices()
                    engine.close()
                    steps_completed += 1
                    progress_bar.progress(1.0)
                
                status_text.text("✅ All steps completed!")
                st.success("🎉 Full workflow completed successfully!")
                st.balloons()
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Workflow failed at step {steps_completed + 1}: {str(e)}")
                progress_bar.progress(0)

elif page == "Backtesting":
    st.title("📊 Backtesting")
    st.markdown("**Test strategies on historical data to evaluate performance**")
    
    # Metrics Explanation
    with st.expander("📖 Understanding Backtest Metrics", expanded=False):
        st.markdown("""
        ### Performance Metrics Explained
        
        #### 💰 Profit & Loss Metrics
        
        **Net Profit**
        - Total profit or loss after all trades
        - Formula: Final Capital - Initial Capital
        - Positive = Profit, Negative = Loss
        - **Interpretation**: Shows absolute rupee gain/loss
        
        **Return %**
        - Percentage return on initial capital
        - Formula: (Net Profit / Initial Capital) × 100
        - **Interpretation**: 
          - 10% = ₹10,000 profit on ₹1,00,000 capital
          - Higher is better, but consider risk
        
        **Total Profit**
        - Sum of all winning trades only
        - Shows gross profit before losses
        - **Interpretation**: Raw profit potential of strategy
        
        **Total Loss**
        - Sum of all losing trades (absolute value)
        - Shows gross losses
        - **Interpretation**: Risk exposure of strategy
        
        #### 📊 Trade Statistics
        
        **Total Trades**
        - Number of trades executed during backtest period
        - **Interpretation**: 
          - Too few = Strategy may be too conservative
          - Too many = May indicate overtrading
          - Ideal: Depends on strategy type (swing trading: 10-50 trades/year)
        
        **Win Rate**
        - Percentage of profitable trades
        - Formula: (Winning Trades / Total Trades) × 100
        - **Interpretation**:
          - 50% = Break-even (if avg win = avg loss)
          - 60%+ = Good win rate
          - 40% can still be profitable if avg win >> avg loss
        
        **Winning Trades**
        - Count of trades that ended in profit
        - **Interpretation**: Shows strategy's success frequency
        
        **Losing Trades**
        - Count of trades that ended in loss
        - **Interpretation**: Shows how often strategy fails
        
        #### 📈 Risk Metrics
        
        **Profit Factor**
        - Ratio of total profit to total loss
        - Formula: Total Profit / Total Loss
        - **Interpretation**:
          - **> 2.0** = Excellent (makes ₹2 for every ₹1 lost)
          - **1.5 - 2.0** = Good
          - **1.0 - 1.5** = Acceptable
          - **< 1.0** = Losing strategy (loses more than gains)
          - **Key**: Most important metric after net profit
        
        **Max Drawdown**
        - Maximum peak-to-trough decline in capital
        - Shows worst-case scenario during backtest
        - **Interpretation**:
          - **-5%** = Lost 5% from peak at worst point
          - Lower (less negative) = Better risk control
          - **-20%** = High risk, may cause psychological stress
          - **Warning**: Past drawdowns don't guarantee future limits
        
        **Average Win**
        - Average profit per winning trade
        - Formula: Total Profit / Winning Trades
        - **Interpretation**: 
          - Higher = Better profit per successful trade
          - Compare with Average Loss for risk/reward
        
        **Average Loss**
        - Average loss per losing trade
        - Formula: Total Loss / Losing Trades
        - **Interpretation**:
          - Lower = Better risk control
          - Should be < Average Win for profitability
        
        #### 🎯 Risk/Reward Analysis
        
        **Ideal Strategy Profile:**
        - Profit Factor > 1.5
        - Win Rate > 50% (or high profit factor with lower win rate)
        - Average Win > Average Loss (at least 2:1 ratio)
        - Max Drawdown < 15%
        - Consistent performance across different market conditions
        
        **Red Flags:**
        - Profit Factor < 1.0 (losing strategy)
        - Max Drawdown > 25% (high risk)
        - Average Loss > Average Win (poor risk management)
        - Very low win rate (< 30%) with low profit factor
        
        #### 📋 Trade Details
        
        **Entry Date / Exit Date**
        - When trade was opened and closed
        - Shows trade duration
        
        **Side**
        - BUY = Long position (profit when price goes up)
        - SELL = Short position (profit when price goes down)
        
        **Entry Price / Exit Price**
        - Price at which trade was entered and exited
        
        **Quantity**
        - Number of shares traded
        
        **P&L (Profit & Loss)**
        - Absolute profit/loss in rupees for this trade
        - Positive = Profit, Negative = Loss
        
        **P&L %**
        - Percentage return on this specific trade
        - Formula: ((Exit Price - Entry Price) / Entry Price) × 100
        
        **Exit Reason**
        - **STOP_LOSS**: Price hit stop loss (risk management worked)
        - **TARGET**: Price hit target (profit taking)
        - **END_DATE**: Trade still open at end of backtest period
        
        ---
        
        **💡 Pro Tip**: A strategy with 40% win rate but 3:1 profit factor (avg win 3× avg loss) 
        can be more profitable than 60% win rate with 1:1 profit factor!
        """)
    
    session = get_session()
    
    # Configuration
    st.subheader("⚙️ Backtest Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Strategy selection
        from strategies.registry import StrategyRegistry
        registry = StrategyRegistry()
        strategies = registry.list_strategies()
        
        if not strategies:
            st.warning("No strategies available. Create strategies first.")
            session.close()
            st.stop()
        
        selected_strategy = st.selectbox("Select Strategy", strategies)
    
    with col2:
        # Ticker selection - filtered by index
        if selected_index:
            watchlist = session.scalars(
                select(Watchlist).filter_by(is_active=True, index_id=selected_index.id)
            ).all()
        else:
            watchlist = session.scalars(select(Watchlist).filter_by(is_active=True)).all()
        
        # Also include companies from index if watchlist is empty
        if not watchlist and selected_index:
            index_tickers = get_index_tickers(selected_index)
            tickers = index_tickers or []
        else:
            tickers = [w.ticker for w in watchlist] if watchlist else []
        
        if not tickers:
            st.warning("Watchlist is empty" + (f" for {selected_index.display_name}" if selected_index else "") + ". Add stocks to watchlist first.")
            session.close()
            st.stop()
        
        selected_ticker = st.selectbox("Select Ticker", tickers)
    
    with col3:
        # Date range
        days_back = st.number_input("Days to Look Back", min_value=30, max_value=3650, value=365, step=30)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        st.info(f"Period: {start_date} to {end_date}")
    
    col4, col5 = st.columns(2)
    with col4:
        initial_capital = st.number_input("Initial Capital (₹)", min_value=10000, value=100000, step=10000)
    with col5:
        position_size_pct = st.slider("Position Size (% of capital)", min_value=1, max_value=50, value=10) / 100
    
    # Run backtest
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        with st.spinner(f"Running backtest for {selected_strategy} on {selected_ticker}..."):
            try:
                from backtesting.engine import BacktestEngine
                
                engine = BacktestEngine(initial_capital=initial_capital)
                result = engine.run_backtest(
                    selected_strategy,
                    selected_ticker,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    position_size_pct
                )
                engine.close()
                
                # Display results
                st.success("✅ Backtest completed!")
                
                # Key Metrics
                st.subheader("📈 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Net Profit", f"₹{result.net_profit:,.2f}")
                    st.metric("Return %", f"{(result.net_profit / initial_capital) * 100:.2f}%")
                
                with col2:
                    st.metric("Total Trades", result.total_trades)
                    st.metric("Win Rate", f"{result.win_rate:.1f}%")
                
                with col3:
                    st.metric("Profit Factor", f"{result.profit_factor:.2f}")
                    st.metric("Max Drawdown", f"{result.max_drawdown:.2f}%")
                
                with col4:
                    st.metric("Avg Win", f"₹{result.avg_win:,.2f}")
                    st.metric("Avg Loss", f"₹{result.avg_loss:,.2f}")
                
                # Detailed breakdown
                st.subheader("📊 Trade Breakdown")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Winning Trades:** {result.winning_trades}")
                    st.write(f"**Total Profit:** ₹{result.total_profit:,.2f}")
                with col2:
                    st.write(f"**Losing Trades:** {result.losing_trades}")
                    st.write(f"**Total Loss:** ₹{result.total_loss:,.2f}")
                
                # Trade list
                if result.trades:
                    st.subheader("📋 Trade History")
                    trades_df = pd.DataFrame(result.trades)
                    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
                    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')
                    
                    display_df = trades_df[[
                        'entry_date', 'exit_date', 'side', 'entry_price', 
                        'exit_price', 'quantity', 'pnl', 'pnl_pct', 'exit_reason'
                    ]].copy()
                    display_df.columns = ['Entry Date', 'Exit Date', 'Side', 'Entry Price', 'Exit Price', 'Quantity', 'P&L', 'P&L %', 'Exit Reason']
                    display_df['Entry Price'] = display_df['Entry Price'].apply(lambda x: f"₹{x:.2f}")
                    display_df['Exit Price'] = display_df['Exit Price'].apply(lambda x: f"₹{x:.2f}")
                    display_df['P&L'] = display_df['P&L'].apply(lambda x: f"₹{x:,.2f}")
                    display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # P&L Chart
                    fig = px.bar(
                        trades_df, 
                        x='exit_date', 
                        y='pnl',
                        color='pnl',
                        color_continuous_scale=['red', 'green'],
                        title="Trade P&L Over Time"
                    )
                    fig.update_layout(xaxis_title="Exit Date", yaxis_title="Profit/Loss (₹)")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Save results option
                if st.button("💾 Save Backtest Results"):
                    try:
                        backtest_run = BacktestRun(
                            strategy_name=result.strategy_name,
                            ticker=result.ticker,
                            start_date=result.start_date,
                            end_date=result.end_date,
                            initial_capital=initial_capital,
                            position_size_pct=position_size_pct,
                            total_trades=result.total_trades,
                            winning_trades=result.winning_trades,
                            losing_trades=result.losing_trades,
                            net_profit=result.net_profit,
                            total_profit=result.total_profit,
                            total_loss=result.total_loss,
                            win_rate=result.win_rate,
                            profit_factor=result.profit_factor,
                            max_drawdown=result.max_drawdown,
                            avg_win=result.avg_win,
                            avg_loss=result.avg_loss,
                            final_capital=initial_capital + result.net_profit
                        )
                        session.add(backtest_run)
                        session.commit()
                        
                        # Save individual trades
                        for trade in result.trades:
                            backtest_trade = BacktestTrade(
                                backtest_run_id=backtest_run.id,
                                entry_date=trade['entry_date'],
                                exit_date=trade['exit_date'],
                                entry_price=trade['entry_price'],
                                exit_price=trade['exit_price'],
                                quantity=trade['quantity'],
                                side=trade['side'],
                                pnl=trade['pnl'],
                                pnl_pct=trade['pnl_pct'],
                                exit_reason=trade['exit_reason']
                            )
                            session.add(backtest_trade)
                        
                        session.commit()
                        st.success("✅ Backtest results saved to database!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save results: {e}")
                        session.rollback()
                
            except Exception as e:
                st.error(f"❌ Backtest failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Historical backtest results
    st.divider()
    st.subheader("📜 Historical Backtest Results")
    
    historical_runs = session.scalars(
        select(BacktestRun).order_by(desc(BacktestRun.created_at)).limit(20)
    ).all()
    
    if historical_runs:
        runs_data = [{
            "Date": r.created_at.strftime("%Y-%m-%d %H:%M"),
            "Strategy": r.strategy_name,
            "Ticker": r.ticker,
            "Period": f"{r.start_date.date()} to {r.end_date.date()}",
            "Trades": r.total_trades,
            "Win Rate": f"{r.win_rate:.1f}%",
            "Net Profit": f"₹{r.net_profit:,.2f}",
            "Return %": f"{(r.net_profit / r.initial_capital) * 100:.2f}%",
            "Profit Factor": f"{r.profit_factor:.2f}"
        } for r in historical_runs]
        
        st.dataframe(pd.DataFrame(runs_data), use_container_width=True)
    else:
        st.info("No historical backtest results. Run a backtest to see results here.")
    
    session.close()

elif page == "Strategies":
    st.title("📚 Trading Strategies")
    st.markdown("**View, understand, and manage all available trading strategies**")
    
    session = get_session()
    
    # Index filter for strategies
    if selected_index:
        st.info(f"📈 Showing strategies for: **{selected_index.display_name}**")
        index_filter = selected_index.id
    else:
        st.info("📈 Showing strategies for: **All Indices**")
        index_filter = None
    
    # Get all available strategies
    from strategies.registry import StrategyRegistry
    registry = StrategyRegistry()
    available_strategies = registry.list_strategies()
    
    if not available_strategies:
        st.warning("No strategies found. Please ensure strategy files are in the strategies/ directory.")
        session.close()
        st.stop()
    
    # Tabs for viewing and managing strategies
    tab1, tab2 = st.tabs(["📖 View Strategies", "➕ Add/Edit Strategy"])
    
    with tab1:
        st.subheader("Available Strategies")
        
        # Get strategy metadata from database, filtered by index if selected
        for strategy_name in available_strategies:
            if index_filter:
                metadata = session.scalar(
                    select(StrategyMetadata).filter_by(
                        strategy_name=strategy_name, 
                        index_id=index_filter,
                        is_active=True
                    )
                )
            else:
                # Show all strategies across all indices
                metadata_list = session.scalars(
                    select(StrategyMetadata).filter_by(
                        strategy_name=strategy_name, 
                        is_active=True
                    )
                ).all()
                metadata = metadata_list[0] if metadata_list else None
            
            # Show index in title if viewing all indices
            title_suffix = f" ({metadata.index.display_name})" if metadata and metadata.index and not index_filter else ""
            with st.expander(f"📊 {metadata.display_name if metadata else strategy_name}{title_suffix}", expanded=False):
                if metadata:
                    # Display full metadata
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Index:** {metadata.index.display_name if metadata.index else 'N/A'}")
                        st.markdown(f"**Category:** {metadata.category}")
                        st.markdown(f"**Risk Level:** {metadata.risk_level}")
                        st.markdown(f"**Recommended Timeframe:** {metadata.recommended_timeframe}")
                        if metadata.author:
                            st.markdown(f"**Author:** {metadata.author}")
                        st.markdown(f"**Version:** {metadata.version}")
                    
                    with col2:
                        if metadata.risk_level == "Low":
                            st.success("🟢 Low Risk")
                        elif metadata.risk_level == "Medium":
                            st.warning("🟡 Medium Risk")
                        elif metadata.risk_level == "High":
                            st.error("🔴 High Risk")
                    
                    st.divider()
                    
                    st.markdown("### Description")
                    st.write(metadata.description)
                    
                    if metadata.how_it_works:
                        st.markdown("### How It Works")
                        st.write(metadata.how_it_works)
                    
                    if metadata.entry_conditions:
                        st.markdown("### Entry Conditions")
                        st.write(metadata.entry_conditions)
                    
                    if metadata.exit_conditions:
                        st.markdown("### Exit Conditions")
                        st.write(metadata.exit_conditions)
                    
                    if metadata.risk_management:
                        st.markdown("### Risk Management")
                        st.write(metadata.risk_management)
                    
                    # Edit button
                    if st.button(f"✏️ Edit {strategy_name}", key=f"edit_{strategy_name}"):
                        st.session_state[f'editing_{strategy_name}'] = True
                        st.rerun()
                else:
                    # No metadata - show basic info
                    st.info(f"Strategy '{strategy_name}' is available but has no documentation yet.")
                    st.write("**Add documentation using the 'Add/Edit Strategy' tab.**")
                    
                    # Try to get basic info from strategy class
                    try:
                        strategy = registry.get_strategy(strategy_name)
                        if strategy:
                            st.write(f"**Type:** {type(strategy).__name__}")
                    except:
                        pass
                    
                    if st.button(f"➕ Add Documentation for {strategy_name}", key=f"add_{strategy_name}"):
                        st.session_state['adding_strategy'] = strategy_name
                        st.session_state['switch_to_tab'] = 2
                        st.rerun()
    
    with tab2:
        st.subheader("Add or Edit Strategy Documentation")
        
        # Strategy selection
        strategy_to_edit = st.selectbox(
            "Select Strategy to Document",
            ["-- New Strategy --"] + available_strategies,
            key="strategy_select"
        )
        
        # Check if we're editing existing
        editing_existing = False
        existing_metadata = None
        
        if strategy_to_edit != "-- New Strategy --":
            existing_metadata = session.scalar(
                select(StrategyMetadata).filter_by(strategy_name=strategy_to_edit)
            )
            if existing_metadata:
                editing_existing = True
        
        # Index selector for strategy
        strategy_indices = session.scalars(select(Index).filter_by(is_active=True).order_by(Index.name)).all()
        index_options = {idx.display_name: idx for idx in strategy_indices}
        
        # Form for strategy documentation
        with st.form("strategy_form", clear_on_submit=False):
            if editing_existing and existing_metadata:
                strategy_name_input = st.text_input("Strategy Name (Internal)", value=existing_metadata.strategy_name, disabled=True)
                display_name = st.text_input("Display Name", value=existing_metadata.display_name or strategy_to_edit)
                # Index selector
                current_index_display = existing_metadata.index.display_name if existing_metadata.index else list(index_options.keys())[0]
                selected_index_display = st.selectbox(
                    "Index",
                    list(index_options.keys()),
                    index=list(index_options.keys()).index(current_index_display) if current_index_display in index_options else 0,
                    key="strategy_index_select"
                )
                selected_strategy_index = index_options[selected_index_display]
                category = st.selectbox(
                    "Category",
                    ["Technical", "Fundamental", "Hybrid", "AI-Based", "Momentum", "Mean Reversion", "Other"],
                    index=["Technical", "Fundamental", "Hybrid", "AI-Based", "Momentum", "Mean Reversion", "Other"].index(existing_metadata.category) if existing_metadata.category in ["Technical", "Fundamental", "Hybrid", "AI-Based", "Momentum", "Mean Reversion", "Other"] else 0
                )
                description = st.text_area("Description", value=existing_metadata.description or "", height=100)
                how_it_works = st.text_area("How It Works", value=existing_metadata.how_it_works or "", height=150)
                entry_conditions = st.text_area("Entry Conditions", value=existing_metadata.entry_conditions or "", height=100)
                exit_conditions = st.text_area("Exit Conditions", value=existing_metadata.exit_conditions or "", height=100)
                risk_management = st.text_area("Risk Management", value=existing_metadata.risk_management or "", height=100)
                recommended_timeframe = st.selectbox(
                    "Recommended Timeframe",
                    ["Day Trading", "Swing Trading", "Position Trading", "Long Term"],
                    index=["Day Trading", "Swing Trading", "Position Trading", "Long Term"].index(existing_metadata.recommended_timeframe) if existing_metadata.recommended_timeframe in ["Day Trading", "Swing Trading", "Position Trading", "Long Term"] else 1
                )
                risk_level = st.selectbox(
                    "Risk Level",
                    ["Low", "Medium", "High"],
                    index=["Low", "Medium", "High"].index(existing_metadata.risk_level) if existing_metadata.risk_level in ["Low", "Medium", "High"] else 1
                )
                author = st.text_input("Author", value=existing_metadata.author or "")
                version = st.text_input("Version", value=existing_metadata.version or "1.0")
            else:
                strategy_name_input = st.text_input("Strategy Name (Internal)", value=strategy_to_edit if strategy_to_edit != "-- New Strategy --" else "")
                display_name = st.text_input("Display Name", value=strategy_to_edit if strategy_to_edit != "-- New Strategy --" else "")
                # Index selector - default to selected index or first available
                default_index_display = selected_index_display if selected_index_display != "All Indices" and selected_index_display in index_options else list(index_options.keys())[0]
                selected_index_display_form = st.selectbox(
                    "Index",
                    list(index_options.keys()),
                    index=list(index_options.keys()).index(default_index_display) if default_index_display in index_options else 0,
                    key="strategy_index_select_new"
                )
                selected_strategy_index = index_options[selected_index_display_form]
                category = st.selectbox(
                    "Category",
                    ["Technical", "Fundamental", "Hybrid", "AI-Based", "Momentum", "Mean Reversion", "Other"]
                )
                description = st.text_area("Description", height=100, placeholder="Brief description of what this strategy does...")
                how_it_works = st.text_area("How It Works", height=150, placeholder="Explain the logic and methodology behind this strategy...")
                entry_conditions = st.text_area("Entry Conditions", height=100, placeholder="When does this strategy generate a BUY signal?")
                exit_conditions = st.text_area("Exit Conditions", height=100, placeholder="When does this strategy exit a position?")
                risk_management = st.text_area("Risk Management", height=100, placeholder="How does this strategy manage risk (stop loss, position sizing, etc.)?")
                recommended_timeframe = st.selectbox(
                    "Recommended Timeframe",
                    ["Day Trading", "Swing Trading", "Position Trading", "Long Term"],
                    index=1  # Default to Swing Trading
                )
                risk_level = st.selectbox(
                    "Risk Level",
                    ["Low", "Medium", "High"],
                    index=1  # Default to Medium
                )
                author = st.text_input("Author", value="")
                version = st.text_input("Version", value="1.0")
            
            submitted = st.form_submit_button("💾 Save Strategy Documentation", use_container_width=True)
            
            if submitted:
                if not strategy_name_input:
                    st.error("Strategy name is required")
                else:
                    try:
                        if editing_existing and existing_metadata:
                            # Update existing
                            existing_metadata.index_id = selected_strategy_index.id
                            existing_metadata.display_name = display_name
                            existing_metadata.category = category
                            existing_metadata.description = description
                            existing_metadata.how_it_works = how_it_works
                            existing_metadata.entry_conditions = entry_conditions
                            existing_metadata.exit_conditions = exit_conditions
                            existing_metadata.risk_management = risk_management
                            existing_metadata.recommended_timeframe = recommended_timeframe
                            existing_metadata.risk_level = risk_level
                            existing_metadata.author = author
                            existing_metadata.version = version
                            existing_metadata.updated_at = datetime.now()
                            st.success(f"✅ Strategy documentation updated for {strategy_name_input}")
                        else:
                            # Create new
                            new_metadata = StrategyMetadata(
                                strategy_name=strategy_name_input,
                                index_id=selected_strategy_index.id,
                                display_name=display_name,
                                category=category,
                                description=description,
                                how_it_works=how_it_works,
                                entry_conditions=entry_conditions,
                                exit_conditions=exit_conditions,
                                risk_management=risk_management,
                                recommended_timeframe=recommended_timeframe,
                                risk_level=risk_level,
                                author=author,
                                version=version
                            )
                            session.add(new_metadata)
                            st.success(f"✅ Strategy documentation added for {strategy_name_input} on {selected_strategy_index.display_name}")
                        
                        session.commit()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                        session.rollback()
        
        # Auto-populate for discovered strategies
        st.divider()
        st.subheader("🔄 Auto-Populate from Discovered Strategies")
        
        undiscovered = [s for s in available_strategies if not session.scalar(
            select(StrategyMetadata).filter_by(strategy_name=s)
        )]
        
        if undiscovered:
            st.info(f"Found {len(undiscovered)} strategies without documentation:")
            for strategy in undiscovered:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"- **{strategy}**")
                with col2:
                    if st.button("📝 Add Now", key=f"quick_add_{strategy}"):
                        # Pre-fill form with strategy name
                        st.session_state['strategy_select'] = strategy
                        st.rerun()
        else:
            st.success("✅ All discovered strategies have documentation!")
    
    session.close()

elif page == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("System Status")
    
    session = get_session()
    
    # Database info
    total_companies = session.scalar(select(func.count(CompanyProfile.ticker)))
    total_market_data = session.scalar(select(func.count(MarketData.id)))
    
    st.metric("Total Companies", total_companies)
    st.metric("Total Market Data Records", total_market_data)
    
    # Quick Actions
    st.subheader("Quick Actions")
    st.info("💡 Use the **Control Center** page (in the sidebar) to execute all workflows and control the system.")
    
    st.markdown("**Navigate to Control Center from the sidebar menu above** ⬆️")
    
    st.divider()
    
    # System Information
    st.subheader("System Information")
    
    # Latest data timestamps
    latest_market_data = session.scalar(
        select(func.max(MarketData.date))
    )
    if latest_market_data:
        st.write(f"**Latest Market Data**: {latest_market_data.strftime('%Y-%m-%d %H:%M')}")
    
    latest_prediction = session.scalar(
        select(func.max(AIPredictions.generated_at))
    )
    if latest_prediction:
        st.write(f"**Latest AI Prediction**: {latest_prediction.strftime('%Y-%m-%d %H:%M')}")
    
    latest_signal = session.scalar(
        select(func.max(TradeSignal.created_at))
    )
    if latest_signal:
        st.write(f"**Latest Trade Signal**: {latest_signal.strftime('%Y-%m-%d %H:%M')}")
    
    session.close()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Nifty 50 AI Swing Trader**")
st.sidebar.markdown("Version 0.1.0")

