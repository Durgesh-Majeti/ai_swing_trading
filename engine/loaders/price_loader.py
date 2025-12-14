"""
DEPRECATED: This module is deprecated. Use engine.etl.ETLModule instead.
This file is kept for backward compatibility only.
"""

import yfinance as yf
from database.models import SessionLocal, CompanyProfile, MarketData
from sqlalchemy import select
from loguru import logger
import pandas as pd
import json
from datetime import datetime

def sync_price_history():
    """
    DEPRECATED: Use engine.etl.ETLModule.sync_market_data() instead.
    This function is kept for backward compatibility only.
    """
    logger.warning("⚠️  sync_price_history() is deprecated. Use engine.etl.ETLModule.sync_market_data() instead.")
    # #region agent log
    log_path = r"d:\Python Projects\Indian Stock Analysis\.cursor\debug.log"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "ALL", "location": "price_loader.py:8", "message": "sync_price_history entry", "data": {}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
    # #endregion
    session = SessionLocal()
    
    # 1. Get all tickers from OUR database (The Source of Truth)
    companies = session.scalars(select(CompanyProfile)).all()
    tickers = [c.ticker for c in companies]
    
    logger.info(f"📈 Syncing Price History for {len(tickers)} companies...")
    # #region agent log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "ALL", "location": "price_loader.py:16", "message": "tickers list", "data": {"ticker_count": len(tickers), "tickers": tickers[:3] if tickers else []}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
    # #endregion

    for ticker in tickers:
        try:
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "ALL", "location": "price_loader.py:20", "message": "before yf.download", "data": {"ticker": ticker}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
            # #endregion
            # Fetch 1 Year of data
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "price_loader.py:25", "message": "after yf.download", "data": {"ticker": ticker, "df_empty": df.empty, "df_shape": list(df.shape) if not df.empty else None, "df_columns": list(df.columns) if not df.empty else None, "df_index_type": str(type(df.index[0])) if not df.empty and len(df.index) > 0 else None}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
            # #endregion
            
            if df.empty: 
                # #region agent log
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H4", "location": "price_loader.py:28", "message": "df.empty is True", "data": {"ticker": ticker}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                continue

            # Bulk Insert Logic (Optimization)
            # We check the latest date in DB to avoid re-downloading old data
            stmt = select(MarketData).filter_by(ticker=ticker).order_by(MarketData.date.desc())
            last_entry = session.scalars(stmt).first()
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "price_loader.py:35", "message": "last_entry check", "data": {"ticker": ticker, "has_last_entry": last_entry is not None, "last_entry_date": str(last_entry.date) if last_entry else None, "last_entry_date_type": str(type(last_entry.date)) if last_entry else None}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
            # #endregion
            
            new_records = []
            for date, row in df.iterrows():
                # #region agent log
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1,H3", "location": "price_loader.py:40", "message": "before date comparison", "data": {"ticker": ticker, "date": str(date), "date_type": str(type(date)), "date_tz": str(date.tz) if hasattr(date, 'tz') else None, "last_entry_date": str(last_entry.date) if last_entry else None, "last_entry_date_type": str(type(last_entry.date)) if last_entry else None}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                # If we have data, only add rows NEWER than what we have
                if last_entry and date <= last_entry.date:
                    # #region agent log
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1,H3", "location": "price_loader.py:45", "message": "date comparison true, skipping", "data": {"ticker": ticker, "date": str(date), "last_entry_date": str(last_entry.date)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                    # #endregion
                    continue
                
                # #region agent log
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2,H4", "location": "price_loader.py:49", "message": "before column access", "data": {"ticker": ticker, "row_keys": list(row.index) if hasattr(row, 'index') else None, "row_type": str(type(row))}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                record = MarketData(
                    ticker=ticker,
                    date=date,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                )
                new_records.append(record)

            if new_records:
                # #region agent log
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "price_loader.py:62", "message": "before session.add_all", "data": {"ticker": ticker, "new_records_count": len(new_records)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                session.add_all(new_records)
                session.commit()
                # #region agent log
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "price_loader.py:65", "message": "after session.commit", "data": {"ticker": ticker, "new_records_count": len(new_records)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
                # #endregion
                logger.success(f"updated {ticker} (+{len(new_records)} days)")
            else:
                logger.info(f"skipping {ticker} (Up to date)")

        except Exception as e:
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "ALL", "location": "price_loader.py:72", "message": "exception caught", "data": {"ticker": ticker, "exception_type": str(type(e).__name__), "exception_message": str(e)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
            # #endregion
            logger.error(f"Failed {ticker}: {e}")
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "price_loader.py:75", "message": "before session.rollback", "data": {"ticker": ticker}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
            # #endregion
            session.rollback()
            # #region agent log
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "price_loader.py:77", "message": "after session.rollback", "data": {"ticker": ticker}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
            # #endregion

    session.close()
    # #region agent log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "ALL", "location": "price_loader.py:81", "message": "sync_price_history exit", "data": {}, "timestamp": int(datetime.now().timestamp() * 1000)}) + "\n")
    # #endregion