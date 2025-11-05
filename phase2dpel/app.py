# app.py
# Live Multi-Crypto Dashboard (CoinGecko) with many indicators, UI tabs, news & sentiment (VADER), candlesticks, alerts
# Ready for Streamlit Cloud

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import math
import plotly.graph_objs as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from nltk import download as nltk_download

# --- NLTK VADER setup (for simple sentiment) ---
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except Exception:
    nltk_download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

# Streamlit config
st.set_page_config(page_title="Advanced Crypto Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Advanced Multi-Crypto Dashboard (CoinGecko)")

# ---------------------
# Sidebar: config
# ---------------------
with st.sidebar:
    st.header("Configuration")
    # 20+ coin options (user requested at least 15)
    coins_dict = {
        "Bitcoin (BTC)": "bitcoin",
        "Ethereum (ETH)": "ethereum",
        "Tether (USDT)": "tether",
        "BNB (BNB)": "binancecoin",
        "XRP (XRP)": "ripple",
        "Cardano (ADA)": "cardano",
        "Solana (SOL)": "solana",
        "Dogecoin (DOGE)": "dogecoin",
        "Polygon (MATIC)": "matic-network",
        "Polkadot (DOT)": "polkadot",
        "Litecoin (LTC)": "litecoin",
        "Avalanche (AVAX)": "avalanche-2",
        "Chainlink (LINK)": "chainlink",
        "Stellar (XLM)": "stellar",
        "Uniswap (UNI)": "uniswap",
        "Aptos (APT)": "aptos",
        "Tron (TRX)": "tron",
        "Shiba Inu (SHIB)": "shiba-inu",
        "Bitcoin Cash (BCH)": "bitcoin-cash",
        "Cosmos (ATOM)": "cosmos",
        "Algorand (ALGO)": "algorand",
        "VeChain (VET)": "vechain"
    }

    selected = st.multiselect("Select cryptocurrencies (min 1):", list(coins_dict.keys()),
                              default=["Bitcoin (BTC)", "Ethereum (ETH)", "Solana (SOL)"])
    days = st.selectbox("History window (days):", [7, 30, 60, 90, 180, 365], index=1)
    timeframe = st.selectbox("Chart timeframe (granularity):", ["1d", "12h", "6h", "1h"], index=0)
    auto_refresh = st.checkbox("Auto-refresh data every 60s", value=True)
    theme_choice = st.radio("Theme:", ["Light", "Dark"], index=0)
    st.markdown("---")
    st.markdown("Indicators:")
    show_rsi = st.checkbox("RSI (14)", True)
    show_macd = st.checkbox("MACD (12,26,9)", True)
    show_bbands = st.checkbox("Bollinger Bands", True)
    show_ema = st.checkbox("EMA 12/26", True)
    show_stoch = st.checkbox("Stochastic Oscillator", True)
    show_vwap = st.checkbox("VWAP", True)
    show_vol = st.checkbox("Volatility (rolling)", True)
    st.markdown("---")
    st.markdown("Models & Alerts:")
    enable_ml = st.checkbox("Enable ML Predictions", True)
    enable_rf = st.checkbox("Include RandomForest in comparison", True)
    enable_alerts = st.checkbox("Enable Alerts (MA & MACD cross)", True)

# Theme (simple CSS)
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .reportview-container {background: #0e1117;}
        .sidebar .sidebar-content {background: #0e1117;}
        .css-1d391kg {color: #E8EDEE;}
        </style>
        """, unsafe_allow_html=True)

# Auto-refresh mechanism
if auto_refresh:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="auto_refresh")

# ---------------------
# Utilities & Data fetching (CoinGecko)
# ---------------------
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

@st.cache_data(ttl=60)
def fetch_market_chart(coin_id: str, days: int):
    # CoinGecko offers prices endpoint (timestamps, price)
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    # data['prices'] = [ [ts, price], ... ]
    df = pd.DataFrame(data["prices"], columns=["timestamp", "Price"])
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["Date", "Price"]]
    # If volumes available, include them
    if "total_volumes" in data:
        vol_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "Volume"])
        vol_df["Date"] = pd.to_datetime(vol_df["timestamp"], unit="ms")
        vol_df = vol_df[["Date", "Volume"]]
        df = df.merge(vol_df, on="Date", how="left")
    else:
        df["Volume"] = np.nan
    return df

@st.cache_data(ttl=900)
def fetch_coingecko_news(coin_id: str):
    # CoinGecko status updates (news-like)
    url = f"{COINGECKO_BASE}/coins/{coin_id}/status_updates"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("status_updates", [])

@st.cache_data(ttl=600)
def fetch_fear_greed():
    # alternative.me F&G index
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        j = r.json()
        item = j["data"][0]
        return int(item["value"]), item["value_classification"]
    except Exception:
        return None, None

# ---------------------
# Indicator functions
# ---------------------
def add_indicators(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)
    # EMA12 & EMA26
    df["EMA12"] = df["Price"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Price"].ewm(span=26, adjust=False).mean()
    # MA7/14 already small windows for smoothing
    df["MA7"] = df["Price"].rolling(7, min_periods=1).mean()
    df["MA14"] = df["Price"].rolling(14, min_periods=1).mean()
    # Momentum
    df["Momentum"] = df["Price"] - df["Price"].shift(1)
    # RSI 14
    delta = df["Price"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=14).mean()
    roll_down = down.rolling(14, min_periods=14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    df["BB_MID"] = df["Price"].rolling(window=20, min_periods=20).mean()
    df["BB_STD"] = df["Price"].rolling(window=20, min_periods=20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]
    # Stochastic %K and %D (14,3)
    low14 = df["Price"].rolling(window=14, min_periods=14).min()
    high14 = df["Price"].rolling(window=14, min_periods=14).max()
    df["%K"] = 100 * (df["Price"] - low14) / (high14 - low14)
    df["%D"] = df["%K"].rolling(window=3, min_periods=3).mean()
    # VWAP (approx using Price*Volume)
    if "Volume" in df.columns and not df["Volume"].isna().all():
        pv = df["Price"] * df["Volume"]
        df["VWAP"] = pv.cumsum() / df["Volume"].cumsum().replace(0, np.nan)
    else:
        df["VWAP"] = np.nan
    # Volatility (rolling std of returns)
    df["Returns"] = df["Price"].pct_change()
    df["Volatility30"] = df["Returns"].rolling(window=30, min_periods=5).std() * np.sqrt(30)
    # Trend label
    df["Trend"] = (df["Price"].shift(-1) > df["Price"]).astype(int)
    return df

# ---------------------
# ML helpers
# ---------------------
def run_models(df: pd.DataFrame, include_rf=True):
    # Prepare
    df_m = df.dropna().reset_index(drop=True)
    features = ["Price", "MA7", "MA14", "Momentum", "RSI", "EMA12", "EMA26", "MACD", "MACD_SIGNAL"]
    for f in features:
        if f not in df_m.columns:
            df_m[f] = np.nan
    df_m = df_m.dropna(subset=features + ["Trend"], how="any").reset_index(drop=True)
    if len(df_m) < 10:
        return None
    X = df_m[features]
    y_price = df_m["Price"].shift(-1).dropna()
    X = X.iloc[:-1]
    y_trend = df_m["Trend"][:-1]
    # time-series split
    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]; X_test = X.iloc[split:]
    y_train_price = y_price.iloc[:split]; y_test_price = y_price.iloc[split:]
    y_train_trend = y_trend.iloc[:split]; y_test_trend = y_trend.iloc[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    # Linear regression for price
    lr = LinearRegression(); lr.fit(X_train_s, y_train_price)
    price_pred_lr = lr.predict(X_test_s)
    # Logistic for trend
    log = LogisticRegression(max_iter=200); log.fit(X_train_s, y_train_trend)
    trend_pred_log = log.predict(X_test_s)
    mae = float(np.mean(np.abs(y_test_price - price_pred_lr)))
    acc = float(np.mean(y_test_trend == trend_pred_log))
    # Random forest optional
    rf_pred = None
    if include_rf:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_s, y_train_price)
        rf_pred = rf.predict(X_test_s)
    return {
        "lr_model": lr,
        "log_model": log,
        "scaler": scaler,
        "X_test_s": X_test_s,
        "price_pred_lr": price_pred_lr,
        "rf_pred": rf_pred,
        "mae": mae,
        "acc": acc,
        "y_test_price": y_test_price,
        "y_test_trend": y_test_trend
    }

# ---------------------
# Alerts helper
# ---------------------
def check_alerts(df: pd.DataFrame):
    alerts = []
    # MA cross (Price crosses MA14)
    if "MA14" in df.columns:
        if len(df) >= 2:
            p_prev, p_now = df["Price"].iloc[-2], df["Price"].iloc[-1]
            ma_prev, ma_now = df["MA14"].iloc[-2], df["MA14"].iloc[-1]
            # Cross above
            if p_prev < ma_prev and p_now > ma_now:
                alerts.append(("MA14 Cross", "Price crossed above MA14 (Bullish)"))
            if p_prev > ma_prev and p_now < ma_now:
                alerts.append(("MA14 Cross", "Price crossed below MA14 (Bearish)"))
    # MACD crossover
    if "MACD" in df.columns and "MACD_SIGNAL" in df.columns and len(df) >= 2:
        macd_prev, macd_now = df["MACD"].iloc[-2], df["MACD"].iloc[-1]
        sig_prev, sig_now = df["MACD_SIGNAL"].iloc[-2], df["MACD_SIGNAL"].iloc[-1]
        if macd_prev < sig_prev and macd_now > sig_now:
            alerts.append(("MACD Cross", "MACD crossed above signal (Bullish)"))
        if macd_prev > sig_prev and macd_now < sig_now:
            alerts.append(("MACD Cross", "MACD crossed below signal (Bearish)"))
    return alerts

# ---------------------
# Main UI: Tabs
# ---------------------
tabs = st.tabs(["Overview", "Indicators", "Predictions", "Signals", "News"])
fg_value, fg_type = fetch_fear_greed()

# Preload data for selected coins
data_store = {}
st.info("Fetching market data (CoinGecko)...")
progress = st.empty()
for i, coin_name in enumerate(selected, start=1):
    coin_id = coins_dict[coin_name]
    df = fetch_market_chart(coin_id, days)
    df = add_indicators(df)
    data_store[coin_name] = df
    progress.info(f"Fetched {i}/{len(selected)}: {coin_name} ({len(df)} rows)")
progress.empty()

# OVERVIEW TAB
with tabs[0]:
    st.header("Overview")
    cols = st.columns(3)
    # Summary metrics for each coin
    for coin_name, df in data_store.items():
        if df.empty: 
            continue
        last = df.iloc[-1]
        change_24h = None
        if len(df) > 1:
            prev = df["Price"].iloc[-2]
            change_24h = (last["Price"] - prev) / prev * 100 if prev != 0 else None
        cols[0].metric(coin_name, f"${last['Price']:.4f}", f"{change_24h:.2f}%" if change_24h is not None else None)
    st.markdown("---")
    st.write(f"Fear & Greed index: **{fg_value}** — {fg_type}" if fg_value is not None else "Fear & Greed unavailable")

# INDICATORS TAB
with tabs[1]:
    st.header("Indicators & Candles")
    for coin_name, df in data_store.items():
        st.subheader(coin_name)
        if df.empty:
            st.write("No data.")
            continue
        # Candlestick (we approximate OHLC from prices by creating small ranges)
        # CoinGecko only provides prices array; for candlestick we can synthesize simple OHLC by grouping daily
        # For simplicity use price points as close prices and compute rolling high/low
        window = 24 if timeframe == "1d" else 6
        df_candle = df.copy()
        df_candle["Open"] = df_candle["Price"].shift(1).fillna(df_candle["Price"])
        df_candle["High"] = df_candle["Price"].rolling(window=window, min_periods=1).max()
        df_candle["Low"] = df_candle["Price"].rolling(window=window, min_periods=1).min()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_candle["Date"],
                                     open=df_candle["Open"],
                                     high=df_candle["High"],
                                     low=df_candle["Low"],
                                     close=df_candle["Price"],
                                     name="OHLC"))
        if show_ema and "EMA12" in df_candle.columns:
            fig.add_trace(go.Scatter(x=df_candle["Date"], y=df_candle["EMA12"], name="EMA12"))
            fig.add_trace(go.Scatter(x=df_candle["Date"], y=df_candle["EMA26"], name="EMA26"))
        if show_bbands and "BB_UPPER" in df_candle.columns:
            fig.add_trace(go.Scatter(x=df_candle["Date"], y=df_candle["BB_UPPER"], name="BB_UPPER", line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df_candle["Date"], y=df_candle["BB_LOWER"], name="BB_LOWER", line=dict(dash='dash')))
        fig.update_layout(height=450, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Indicator mini-panels
        i1, i2, i3 = st.columns(3)
        last = df.iloc[-1]
        if show_rsi:
            i1.metric("RSI (14)", f"{last.get('RSI', np.nan):.2f}")
        if show_vwap:
            i2.metric("VWAP", f"{last.get('VWAP', np.nan):.4f}")
        if show_vol:
            i3.metric("30d Volatility", f"{last.get('Volatility30', np.nan):.4f}")

        # show indicator tables
        st.dataframe(df.tail(8)[["Date","Price","EMA12","EMA26","RSI","MACD","MACD_SIGNAL","BB_UPPER","BB_LOWER","VWAP"]].reset_index(drop=True))

# PREDICTIONS TAB
with tabs[2]:
    st.header("Predictions (ML)")
    for coin_name, df in data_store.items():
        st.subheader(coin_name)
        model_res = run_models(df, include_rf=enable_rf)
        if model_res is None:
            st.write("Insufficient data for ML models.")
            continue
        # get latest preds
        lr_pred_latest = float(model_res["price_pred_lr"][-1])
        rf_pred_latest = float(model_res["rf_pred"][-1]) if model_res["rf_pred"] is not None else None
        acc = model_res["acc"]
        mae = model_res["mae"]
        # trend from logistic
        trend = int(model_res["log_model"].predict(model_res["X_test_s"])[-1])
        signal = "BUY" if trend == 1 else "SELL"
        cols = st.columns(4)
        cols[0].metric("LinearReg Pred", f"${lr_pred_latest:.4f}")
        cols[1].metric("RandomForest Pred", f"${rf_pred_latest:.4f}" if rf_pred_latest is not None else "N/A")
        cols[2].metric("MAE (LR)", f"{mae:.4f}")
        cols[3].metric("Trend Acc", f"{acc*100:.2f}%")
        st.write(f"Signal: **{signal}**")
        # profit estimate vs current
        current = df["Price"].iloc[-1]
        profit_lr = (lr_pred_latest - current) / current * 100
        st.write(f"Estimated next-step profit (LR): {profit_lr:.2f}% (current ${current:.4f})")

# SIGNALS TAB
with tabs[3]:
    st.header("Signals & Alerts")
    for coin_name, df in data_store.items():
        st.subheader(coin_name)
        alerts = check_alerts(df) if enable_alerts else []
        if alerts:
            for a in alerts:
                st.warning(f"{a[0]} — {a[1]}")
        else:
            st.write("No alerts detected for latest candle.")

# NEWS TAB
with tabs[4]:
    st.header("News & Sentiment (CoinGecko status updates)")
    for coin_name in selected:
        st.subheader(coin_name)
        coin_id = coins_dict[coin_name]
        updates = fetch_coingecko_news(coin_id)
        if not updates:
            st.write("No recent updates.")
            continue
        for u in updates[:5]:
            title = u.get("title") or u.get("description")[:80]
            desc = u.get("description") or ""
            created = u.get("created_at", "")
            st.markdown(f"**{title}** — _{created}_")
            st.write(desc)
            # simple sentiment on description
            s = sia.polarity_scores(desc or title)
            st.write(f"Sentiment: pos {s['pos']:.2f} | neu {s['neu']:.2f} | neg {s['neg']:.2f} | comp {s['compound']:.2f}")
            st.markdown("---")

# Footer / disclaimers
st.markdown("---")
st.markdown("**Notes:** Demo dashboard for educational purposes. Not financial advice. CoinGecko free API used; respect rate-limits for heavy use. For production, consider server-side caching and paid data provider.")
