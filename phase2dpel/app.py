# app.py
# Live Multi-Crypto Dashboard using Binance Public API (no API key)
# Features: Indicators (RSI, MACD, EMA, Bollinger), Multi-model ML (LR, RF, XGBoost optional),
# Trend classification (Logistic), Profit/Loss, Risk Score, 60s auto-refresh.

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import warnings

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# optional xgboost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Binance Live Crypto Dashboard", layout="wide")
# auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="binance_refresh")

st.title("🔶 Binance Live Multi-Crypto Dashboard (No API Key)")
st.markdown("Live data from Binance public API (klines). NOT using any API key or account data.")

# ----------------------------
# Supported coins (USDT pairs)
# ----------------------------
BINANCE_PAIRS = {
    "Bitcoin (BTC/USDT)": "BTCUSDT",
    "Ethereum (ETH/USDT)": "ETHUSDT",
    "Tether (USDT) - placeholder": "USDTUSDT",  # not tradable, keep for UI if needed
    "Binance Coin (BNB/USDT)": "BNBUSDT",
    "Dogecoin (DOGE/USDT)": "DOGEUSDT",
    "Polygon (MATIC/USDT)": "MATICUSDT",
    "Ripple (XRP/USDT)": "XRPUSDT",
    "Cardano (ADA/USDT)": "ADAUSDT",
    "Solana (SOL/USDT)": "SOLUSDT",
    "Polkadot (DOT/USDT)": "DOTUSDT",
    "Bitcoin Cash (BCH/USDT)": "BCHUSDT",
    "Litecoin (LTC/USDT)": "LTCUSDT",
    "Avalanche (AVAX/USDT)": "AVAXUSDT"
}

with st.sidebar:
    st.header("Configuration")
    selected = st.multiselect("Select pairs:", list(BINANCE_PAIRS.keys()),
                              default=["Bitcoin (BTC/USDT)", "Ethereum (ETH/USDT)"])
    days = st.selectbox("History window (days):", [1, 7, 30, 90, 180, 365], index=2)
    cand_interval = st.selectbox("Candle interval:", ["1m", "5m", "15m", "1h", "4h", "1d"], index=5)
    models_choice = st.multiselect("Models to compare (price):", ["Linear Regression", "Random Forest", "XGBoost"], default=["Linear Regression", "Random Forest"])
    st.markdown("XGBoost will be used only if installed on host.")
    st.sidebar.markdown("Refresh is automatic every 60s.")

if not selected:
    st.warning("Select at least one pair.")
    st.stop()

# ----------------------------
# Helpers: Binance klines fetch
# ----------------------------
@st.cache_data(ttl=60)
def fetch_binance_klines(symbol: str, interval: str, limit: int = 500):
    """
    Fetch klines from Binance public API.
    symbol: e.g., 'BTCUSDT'
    interval: '1d', '1h', etc.
    limit: number of candles (max 1000 for Binance)
    """
    base = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(base, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # kline format: [ openTime, open, high, low, close, volume, closeTime, ... ]
    df = pd.DataFrame(data, columns=[
        'open_time','open','high','low','close','volume','close_time',
        'quote_asset_volume','number_of_trades','taker_buy_base','taker_buy_quote','ignore'
    ])
    df['Date'] = pd.to_datetime(df['close_time'], unit='ms')
    df['Price'] = df['close'].astype(float)
    df = df[['Date', 'Price', 'open','high','low','volume']].copy()
    df[['open','high','low','volume']] = df[['open','high','low','volume']].astype(float)
    return df

# compute limit from days & interval
def estimate_limit(days: int, interval: str):
    """Estimate approx number of candles to request from Binance for the timeframe"""
    if interval.endswith('m'):
        minutes = int(interval[:-1])
        candles_per_day = 24 * 60 // minutes
    elif interval.endswith('h'):
        hours = int(interval[:-1])
        candles_per_day = 24 // hours
    elif interval.endswith('d'):
        days_interval = int(interval[:-1])
        candles_per_day = 1 // days_interval if days_interval > 1 else 1
        # For daily, 1 per day
        if days_interval == 1:
            candles_per_day = 1
    else:
        candles_per_day = 1
    # safeguard and min limit
    limit = max(30, min(1000, candles_per_day * max(1, days)))
    return int(limit)

# ----------------------------
# Indicators
# ----------------------------
def add_indicators(df: pd.DataFrame):
    if df.empty:
        return df
    df = df.copy().reset_index(drop=True)
    # MA
    df['MA7'] = df['Price'].rolling(7, min_periods=1).mean()
    df['MA14'] = df['Price'].rolling(14, min_periods=1).mean()
    # Momentum
    df['Momentum'] = df['Price'] - df['Price'].shift(1)
    # RSI (14)
    delta = df['Price'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # EMA20
    df['EMA20'] = df['Price'].ewm(span=20, adjust=False).mean()
    # MACD
    ema12 = df['Price'].ewm(span=12, adjust=False).mean()
    ema26 = df['Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    df['BB_MID'] = df['Price'].rolling(window=20, min_periods=20).mean()
    df['BB_STD'] = df['Price'].rolling(window=20, min_periods=20).std()
    df['BB_UPPER'] = df['BB_MID'] + (df['BB_STD'] * 2)
    df['BB_LOWER'] = df['BB_MID'] - (df['BB_STD'] * 2)
    # Trend label (next candle up/down)
    df['Trend'] = (df['Price'].shift(-1) > df['Price']).astype(int)
    return df

# ----------------------------
# Risk & Profit
# ----------------------------
def get_risk_score(df: pd.DataFrame):
    if df.empty or len(df) < 2:
        return "UNKNOWN"
    vol = df['Price'].pct_change().dropna().std()
    if vol < 0.02:
        return "LOW"
    elif vol < 0.05:
        return "MEDIUM"
    else:
        return "HIGH"

def profit_loss_pct(current_price, predicted_price):
    if current_price is None or predicted_price is None or current_price == 0:
        return None
    return round(((predicted_price - current_price) / current_price) * 100, 2)

# ----------------------------
# Multi-model trainer & predictor
# ----------------------------
def model_compare(X_train, y_train, X_test, selected_models):
    preds = {}
    trained = {}
    if "Linear Regression" in selected_models:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        preds["Linear Regression"] = lr.predict(X_test)
        trained["Linear Regression"] = lr
    if "Random Forest" in selected_models:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        preds["Random Forest"] = rf.predict(X_test)
        trained["Random Forest"] = rf
    if "XGBoost" in selected_models:
        if XGB_AVAILABLE:
            xb = XGBRegressor(n_estimators=200, learning_rate=0.1, verbosity=0, random_state=42)
            xb.fit(X_train, y_train)
            preds["XGBoost"] = xb.predict(X_test)
            trained["XGBoost"] = xb
        else:
            preds["XGBoost"] = None
    return preds, trained

# ----------------------------
# Fetch & process selected coins
# ----------------------------
coin_data = {}
st.info("Fetching data from Binance public API...")
fetch_progress = st.empty()
for idx, display_name in enumerate(selected, start=1):
    symbol = BINANCE_PAIRS.get(display_name)
    if symbol is None:
        continue
    limit = estimate_limit(days, cand_interval)
    try:
        df = fetch_binance_klines(symbol, cand_interval, limit=limit)
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        df = pd.DataFrame(columns=['Date', 'Price'])
    df = add_indicators(df)
    coin_data[display_name] = df
    fetch_progress.info(f"Fetched {idx}/{len(selected)}: {display_name} ({symbol}), candles={len(df)}")

fetch_progress.empty()
st.success("Data fetched & indicators computed.")

# ----------------------------
# Top chart: multi-coin price lines
# ----------------------------
st.markdown("## 📈 Price Chart")
fig = go.Figure()
for name, df in coin_data.items():
    if df.empty:
        continue
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name=name))
fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price (USDT)")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Results table
# ----------------------------
st.markdown("## 🔮 Predictions & Signals")
results = []

for name, df in coin_data.items():
    if df.empty or len(df) < 30:
        results.append({
            "Coin": name,
            "Predicted Price": None,
            "Predicted Trend": None,
            "Signal": "NO_DATA",
            "MAE": None,
            "Accuracy": None,
            "Profit (%)": None,
            "Risk": get_risk_score(df),
            "Model Predictions": {}
        })
        continue

    df_model = df.copy().reset_index(drop=True)
    features = ['Price', 'MA7', 'MA14', 'Momentum', 'RSI', 'EMA20', 'MACD', 'MACD_SIGNAL']
    for f in features:
        if f not in df_model.columns:
            df_model[f] = np.nan
    df_model = df_model.dropna(subset=features + ['Trend'], how='any').reset_index(drop=True)
    if len(df_model) < 10:
        results.append({
            "Coin": name,
            "Predicted Price": None,
            "Predicted Trend": None,
            "Signal": "INSUFFICIENT_DATA",
            "MAE": None,
            "Accuracy": None,
            "Profit (%)": None,
            "Risk": get_risk_score(df),
            "Model Predictions": {}
        })
        continue

    # Prepare X, y
    X = df_model[features]
    y_price = df_model['Price'].shift(-1).dropna()
    X = X.iloc[:-1]
    y_trend = df_model['Trend'][:-1]

    # train-test split (time series style)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train_price = y_price.iloc[:split_idx]
    y_test_price = y_price.iloc[split_idx:]
    y_train_trend = y_trend.iloc[:split_idx]
    y_test_trend = y_trend.iloc[split_idx:]

    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # baseline models
    lr = LinearRegression()
    lr.fit(X_train_s, y_train_price)
    price_pred_lr = lr.predict(X_test_s)

    log = LogisticRegression(max_iter=200)
    log.fit(X_train_s, y_train_trend)
    trend_pred_log = log.predict(X_test_s)

    mae_lr = float(np.mean(np.abs(y_test_price - price_pred_lr)))
    acc_log = float(np.mean(y_test_trend == trend_pred_log))

    # multi-model comparison
    preds_dict, trained_models = model_compare(X_train_s, y_train_price, X_test_s, models_choice)

    # get last predicted value from each model's test predictions
    model_latest_preds = {}
    for m, arr in preds_dict.items():
        if arr is None:
            model_latest_preds[m] = None
        else:
            try:
                model_latest_preds[m] = float(arr[-1])
            except Exception:
                model_latest_preds[m] = None

    # choose primary predicted price (prefer Linear)
    primary_pred = model_latest_preds.get("Linear Regression") or model_latest_preds.get("Random Forest") or model_latest_preds.get("XGBoost")

    trend_latest = int(trend_pred_log[-1]) if len(trend_pred_log) > 0 else None
    signal = "Buy" if trend_latest == 1 else "Sell"

    current_price = df['Price'].iloc[-1]
    profit_pct = profit_loss_pct(current_price, primary_pred) if primary_pred is not None else None

    results.append({
        "Coin": name,
        "Predicted Price": round(primary_pred, 6) if primary_pred is not None else None,
        "Predicted Trend": "UP" if trend_latest == 1 else ("DOWN" if trend_latest == 0 else None),
        "Signal": signal if primary_pred is not None else "NO_PRED",
        "MAE": round(mae_lr, 6),
        "Accuracy": f"{acc_log*100:.2f}%",
        "Profit (%)": f"{profit_pct}%" if profit_pct is not None else None,
        "Risk": get_risk_score(df),
        "Model Predictions": model_latest_preds
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df)

# ----------------------------
# Detailed coin views
# ----------------------------
st.markdown("## 🔍 Detailed Coin Views")
for name, df in coin_data.items():
    st.subheader(name)
    if df.empty:
        st.write("No data.")
        continue

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name='Price'))
        if 'EMA20' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], mode='lines', name='EMA20'))
        if 'BB_UPPER' in df.columns and not df['BB_UPPER'].isna().all():
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_UPPER'], mode='lines', name='BB_UPPER', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_LOWER'], mode='lines', name='BB_LOWER', line=dict(dash='dash')))
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Price (USDT)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        last = df.iloc[-1]
        st.write(f"- Price: ${last['Price']:.6f}")
        if not np.isnan(last.get('RSI', np.nan)):
            st.write(f"- RSI: {last['RSI']:.2f}")
        if not np.isnan(last.get('MACD', np.nan)):
            st.write(f"- MACD: {last['MACD']:.6f}")
        st.write(f"- Risk: {get_risk_score(df)}")
        st.write("---")
        st.line_chart(df['Price'].tail(50).reset_index(drop=True))

    st.write("Indicators (last 10 rows):")
    st.dataframe(df.tail(10).reset_index(drop=True))

# ----------------------------
# Footer / Notes
# ----------------------------
st.markdown("## 📝 Notes & Deployment")
st.markdown("""
- This app uses Binance public API `GET /api/v3/klines` (no API key).  
- Binance rate limits exist (e.g., weight limits). For many users, cache server-side or use a paid data feed.  
- XGBoost is optional — include `xgboost` in requirements if you want it.  
- This is for educational/demo purposes — not financial advice.
""")

st.markdown("### Quick run")
st.code("pip install -r requirements.txt\nstreamlit run app.py")
