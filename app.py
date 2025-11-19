# ==========================================
#   AI TRADING BOT – STREAMLIT CLOUD SAFE
#   Interface Black | Atualização Live 30s
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Trading Bot", layout="wide")

st.markdown("""
<style>
body { background-color: #000; color: #fff; }
.metric-card { background: linear-gradient(90deg, #0f172a, #001219); padding: 12px; border-radius: 12px; }
.signal-up { background: linear-gradient(90deg,#04290f,#058a1f); color: white; padding:14px; border-radius:10px; font-weight:700; }
.signal-down { background: linear-gradient(90deg,#2b0606,#8b0000); color: white; padding:14px; border-radius:10px; font-weight:700; }
.small-muted { color: #94a3b8; font-size:12px }
</style>
""", unsafe_allow_html=True)

# -------------- Exchanges --------------
EXCHANGES = {
    "Binance": ccxt.binance({'enableRateLimit': True}),
    "KuCoin": ccxt.kucoin({'enableRateLimit': True}),
    "Kraken": ccxt.kraken({'enableRateLimit': True}),
    "Coinbase": ccxt.coinbase({'enableRateLimit': True}),
}

PAIRS = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "EUR/USD", "GBP/USD", "XAU/USD"]
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h"]

# --------------- Helpers ----------------
@st.cache_data(ttl=20)
def fetch_data(_exchange, symbol, timeframe):
    try:
        df = _exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
        df = pd.DataFrame(df, columns=["time","open","high","low","close","volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except:
        for name, ex in EXCHANGES.items():
            if ex != _exchange:
                try:
                    df = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
                    df = pd.DataFrame(df, columns=["time","open","high","low","close","volume"])
                    df["time"] = pd.to_datetime(df["time"], unit="ms")
                    st.info(f"Fallback para: {name}")
                    return df
                except:
                    continue
        st.error("Nenhuma exchange conseguiu enviar dados.")
        return pd.DataFrame()

def add_features(df):
    df["return"] = df["close"].pct_change()
    df["volatility"] = (df["high"] - df["low"]) / (df["close"]+1e-9)
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma14"] = df["close"].rolling(14).mean()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    return df.dropna()

# -------- Candle pattern detector --------
def detect_patterns(df):
    patterns = []
    o, h, l, c = df.iloc[-1][["open","high","low","close"]]
    prev = df.iloc[-2]

    # Hammer
    if (min(o,c)-l > 2*abs(c-o)) and (h-max(o,c) < 0.25*abs(c-o)):
        patterns.append("Hammer")

    # Doji
    if abs(c-o) < 0.1*(h-l):
        patterns.append("Doji")

    # Bullish Engulfing
    if (prev.close < prev.open) and (c > o) and (c > prev.open) and (o < prev.close):
        patterns.append("Bullish Engulfing")

    # Bearish Engulfing
    if (prev.close > prev.open) and (c < o) and (o > prev.close) and (c < prev.open):
        patterns.append("Bearish Engulfing")

    return patterns

# ------------ UI Selection --------------
st.title("🖤 AI Trading Bot – Black Interface")

colA, colB, colC = st.columns(3)
exchange_sel = colA.selectbox("Exchange", list(EXCHANGES.keys()))
pair_sel = colB.selectbox("Par", PAIRS)
timeframe_sel = colC.selectbox("Timeframe", TIMEFRAMES, index=2)

start = st.button("▶️ Start Live")
stop = st.button("⏹ Stop")
manual = st.button("⚡ Atualizar Agora")

if "live" not in st.session_state:
    st.session_state.live = False
if start: st.session_state.live = True
if stop: st.session_state.live = False

chart_area = st.empty()
signal_area = st.empty()
pattern_area = st.empty()
heatmap_area = st.empty()

# ------------- Dashboard Update ----------
def update_dashboard():
    df = fetch_data(EXCHANGES[exchange_sel], pair_sel, timeframe_sel)
    if df.empty:
        return

    df = add_features(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Sinal simples
    direction = "SUBIR" if last.close > prev.close else "DESCER"
    color = "signal-up" if direction == "SUBIR" else "signal-down"

    signal_area.markdown(
        f"<div class='{color}'>Sinal: {direction} | Último preço: {last.close:.2f}</div>",
        unsafe_allow_html=True
    )

    # Padrões detectados
    patterns = detect_patterns(df)
    pattern_area.markdown(
        f"<div class='metric-card'><b>Padrões:</b> {', '.join(patterns) if patterns else 'Nenhum'}</div>",
        unsafe_allow_html=True
    )

    # Candlestick
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])
    fig.update_layout(template="plotly_dark", height=600)
    chart_area.plotly_chart(fig, use_container_width=True)

    # Heatmap volatilidade
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=[df["volatility"].tail(50)],
        colorscale="Inferno"
    ))
    heatmap_fig.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0))
    heatmap_area.plotly_chart(heatmap_fig, use_container_width=True)

# ----------- Live Update --------------
if st.session_state.live:
    st_autorefresh(interval=30000, key="refresh")
    update_dashboard()

# Manual
if manual:
    update_dashboard()

if not st.session_state.live and not manual:
    st.info("Clique ▶️ Start Live ou ⚡ Atualizar Agora para carregar dados.")
