import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import vectorbt as vbt
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. UI CONFIGURATION & JARVIS CSS
# ==========================================
st.set_page_config(page_title="The System | Jarvis UI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #02060f; }
    h1, h2, h3, h4, p, span { color: #00e5ff !important; font-family: 'Courier New', Courier, monospace; }
    h1 { text-shadow: 0px 0px 10px #00e5ff; }
    .stSelectbox label, .stRadio label, .stTextInput label { color: #00e5ff !important; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricValue"] { color: #00e5ff; text-shadow: 0px 0px 8px #00e5ff; font-family: 'Courier New'; }
    div[data-testid="stMetricLabel"] { color: #88c0d0; }
    .jarvis-panel { 
        border: 1px solid #00e5ff; padding: 20px; border-radius: 5px; 
        background: rgba(0,229,255,0.05); color: #e5e9f0; 
        font-family: 'Courier New', monospace; box-shadow: 0px 0px 15px rgba(0, 229, 255, 0.2); 
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE & CACHING
# ==========================================
@st.cache_data(ttl=900)
def fetch_data(ticker, period="2y"):
    df = yf.Ticker(ticker).history(period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        df.index = df.index.tz_localize(None)
    return df

@st.cache_data(ttl=900)
def fetch_vix():
    try:
        vix = yf.Ticker("^INDIAVIX").history(period="1mo")
        return vix['Close'].iloc[-1] if not vix.empty else 15.0
    except:
        return 15.0

@st.cache_data(ttl=3600)
def fetch_graham_metrics(ticker):
    try:
        info = yf.Ticker(ticker).info
        eps = info.get('trailingEPS', 0)
        bvps = info.get('bookValue', 0)
        if eps and bvps and eps > 0 and bvps > 0:
            return np.sqrt(22.5 * eps * bvps)
    except:
        pass
    return None

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("### THE SYSTEM :: CONTROL PANEL")
asset_type = st.sidebar.radio("Asset Class", ["Nifty 50 Equity", "Market Indices"])

if asset_type == "Market Indices":
    tickers = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Sensex": "^BSESN", "FinNifty": "^CNXFIN"}
else:
    tickers = {"Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS", "Infosys": "INFY.NS", "ITC": "ITC.NS", "Custom": "CUSTOM"}

selected_name = st.sidebar.selectbox("Select Asset", list(tickers.keys()))
if selected_name == "Custom":
    ticker = st.sidebar.text_input("Enter Ticker (e.g. TATAMOTORS.NS):", "TATAMOTORS.NS")
else:
    ticker = tickers[selected_name]

# ==========================================
# 4. TECHNICAL ANALYSIS ENGINE
# ==========================================
st.title(f"JARVIS :: F&O TACTICAL OVERVIEW [{ticker}]")

df = fetch_data(ticker)
vix = fetch_vix()
graham_number = fetch_graham_metrics(ticker) if asset_type == "Nifty 50 Equity" else None

if df.empty:
    st.error("Jarvis Error: Data source offline or invalid ticker.")
    st.stop()

# Indicators (Pandas-TA)
df['EMA_20'] = ta.ema(df['Close'], length=20)
df['EMA_50'] = ta.ema(df['Close'], length=50)
df.ta.macd(append=True)
df['RSI'] = ta.rsi(df['Close'], length=14)
df.ta.bbands(append=True)

df.dropna(inplace=True)
latest = df.iloc[-1]
current_price = latest['Close']

# Dynamic Column Detection
macd_col = [c for c in df.columns if c.startswith('MACD_')][0]
sig_col = [c for c in df.columns if c.startswith('MACDs_')][0]
bb_upper = [c for c in df.columns if c.startswith('BBU_')][0]
bb_mid = [c for c in df.columns if c.startswith('BBM_')][0]

# Weighted Scoring Model
score = 0

# 1. Trend (EMA 20/50 Crossover) - 30%
if latest['EMA_20'] > latest['EMA_50']: 
    score += 30

# 2. Momentum (MACD) - 20%
if latest[macd_col] > latest[sig_col]: 
    score += 20

# 3. Strength (RSI) - 20%
if 40 <= latest['RSI'] <= 70: 
    score += 20
elif latest['RSI'] < 40: 
    score += 10

# 4. Breakout (Bollinger) - 15%
if current_price > latest[bb_mid] and current_price < latest[bb_upper]: 
    score += 15

# 5. Valuation (Graham Intrinsic) - 15%
if graham_number:
    if current_price < graham_number: 
        score += 15
else:
    if latest['EMA_20'] > latest['EMA_50']: 
        score += 10
    if latest[macd_col] > latest[sig_col]: 
        score += 5

# Signal & F&O Strategy Routing
if score >= 70: signal_label = "STRONG BUY"
elif 55 <= score < 70: signal_label = "BUY"
elif 45 <= score < 55: signal_label = "NEUTRAL"
elif 30 <= score < 45: signal_label = "SELL"
else: signal_label = "STRONG SELL"

if vix < 13:
    if score >= 55: strategy = "Bull Call Spread / Long Call (Debit)"
    elif score <= 45: strategy = "Bear Put Spread / Long Put (Debit)"
    else: strategy = "Calendar Spread"
elif vix <= 20:
    if score >= 55: strategy = "Bull Put Spread (Credit)"
    elif score <= 45: strategy = "Bear Call Spread (Credit)"
    else: strategy = "Iron Condor (Delta Neutral)"
else:
    if score >= 55: strategy = "Deep OTM Bull Put Spread"
    elif score <= 45: strategy = "Deep OTM Bear Call Spread"
    else: strategy = "Short Straddle / Iron Butterfly (High Theta Decay)"

# ==========================================
# 5. VECTORBT BACKTESTING ENGINE
# ==========================================
df['Signal_Score'] = 0
df['Signal_Score'] = np.where(df['EMA_20'] > df['EMA_50'], df['Signal_Score'] + 30, df['Signal_Score'])
df['Signal_Score'] = np.where(df[macd_col] > df[sig_col], df['Signal_Score'] + 20, df['Signal_Score'])
df['Signal_Score'] = np.where((df['RSI'] >= 40) & (df['RSI'] <= 70), df['Signal_Score'] + 20, df['Signal_Score'])
df['Signal_Score'] = np.where((df['Close'] > df[bb_mid]) & (df['Close'] < df[bb_upper]), df['Signal_Score'] + 15, df['Signal_Score'])

entries = df['Signal_Score'] >= 70
exits = df['Signal_Score'] <= 30

try:
    pf = vbt.Portfolio.from_signals(df['Close'], entries, exits, init_cash=100000, fees=0.001)
    win_rate = float(pf.trades.win_rate() * 100) if pf.trades.count() > 0 else 0.0
    total_return = float(pf.total_return() * 100)
except Exception:
    win_rate = 0.0
    total_return = 0.0

# ==========================================
# 6. UI DISPLAY
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("LTP", f"₹{current_price:,.2f}")
col2.metric("India VIX", f"{vix:.2f}")
if graham_number: col3.metric("Graham Intrinsic", f"₹{graham_number:,.2f}")
else: col3.metric("Graham Intrinsic", "N/A (Index)")
col4.metric("Signal Net Score", f"{score}/100", signal_label)

st.markdown("### SYSTEM COMMENTARY")
graham_status = "Trading at a discount to intrinsic value." if graham_number and current_price < graham_number else ("Trading at a premium." if graham_number else "Macro-index detected; bypassing Graham formula.")

jarvis_text = f"""
<div class="jarvis-panel">
    <strong>J.A.R.V.I.S DIAGNOSTIC:</strong><br><br>
    Analysis complete for <strong>{ticker}</strong>.<br>
    The quantitative model generated a composite score of <strong>{score}/100</strong>, triggering a <strong>{signal_label}</strong> signal.<br>
    Valuation status: {graham_status}<br><br>
    India VIX is sitting at <strong>{vix:.2f}</strong>. Under current implied volatility conditions, the recommended derivative structure is a <strong>{strategy}</strong>.<br>
    VectorBT automated backtest performance across all historical cycles yields a win rate of <strong>{win_rate:.1f}%</strong> with a total return of <strong>{total_return:.1f}%</strong>.
</div>
"""
st.markdown(jarvis_text, unsafe_allow_html=True)

# Charting
st.markdown("---")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='cyan', width=1), name='EMA 20'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='magenta', width=1), name='EMA 50'))
fig.update_layout(template='plotly_dark', title=f"{ticker} Technical Layout", height=600, margin=dict(l=0, r=0, b=0, t=40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)

st.markdown("<p style='text-align: center; font-size: 11px; color: #88c0d0;'>DISCLAIMER: FOR EDUCATIONAL PURPOSES ONLY. This app provides algorithmic visualizations and does not constitute financial advice.</p>", unsafe_allow_html=True)
