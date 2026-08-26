import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIG & STARK INDUSTRIES HUD CSS
# ==========================================
st.set_page_config(
    page_title="J.A.R.V.I.S. :: Tactical Market HUD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@500;600;700&display=swap');

    .stApp {
        background-color: #040812;
        background-image: radial-gradient(circle at 50% 10%, rgba(0, 243, 255, 0.08) 0%, transparent 70%);
        color: #e0f7fc;
        font-family: 'Rajdhani', sans-serif;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00f3ff !important;
        text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
        letter-spacing: 1.5px;
    }

    /* Glassmorphism Card Panels */
    .jarvis-card {
        background: rgba(8, 18, 33, 0.75);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-radius: 6px;
        padding: 18px;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.12), inset 0 0 15px rgba(0, 243, 255, 0.05);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    .jarvis-badge {
        background: rgba(0, 243, 255, 0.15);
        border: 1px solid #00f3ff;
        color: #00f3ff;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: 'Orbitron', monospace;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .stMetric {
        background: rgba(10, 25, 47, 0.6);
        border: 1px solid rgba(0, 243, 255, 0.2);
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        color: #00f3ff !important;
        text-shadow: 0 0 8px #00f3ff;
    }

    .hud-line {
        height: 2px;
        background: linear-gradient(90deg, #00f3ff, rgba(0,243,255,0.2), transparent);
        margin: 15px 0;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #040812; }
    ::-webkit-scrollbar-thumb { background: #00f3ff; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. RESILIENT DATA FETCHING & CALCULATIONS
# ==========================================
@st.cache_data(ttl=600)
def fetch_stock_data(ticker_symbol, period="1y"):
    """Fetches market data with multi-index cleanup and validation."""
    try:
        df = yf.Ticker(ticker_symbol).history(period=period)
        if df.empty:
            df = yf.download(ticker_symbol, period=period, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_vix():
    try:
        vix_df = yf.Ticker("^INDIAVIX").history(period="5d")
        if not vix_df.empty:
            return vix_df['Close'].iloc[-1]
    except:
        pass
    return 15.0

def compute_technical_indicators(df):
    """Computes technical indicators using pure Pandas."""
    data = df.copy()
    
    # 1. EMAs
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    # 2. RSI (14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD (12, 26, 9)
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # 4. Bollinger Bands (20, 2)
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + (std * 2)
    data['BB_Lower'] = data['BB_Mid'] - (std * 2)
    
    return data.dropna()

def backtest_strategy(data):
    """Vectorized strategy backtester with per-asset accuracy calculation."""
    df = data.copy()
    
    # Entry: Bullish EMA crossover + MACD confirmation
    df['Buy_Signal'] = (df['EMA_20'] > df['EMA_50']) & (df['MACD'] > df['MACD_Signal'])
    # Exit: Bearish crossover
    df['Sell_Signal'] = (df['EMA_20'] < df['EMA_50']) | (df['MACD'] < df['MACD_Signal'])
    
    position = 0
    trades = []
    entry_price = 0
    
    for i in range(len(df)):
        if position == 0 and df['Buy_Signal'].iloc[i]:
            position = 1
            entry_price = df['Close'].iloc[i]
        elif position == 1 and df['Sell_Signal'].iloc[i]:
            exit_price = df['Close'].iloc[i]
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            trades.append(pnl_pct)
            position = 0
            
    if trades:
        win_trades = [t for t in trades if t > 0]
        accuracy = (len(win_trades) / len(trades)) * 100
        total_return = sum(trades)
    else:
        accuracy = 0.0
        total_return = 0.0
        
    return round(accuracy, 1), round(total_return, 1), len(trades)

# Asset Catalog
ASSET_CATALOG = {
    "Nifty 50 Index": "^NSEI",
    "Bank Nifty Index": "^NSEBANK",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "State Bank of India": "SBIN.NS",
    "ITC Limited": "ITC.NS"
}

# ==========================================
# 3. HUD TOP HEADER
# ==========================================
st.markdown("""
<div class="jarvis-card" style="display: flex; justify-content: space-between; align-items: center;">
    <div>
        <span class="jarvis-badge">STARK INDUSTRIES PROTOCOL</span>
        <h1 style="margin: 5px 0 0 0; font-size: 26px;">J.A.R.V.I.S. QUANTITATIVE TACTICAL HUD</h1>
    </div>
    <div style="text-align: right;">
        <span class="jarvis-badge" style="border-color: #00ffaa; color: #00ffaa;">SYSTEM ONLINE</span>
        <p style="margin: 5px 0 0 0; font-size: 13px; color: #88c0d0;">MARKET DATAFEED: ACTIVE</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. MULTI-ASSET SCANNER & BACKTEST TABLE
# ==========================================
st.markdown("### 🛰️ MULTI-ASSET SCANNER & ACCURACY OVERVIEW")

table_data = []

with st.spinner("J.A.R.V.I.S. is calculating technical indicators and backtest metrics..."):
    for name, symbol in ASSET_CATALOG.items():
        raw_df = fetch_stock_data(symbol, period="1y")
        if not raw_df.empty and len(raw_df) > 50:
            tech_df = compute_technical_indicators(raw_df)
            accuracy, tot_ret, num_trades = backtest_strategy(tech_df)
            latest = tech_df.iloc[-1]
            
            # Scoring
            score = 0
            if latest['EMA_20'] > latest['EMA_50']: score += 30
            if latest['MACD'] > latest['MACD_Signal']: score += 25
            if 40 <= latest['RSI'] <= 70: score += 25
            if latest['Close'] > latest['BB_Mid']: score += 20
            
            if score >= 70: signal = "STRONG BUY"
            elif score >= 55: signal = "BUY"
            elif score >= 40: signal = "NEUTRAL"
            else: signal = "SELL"
            
            table_data.append({
                "Asset Name": name,
                "Symbol": symbol,
                "LTP (₹)": round(latest['Close'], 2),
                "Signal": signal,
                "Score": score,
                "RSI (14)": round(latest['RSI'], 1),
                "Backtest Accuracy (%)": accuracy,
                "Total Return (%)": tot_ret,
                "Total Trades": num_trades
            })

scanner_df = pd.DataFrame(table_data)

st.dataframe(
    scanner_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "LTP (₹)": st.column_config.NumberColumn(format="₹%.2f"),
        "Backtest Accuracy (%)": st.column_config.ProgressColumn(
            "Backtest Accuracy (%)", format="%.1f%%", min_value=0, max_value=100
        ),
        "Total Return (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Score": st.column_config.NumberColumn(format="%d / 100")
    }
)

st.markdown('<div class="hud-line"></div>', unsafe_allow_html=True)

# ==========================================
# 5. DETAILED ANALYSIS & GRAPHICAL DISPLAY
# ==========================================
st.markdown("### 🎯 DEEP DIVE TACTICAL ANALYSIS")

selected_asset_name = st.selectbox(
    "SELECT STOCK / INDEX FOR DETAILED HUD GRAPHICAL DISPLAY:",
    list(ASSET_CATALOG.keys()),
    index=0
)
selected_ticker = ASSET_CATALOG[selected_asset_name]

# Fetch asset data
df_selected = fetch_stock_data(selected_ticker, period="1y")
df_tech = compute_technical_indicators(df_selected)
accuracy, tot_return, trade_cnt = backtest_strategy(df_tech)
vix_val = fetch_vix()

latest_row = df_tech.iloc[-1]
curr_price = latest_row['Close']

# Dynamic Scoring & Option Strategy Routing
score = 0
if latest_row['EMA_20'] > latest_row['EMA_50']: score += 30
if latest_row['MACD'] > latest_row['MACD_Signal']: score += 25
if 40 <= latest_row['RSI'] <= 70: score += 25
if latest_row['Close'] > latest_row['BB_Mid']: score += 20

if score >= 70:
    sig_text = "STRONG BUY"
    opt_strategy = "Bull Call Spread" if vix_val < 15 else "Bull Put Spread (Credit)"
elif score >= 55:
    sig_text = "BUY"
    opt_strategy = "Long Call / Call Spread"
elif score >= 40:
    sig_text = "NEUTRAL"
    opt_strategy = "Iron Condor / Calendar Spread"
else:
    sig_text = "SELL / BEARISH"
    opt_strategy = "Bear Put Spread" if vix_val < 15 else "Bear Call Spread (Credit)"

# JARVIS Interactive Terminal Commentary
st.markdown(f"""
<div class="jarvis-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span style="font-family: 'Orbitron'; font-weight: 700; color: #00f3ff; font-size: 16px;">
            🤖 STARK AI DIAGNOSTIC :: {selected_asset_name.upper()}
        </span>
        <span class="jarvis-badge">ACCURACY: {accuracy}%</span>
    </div>
    <div style="font-family: monospace; font-size: 14px; color: #c0edf7; line-height: 1.6;">
        > <strong>TACTICAL SCORE:</strong> {score}/100 | <strong>ACTION:</strong> <span style="color:#00ffaa;">{sig_text}</span><br>
        > <strong>CURRENT PRICE:</strong> ₹{curr_price:,.2f} | <strong>INDIA VIX:</strong> {vix_val:.2f}<br>
        > <strong>RSI (14):</strong> {latest_row['RSI']:.1f} ({'OVERBOUGHT' if latest_row['RSI']>70 else 'OVERSOLD' if latest_row['RSI']<30 else 'NEUTRAL RANGE'})<br>
        > <strong>F&O STRATEGY SUGGESTION:</strong> Deploy <strong>{opt_strategy}</strong> based on implied volatility parameters.<br>
        > <strong>HISTORICAL BACKTEST:</strong> Achieved <strong>{accuracy}% accuracy</strong> over {trade_cnt} executed signal cycles with <strong>{tot_return}% total return</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# Metrics Grid
m1, m2, m3, m4 = st.columns(4)
m1.metric("LTP", f"₹{curr_price:,.2f}")
m2.metric("Backtest Accuracy", f"{accuracy}%")
m3.metric("RSI (14)", f"{latest_row['RSI']:.1f}")
m4.metric("Tactical Score", f"{score}/100", sig_text)

# Plotly Interactive Multi-Subchart (Price + BB + EMA, MACD, RSI)
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=(
        f"{selected_asset_name} Price, EMAs & Bollinger Bands",
        "MACD (Moving Average Convergence Divergence)",
        "RSI (Relative Strength Index)"
    ),
    row_heights=[0.55, 0.25, 0.20]
)

# 1. Price + EMAs + Bollinger Bands
fig.add_trace(go.Candlestick(
    x=df_tech.index, open=df_tech['Open'], high=df_tech['High'],
    low=df_tech['Low'], close=df_tech['Close'], name="Candlesticks"
), row=1, col=1)

fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['EMA_20'], line=dict(color='#00f3ff', width=1.5), name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['EMA_50'], line=dict(color='#ff00ff', width=1.5), name="EMA 50"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_Upper'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'), name="BB Upper"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_Lower'], line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(0, 243, 255, 0.03)', name="BB Lower"), row=1, col=1)

# 2. MACD Subchart
colors = np.where(df_tech['MACD_Hist'] >= 0, '#00ffaa', '#ff0055')
fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_Hist'], marker_color=colors, name="MACD Hist"), row=2, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], line=dict(color='#00f3ff', width=1.5), name="MACD Line"), row=2, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_Signal'], line=dict(color='#ffaa00', width=1.5), name="Signal Line"), row=2, col=1)

# 3. RSI Subchart
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['RSI'], line=dict(color='#00f3ff', width=1.5), name="RSI"), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#ff0055", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#00ffaa", row=3, col=1)

# Layout adjustments for HUD theme
fig.update_layout(
    template="plotly_dark",
    height=800,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(4, 8, 18, 0.8)',
    margin=dict(l=10, r=10, t=40, b=10),
    showlegend=True,
    xaxis_rangeslider_visible=False
)

fig.update_xaxes(showgrid=True, gridcolor='rgba(0, 243, 255, 0.1)')
fig.update_yaxes(showgrid=True, gridcolor='rgba(0, 243, 255, 0.1)')

st.plotly_chart(fig, use_container_width=True)

st.caption("⚡ J.A.R.V.I.S. QUANT SYSTEM :: FOR EDUCATIONAL & ALGORITHMIC RESEARCH PURPOSES ONLY")
