import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIG & STARK HUD STYLING
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

    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00f3ff !important;
        text-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
        letter-spacing: 1.5px;
    }

    .jarvis-card {
        background: rgba(8, 18, 33, 0.85);
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

    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        color: #00f3ff !important;
        text-shadow: 0 0 8px #00f3ff;
    }

    .hud-line {
        height: 2px;
        background: linear-gradient(90deg, #00f3ff, rgba(0,243,255,0.2), transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BULLETPROOF DATA & TECHNICAL ENGINE
# ==========================================
DEFAULT_HORIZON = "5y"

@st.cache_data(ttl=900)
def fetch_stock_data(ticker_symbol, period=DEFAULT_HORIZON):
    """Fetches market data with MultiIndex header flattening."""
    try:
        t = yf.Ticker(ticker_symbol)
        df = t.history(period=period, auto_adjust=True)
        
        if df.empty:
            df = yf.download(ticker_symbol, period=period, progress=False, auto_adjust=True)

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                matched_name = None
                for item in col:
                    item_str = str(item).title()
                    if item_str in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                        matched_name = item_str
                        break
                new_cols.append(matched_name if matched_name else str(col[0]))
            df.columns = new_cols
        else:
            df.columns = [str(c).title() for c in df.columns]

        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']

        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        valid_cols = [c for c in req_cols if c in df.columns]

        if 'Close' not in valid_cols or len(df) < 30:
            return pd.DataFrame()

        df = df[valid_cols].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_vix():
    try:
        vix_df = yf.Ticker("^INDIAVIX").history(period="5d")
        if not vix_df.empty:
            return float(vix_df['Close'].iloc[-1])
    except:
        pass
    return 15.0

def compute_technical_indicators(df):
    """Computes technical indicators using pure Pandas."""
    data = df.copy()
    
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + (std * 2)
    data['BB_Lower'] = data['BB_Mid'] - (std * 2)
    
    return data.dropna()

def run_backtest_simulations(data):
    """Runs trade simulations and computes exact strategy metrics."""
    df = data.copy()
    
    df['Buy_Condition'] = (df['EMA_10'] > df['EMA_20']) & (df['MACD'] > df['MACD_Signal']) & (df['RSI'] > 45)
    df['Sell_Condition'] = (df['EMA_10'] < df['EMA_20']) | (df['MACD'] < df['MACD_Signal']) | (df['RSI'] > 75)
    
    in_position = False
    entry_price = 0.0
    trades = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        
        if not in_position and df['Buy_Condition'].iloc[i]:
            in_position = True
            entry_price = price
        elif in_position and df['Sell_Condition'].iloc[i]:
            pnl_pct = ((price - entry_price) / entry_price) * 100
            trades.append(pnl_pct)
            in_position = False

    if trades:
        win_trades = [t for t in trades if t > 0]
        accuracy = (len(win_trades) / len(trades)) * 100
        total_return = sum(trades)
        avg_profit = np.mean([t for t in trades if t > 0]) if win_trades else 0.0
        avg_loss = abs(np.mean([t for t in trades if t < 0])) if len(win_trades) < len(trades) else 1.0
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else avg_profit
    else:
        accuracy, total_return, profit_factor = 0.0, 0.0, 0.0

    return round(accuracy, 1), round(total_return, 1), len(trades), round(profit_factor, 2)

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
        <h1 style="margin: 5px 0 0 0; font-size: 26px;">J.A.R.V.I.S. QUANT TACTICAL HUD</h1>
    </div>
    <div style="text-align: right;">
        <span class="jarvis-badge" style="border-color: #00ffaa; color: #00ffaa;">SYSTEM ONLINE</span>
        <p style="margin: 5px 0 0 0; font-size: 13px; color: #88c0d0;">DATAFEED: UNIFIED (5-YEAR HORIZON)</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. UNIFIED SCANNER & DATA CACHE
# ==========================================
table_data = []
asset_cache = {}

with st.spinner("J.A.R.V.I.S. is parsing 5-year historical feeds and standardizing metrics..."):
    for name, symbol in ASSET_CATALOG.items():
        raw_df = fetch_stock_data(symbol, period=DEFAULT_HORIZON)
        if not raw_df.empty:
            tech_df = compute_technical_indicators(raw_df)
            accuracy, tot_ret, num_simulations, prof_factor = run_backtest_simulations(tech_df)
            latest = tech_df.iloc[-1]
            
            score = 0
            if latest['EMA_10'] > latest['EMA_50']: score += 30
            if latest['MACD'] > latest['MACD_Signal']: score += 25
            if 40 <= latest['RSI'] <= 70: score += 25
            if latest['Close'] > latest['BB_Mid']: score += 20
            
            signal = "STRONG BUY" if score >= 75 else "BUY" if score >= 55 else "NEUTRAL" if score >= 40 else "SELL"
            
            metrics_dict = {
                "Asset Name": name,
                "Symbol": symbol,
                "LTP (₹)": round(latest['Close'], 2),
                "Signal": signal,
                "Tactical Score": score,
                "RSI (14)": round(latest['RSI'], 1),
                "Simulations Count": num_simulations,
                "Win Accuracy (%)": accuracy,
                "Total Return (%)": tot_ret,
                "Profit Factor": prof_factor
            }
            
            table_data.append(metrics_dict)
            asset_cache[name] = {
                "tech_df": tech_df,
                "metrics": metrics_dict
            }

scanner_df = pd.DataFrame(table_data)

st.markdown("### 🛰️ MULTI-ASSET SIMULATION SCANNER")
st.dataframe(
    scanner_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "LTP (₹)": st.column_config.NumberColumn(format="₹%.2f"),
        "Win Accuracy (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        "Total Return (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Tactical Score": st.column_config.NumberColumn(format="%d / 100")
    }
)

st.markdown('<div class="hud-line"></div>', unsafe_allow_html=True)

# ==========================================
# 5. SYNCHRONIZED GRAPHICAL HUD & DIAGNOSTICS
# ==========================================
st.markdown("### 🎯 GRAPHICAL HUD & J.A.R.V.I.S. DIAGNOSTICS")

selected_asset_name = st.selectbox(
    "SELECT ASSET FOR GRAPHICAL ANALYSIS & SUBCHARTS:",
    list(ASSET_CATALOG.keys()),
    index=0
)

cached_asset = asset_cache[selected_asset_name]
df_tech = cached_asset["tech_df"]
m = cached_asset["metrics"]
vix_val = fetch_vix()

curr_price = m["LTP (₹)"]
score = m["Tactical Score"]
accuracy = m["Win Accuracy (%)"]
trade_cnt = m["Simulations Count"]
tot_return = m["Total Return (%)"]
profit_factor = m["Profit Factor"]
sig_text = m["Signal"]

opt_strategy = "Bull Call Spread" if (score >= 55 and vix_val < 16) else "Bull Put Spread (Credit)" if score >= 55 else "Iron Condor" if score >= 40 else "Bear Put Spread"

st.markdown(f"""
<div class="jarvis-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span style="font-family: 'Orbitron'; font-weight: 700; color: #00f3ff; font-size: 16px;">
            🤖 STARK AI DIAGNOSTIC :: {selected_asset_name.upper()}
        </span>
        <span class="jarvis-badge">ACCURACY: {accuracy}% ({trade_cnt} SIMULATIONS)</span>
    </div>
    <div style="font-family: monospace; font-size: 14px; color: #c0edf7; line-height: 1.7;">
        > <strong>TACTICAL EVALUATION:</strong> Composite Score of <strong>{score}/100</strong> triggering a <strong>{sig_text}</strong> state.<br>
        > <strong>LTP:</strong> ₹{curr_price:,.2f} | <strong>INDIA VIX:</strong> {vix_val:.2f} | <strong>RSI (14):</strong> {m['RSI (14)']}<br>
        > <strong>RECOMMENDED DERIVATIVE STRUCTURE:</strong> Deploy a <strong>{opt_strategy}</strong>.<br>
        > <strong>SIMULATION METRICS:</strong> Analyzed <strong>{trade_cnt} historical trade executions</strong>. Strategy generated <strong>{accuracy}% Win Accuracy</strong>, a <strong>{profit_factor} Profit Factor</strong>, and <strong>{tot_return}% Net Return</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("LTP", f"₹{curr_price:,.2f}")
m2.metric("Win Accuracy", f"{accuracy}%", f"{trade_cnt} Trades")
m3.metric("Total Return", f"{tot_return}%")
m4.metric("Tactical Score", f"{score}/100", sig_text)

# Plotly Subcharts (Price + BB, MACD, RSI)
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    subplot_titles=(
        f"{selected_asset_name} Price, EMAs & Bollinger Bands",
        "MACD (12, 26, 9)",
        "RSI (14)"
    ),
    row_heights=[0.55, 0.25, 0.20]
)

fig.add_trace(go.Candlestick(
    x=df_tech.index, open=df_tech['Open'], high=df_tech['High'],
    low=df_tech['Low'], close=df_tech['Close'], name="Price"
), row=1, col=1)

fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['EMA_10'], line=dict(color='#00ffaa', width=1), name="EMA 10"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['EMA_50'], line=dict(color='#ff00ff', width=1), name="EMA 50"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_Upper'], line=dict(color='rgba(0, 243, 255, 0.4)', width=1, dash='dash'), name="BB Upper"), row=1, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['BB_Lower'], line=dict(color='rgba(0, 243, 255, 0.4)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(0, 243, 255, 0.04)', name="BB Lower"), row=1, col=1)

colors = np.where(df_tech['MACD_Hist'] >= 0, '#00ffaa', '#ff0055')
fig.add_trace(go.Bar(x=df_tech.index, y=df_tech['MACD_Hist'], marker_color=colors, name="MACD Hist"), row=2, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD'], line=dict(color='#00f3ff', width=1.5), name="MACD Line"), row=2, col=1)
fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['MACD_Signal'], line=dict(color='#ffaa00', width=1.5), name="Signal Line"), row=2, col=1)

fig.add_trace(go.Scatter(x=df_tech.index, y=df_tech['RSI'], line=dict(color='#00f3ff', width=1.5), name="RSI"), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#ff0055", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#00ffaa", row=3, col=1)

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
