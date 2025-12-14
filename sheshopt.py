import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import requests
from bs4 import BeautifulSoup
import json
import re

st.set_page_config(page_title="Global Options Pricing", page_icon="🌍", layout="wide")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);}
    h1 {color: #60a5fa;}
    .stAlert {background-color: rgba(30, 41, 59, 0.5);}
</style>
""", unsafe_allow_html=True)

def black_scholes(S, K, T, r, sigma, opt_type='call'):
    if T <= 0:
        return max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, opt_type='call'):
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta_call = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
    theta_put = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    theta = theta_call if opt_type == 'call' else theta_put
    vega = S*norm.pdf(d1)*np.sqrt(T)/100
    rho = K*T*np.exp(-r*T)*(norm.cdf(d2) if opt_type == 'call' else -norm.cdf(-d2))/100
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

@st.cache_data(ttl=60)
def fetch_nse_data(symbol):
    """Fetch NSE India stock data"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            price_info = data.get('priceInfo', {})
            return {
                'price': price_info.get('lastPrice', 0),
                'change': price_info.get('change', 0),
                'pChange': price_info.get('pChange', 0)
            }
    except:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_bse_data(symbol):
    """Fetch BSE India stock data"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w?scripcode={symbol}&flag=0"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'Data' in data and data['Data']:
                latest = data['Data'][0]
                return {
                    'price': latest.get('CurrRate', 0),
                    'change': latest.get('pChange', 0)
                }
    except:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_yahoo_price(symbol):
    """Fetch price from Yahoo Finance"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                meta = data['chart']['result'][0]['meta']
                return {
                    'price': meta.get('regularMarketPrice', 0),
                    'prev_close': meta.get('previousClose', 0),
                    'currency': meta.get('currency', 'USD')
                }
    except:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_alpha_vantage(symbol, api_key='demo'):
    """Fetch from Alpha Vantage"""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                return {
                    'price': float(quote.get('05. price', 0)),
                    'prev_close': float(quote.get('08. previous close', 0))
                }
    except:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_twelve_data(symbol, api_key='demo'):
    """Fetch from Twelve Data"""
    try:
        url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'close' in data:
                return {'price': float(data['close'])}
    except:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_finnhub(symbol, api_key='demo'):
    """Fetch from Finnhub"""
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'c' in data and data['c'] > 0:
                return {'price': data['c'], 'prev_close': data.get('pc', data['c'])}
    except:
        pass
    return None

def fetch_stock_price_multi_source(symbol, exchange='AUTO'):
    """Try multiple data sources to fetch stock price"""
    
    if exchange == 'NSE':
        data = fetch_nse_data(symbol.replace('.NS', ''))
        if data and data['price'] > 0:
            return data
    
    if exchange == 'BSE':
        data = fetch_bse_data(symbol.replace('.BO', ''))
        if data and data['price'] > 0:
            return data
    
    data = fetch_yahoo_price(symbol)
    if data and data['price'] > 0:
        return data
    
    data = fetch_alpha_vantage(symbol)
    if data and data['price'] > 0:
        return data
    
    data = fetch_twelve_data(symbol)
    if data and data['price'] > 0:
        return data
    
    data = fetch_finnhub(symbol)
    if data and data['price'] > 0:
        return data
    
    return None

def generate_comprehensive_options(symbol, spot, r, sigma, num_strikes, strike_step, num_expiries, days_ahead):
    """Generate comprehensive options chain"""
    
    strikes = []
    atm_strike = round(spot / strike_step) * strike_step
    
    for i in range(-num_strikes//2, num_strikes//2 + 1):
        strike = atm_strike + (i * strike_step)
        if strike > 0:
            strikes.append(strike)
    
    expiries = []
    for i in range(num_expiries):
        exp_date = datetime.now() + timedelta(days=days_ahead + (i * 7))
        expiries.append((exp_date, exp_date.strftime("%d %b %y")))
    
    options = []
    
    for exp_date, exp_label in expiries:
        T = (exp_date - datetime.now()).days / 365.0
        
        for strike in strikes:
            for opt_type in ['call', 'put']:
                theo_price = black_scholes(spot, strike, T, r, sigma, opt_type)
                intrinsic = max(spot - strike, 0) if opt_type == 'call' else max(strike - spot, 0)
                time_value = theo_price - intrinsic
                greeks = calculate_greeks(spot, strike, T, r, sigma, opt_type)
                
                option_code = f"{symbol} {exp_label} {strike} {'CE' if opt_type == 'call' else 'PE'}"
                
                options.append({
                    'Option Code': option_code,
                    'Symbol': symbol,
                    'Spot Price': spot,
                    'Strike': strike,
                    'Type': 'CALL' if opt_type == 'call' else 'PUT',
                    'Expiry': exp_label,
                    'Expiry Date': exp_date.strftime('%Y-%m-%d'),
                    'Days to Expiry': (exp_date - datetime.now()).days,
                    'Theoretical Price': theo_price,
                    'Intrinsic Value': intrinsic,
                    'Time Value': time_value,
                    'Delta': greeks['delta'],
                    'Gamma': greeks['gamma'],
                    'Theta': greeks['theta'],
                    'Vega': greeks['vega'],
                    'Rho': greeks['rho'],
                    'ITM': (opt_type == 'call' and spot > strike) or (opt_type == 'put' and spot < strike),
                    'Moneyness': 'ITM' if ((opt_type == 'call' and spot > strike) or (opt_type == 'put' and spot < strike)) else 'OTM'
                })
    
    return pd.DataFrame(options)

st.title("🌍 Global Options Pricing - Universal Market Coverage")
st.markdown("**Generate options for ANY stock from ANY exchange worldwide**")

st.info("💡 **Built from scratch** - Uses multiple data sources: NSE API, BSE API, Yahoo Finance, Alpha Vantage, Twelve Data, Finnhub + more. Works for stocks from India (NSE/BSE), US, UK, Europe, Asia, and all other global markets!")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📝 Enter Stock Symbols")
    
    symbols_text = st.text_area(
        "Enter symbols (one per line) - add exchange suffix if needed:",
        height=150,
        placeholder="""Examples:
RELIANCE.NS (NSE India)
500325.BO (BSE India - Reliance BSE code)
^BSESN (BSE SENSEX)
^NSEI (NSE NIFTY)
AAPL (US - Apple)
TSLA (US - Tesla)
MSFT (US - Microsoft)
GOOGL (US - Google)
VODAFONE.L (London)
BMW.DE (Germany)
7203.T (Japan - Toyota)
0700.HK (Hong Kong - Tencent)
SBIN.NS (State Bank of India)
TCS.NS (Tata Consultancy)
INFY.NS (Infosys)""",
        help="NSE: .NS | BSE: .BO | London: .L | Germany: .DE | Japan: .T | HK: .HK | US: no suffix"
    )
    
    symbols = [s.strip().upper() for s in symbols_text.split('\n') if s.strip()]

with col2:
    st.subheader("⚙️ Parameters")
    
    exchange = st.selectbox("Data Priority", ["AUTO", "NSE", "BSE", "Yahoo", "Multi-Source"])
    risk_free = st.slider("Risk-Free Rate %", 0.0, 20.0, 6.5, 0.5) / 100
    iv = st.slider("Implied Volatility %", 5.0, 150.0, 30.0, 5.0) / 100

st.subheader("📊 Options Chain Settings")

col1, col2, col3, col4 = st.columns(4)

with col1:
    num_strikes = st.slider("Strike Prices", 5, 51, 15, 2)
with col2:
    strike_step = st.number_input("Strike Step", 1, 1000, 50)
with col3:
    num_expiries = st.slider("Expiry Dates", 1, 24, 6)
with col4:
    days_ahead = st.slider("First Expiry (days)", 1, 180, 7)

col1, col2 = st.columns([1, 5])
with col1:
    generate = st.button("🚀 GENERATE OPTIONS", type="primary", use_container_width=True)
with col2:
    if generate:
        st.info(f"Generating {num_strikes * num_expiries * 2 * len(symbols):,} option contracts...")

if 'all_options' not in st.session_state:
    st.session_state.all_options = None

if generate and symbols:
    all_options = []
    progress = st.progress(0)
    status = st.empty()
    
    for idx, symbol in enumerate(symbols):
        status.text(f"Fetching {symbol} ({idx+1}/{len(symbols)})...")
        
        stock_data = fetch_stock_price_multi_source(symbol, exchange)
        
        if stock_data and stock_data['price'] > 0:
            price = stock_data['price']
            change = stock_data.get('change', 0)
            pct = stock_data.get('pChange', 0)
            
            st.success(f"✓ {symbol}: ₹{price:.2f} ({pct:+.2f}%)" if '.NS' in symbol or '.BO' in symbol else f"✓ {symbol}: ${price:.2f}")
            
            options_df = generate_comprehensive_options(
                symbol, price, risk_free, iv, 
                num_strikes, strike_step, num_expiries, days_ahead
            )
            all_options.append(options_df)
        else:
            st.warning(f"⚠️ {symbol}: Could not fetch price")
        
        progress.progress((idx + 1) / len(symbols))
    
    if all_options:
        st.session_state.all_options = pd.concat(all_options, ignore_index=True)
        st.success(f"✅ Generated {len(st.session_state.all_options):,} options contracts!")
    
    progress.empty()
    status.empty()

if st.session_state.all_options is not None:
    st.markdown("---")
    st.subheader("📈 Options Chain Data")
    
    df = st.session_state.all_options
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search = st.text_input("🔍 Search", placeholder="Code, symbol, strike, expiry...")
    with col2:
        filter_type = st.selectbox("Type", ["All", "CALL", "PUT"])
    with col3:
        filter_money = st.selectbox("Moneyness", ["All", "ITM", "OTM"])
    
    filtered = df.copy()
    
    if search:
        mask = (
            filtered['Option Code'].str.contains(search, case=False, na=False) |
            filtered['Symbol'].str.contains(search, case=False, na=False) |
            filtered['Strike'].astype(str).str.contains(search, na=False) |
            filtered['Expiry'].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    
    if filter_type != "All":
        filtered = filtered[filtered['Type'] == filter_type]
    
    if filter_money != "All":
        filtered = filtered[filtered['Moneyness'] == filter_money]
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total", f"{len(filtered):,}")
    with col2:
        st.metric("Calls", f"{len(filtered[filtered['Type']=='CALL']):,}")
    with col3:
        st.metric("Puts", f"{len(filtered[filtered['Type']=='PUT']):,}")
    with col4:
        st.metric("ITM", f"{len(filtered[filtered['ITM']]):,}")
    with col5:
        st.metric("Symbols", filtered['Symbol'].nunique())
    with col6:
        st.metric("Expiries", filtered['Expiry'].nunique())
    
    st.dataframe(
        filtered.style.format({
            'Spot Price': '₹{:.2f}',
            'Strike': '₹{:.0f}',
            'Theoretical Price': '₹{:.3f}',
            'Intrinsic Value': '₹{:.3f}',
            'Time Value': '₹{:.3f}',
            'Delta': '{:.4f}',
            'Gamma': '{:.5f}',
            'Theta': '{:.4f}',
            'Vega': '{:.4f}',
            'Rho': '{:.4f}'
        }),
        use_container_width=True,
        height=600
    )
    
    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        f"options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

else:
    st.markdown("### 🌟 Supported Markets & Examples:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **🇮🇳 India**
        - **NSE**: RELIANCE.NS, TCS.NS, INFY.NS
        - **BSE**: 500325.BO (Reliance)
        - **Indices**: ^NSEI, ^BSESN
        - SBIN.NS, HDFCBANK.NS
        """)
    
    with col2:
        st.markdown("""
        **🇺🇸 United States**
        - AAPL, MSFT, GOOGL
        - TSLA, NVDA, AMD
        - ^GSPC (S&P 500)
        - ^DJI (Dow Jones)
        """)
    
    with col3:
        st.markdown("""
        **🇬🇧 UK & Europe**
        - VODAFONE.L (London)
        - BP.L (BP)
        - BMW.DE (Germany)
        - SAP.DE
        """)
    
    with col4:
        st.markdown("""
        **🌏 Asia Pacific**
        - 7203.T (Toyota-Japan)
        - 0700.HK (Tencent-HK)
        - 005930.KS (Samsung)
        - Any global stock!
        """)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Sources")
st.sidebar.markdown("""
1. **NSE India API** (Direct)
2. **BSE India API** (Direct)
3. **Yahoo Finance** (Global)
4. **Alpha Vantage** (Backup)
5. **Twelve Data** (Backup)
6. **Finnhub** (Backup)

**Multi-source fallback** ensures maximum coverage!
""")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Pro Tips")
st.sidebar.markdown("""
- Use correct exchange suffixes
- Generate 1000+ options at once
- Export to CSV for analysis
- Adjust IV per market conditions
- ATM options have highest gamma
""")

st.sidebar.markdown("---")
st.sidebar.success("✅ **Built from scratch** - No limitations!")
