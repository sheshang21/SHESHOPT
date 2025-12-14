import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import requests
import json
import time

st.set_page_config(page_title="Options Pricing", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);}
    h1 {color: #60a5fa;}
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
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta_c = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d1-sigma*np.sqrt(T)))/365
    theta_p = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-(d1-sigma*np.sqrt(T))))/365
    theta = theta_c if opt_type == 'call' else theta_p
    vega = S*norm.pdf(d1)*np.sqrt(T)/100
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

@st.cache_data(ttl=60)
def fetch_stock_price_multiple_sources(symbol):
    """Try multiple sources to get stock price"""
    
    # Try Yahoo Finance v8
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://finance.yahoo.com/',
            'Origin': 'https://finance.yahoo.com'
        }
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                price = data['chart']['result'][0]['meta'].get('regularMarketPrice', 0)
                if price > 0:
                    return price, None
    except:
        pass
    
    # Try Yahoo v10
    try:
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=price"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'quoteSummary' in data and 'result' in data['quoteSummary']:
                result = data['quoteSummary']['result'][0]
                price = result.get('price', {}).get('regularMarketPrice', {}).get('raw', 0)
                if price > 0:
                    return price, None
    except:
        pass
    
    return None, "Could not fetch price from any source"

@st.cache_data(ttl=60)
def fetch_options_yahoo_v1(symbol):
    """Method 1: Yahoo v7 API"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://finance.yahoo.com/',
            'Origin': 'https://finance.yahoo.com'
        }
        
        url = f"https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if 'optionChain' in data and 'result' in data['optionChain'] and data['optionChain']['result']:
                result = data['optionChain']['result'][0]
                spot = result.get('quote', {}).get('regularMarketPrice', 0)
                expiries = result.get('expirationDates', [])
                
                if spot > 0 and expiries:
                    all_opts = []
                    
                    for exp_ts in expiries[:6]:  # Get first 6 expiries to avoid timeout
                        try:
                            url_exp = f"https://query1.finance.yahoo.com/v7/finance/options/{symbol}?date={exp_ts}"
                            resp_exp = requests.get(url_exp, headers=headers, timeout=10)
                            
                            if resp_exp.status_code == 200:
                                data_exp = resp_exp.json()
                                opts = data_exp['optionChain']['result'][0].get('options', [])
                                
                                for opt in opts:
                                    exp_date = datetime.fromtimestamp(exp_ts)
                                    exp_label = exp_date.strftime("%d-%b-%Y")
                                    dte = (exp_date - datetime.now()).days
                                    
                                    for c in opt.get('calls', []):
                                        all_opts.append({
                                            'Code': f"{symbol} {exp_label} {c.get('strike',0):.1f} CE",
                                            'Symbol': symbol, 'Spot': spot, 'Strike': c.get('strike',0),
                                            'Type': 'CALL', 'Expiry': exp_label, 'DTE': dte,
                                            'LTP': c.get('lastPrice',0), 'Bid': c.get('bid',0), 'Ask': c.get('ask',0),
                                            'Volume': c.get('volume',0), 'OI': c.get('openInterest',0),
                                            'IV': c.get('impliedVolatility',0)*100, 'ITM': c.get('inTheMoney',False)
                                        })
                                    
                                    for p in opt.get('puts', []):
                                        all_opts.append({
                                            'Code': f"{symbol} {exp_label} {p.get('strike',0):.1f} PE",
                                            'Symbol': symbol, 'Spot': spot, 'Strike': p.get('strike',0),
                                            'Type': 'PUT', 'Expiry': exp_label, 'DTE': dte,
                                            'LTP': p.get('lastPrice',0), 'Bid': p.get('bid',0), 'Ask': p.get('ask',0),
                                            'Volume': p.get('volume',0), 'OI': p.get('openInterest',0),
                                            'IV': p.get('impliedVolatility',0)*100, 'ITM': p.get('inTheMoney',False)
                                        })
                        except:
                            continue
                    
                    if all_opts:
                        return pd.DataFrame(all_opts), spot, None
        
        return None, None, f"Yahoo v1 failed: HTTP {resp.status_code}"
    except Exception as e:
        return None, None, f"Yahoo v1 error: {str(e)}"

@st.cache_data(ttl=60)
def fetch_options_yahoo_v2(symbol):
    """Method 2: Alternative Yahoo endpoint"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://finance.yahoo.com/',
            'Origin': 'https://finance.yahoo.com',
            'Connection': 'keep-alive'
        }
        
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if 'optionChain' in data and 'result' in data['optionChain'] and data['optionChain']['result']:
                result = data['optionChain']['result'][0]
                spot = result.get('quote', {}).get('regularMarketPrice', 0)
                
                if spot > 0 and 'options' in result and result['options']:
                    all_opts = []
                    
                    for opt in result['options'][:3]:  # First 3 expiries
                        exp_ts = opt.get('expirationDate', 0)
                        exp_date = datetime.fromtimestamp(exp_ts)
                        exp_label = exp_date.strftime("%d-%b-%Y")
                        dte = (exp_date - datetime.now()).days
                        
                        for c in opt.get('calls', []):
                            all_opts.append({
                                'Code': f"{symbol} {exp_label} {c.get('strike',0):.1f} CE",
                                'Symbol': symbol, 'Spot': spot, 'Strike': c.get('strike',0),
                                'Type': 'CALL', 'Expiry': exp_label, 'DTE': dte,
                                'LTP': c.get('lastPrice',0), 'Bid': c.get('bid',0), 'Ask': c.get('ask',0),
                                'Volume': c.get('volume',0), 'OI': c.get('openInterest',0),
                                'IV': c.get('impliedVolatility',0)*100, 'ITM': c.get('inTheMoney',False)
                            })
                        
                        for p in opt.get('puts', []):
                            all_opts.append({
                                'Code': f"{symbol} {exp_label} {p.get('strike',0):.1f} PE",
                                'Symbol': symbol, 'Spot': spot, 'Strike': p.get('strike',0),
                                'Type': 'PUT', 'Expiry': exp_label, 'DTE': dte,
                                'LTP': p.get('lastPrice',0), 'Bid': p.get('bid',0), 'Ask': p.get('ask',0),
                                'Volume': p.get('volume',0), 'OI': p.get('openInterest',0),
                                'IV': p.get('impliedVolatility',0)*100, 'ITM': p.get('inTheMoney',False)
                            })
                    
                    if all_opts:
                        return pd.DataFrame(all_opts), spot, None
        
        return None, None, f"Yahoo v2 failed: HTTP {resp.status_code}"
    except Exception as e:
        return None, None, f"Yahoo v2 error: {str(e)}"

def generate_synthetic_options(symbol, spot, num_expiries=4, r=0.05, sigma=0.30):
    """Generate synthetic options with theoretical pricing when APIs fail"""
    
    strikes = []
    atm = round(spot)
    step = max(5, round(spot * 0.025))  # 2.5% steps
    
    for i in range(-10, 11):
        strikes.append(atm + i*step)
    
    expiries = []
    for i in range(num_expiries):
        days = 7 + (i * 30)
        exp_date = datetime.now() + timedelta(days=days)
        expiries.append((exp_date, exp_date.strftime("%d-%b-%Y"), days))
    
    opts = []
    
    for exp_date, exp_label, dte in expiries:
        T = dte / 365.0
        
        for K in strikes:
            for opt_type, opt_label in [('call', 'CE'), ('put', 'PE')]:
                theo = black_scholes(spot, K, T, r, sigma, opt_type)
                intrinsic = max(spot-K, 0) if opt_type=='call' else max(K-spot, 0)
                greeks = calculate_greeks(spot, K, T, r, sigma, opt_type)
                
                # Simulate bid/ask spread
                spread = max(0.05, theo * 0.02)
                bid = max(0, theo - spread)
                ask = theo + spread
                
                opts.append({
                    'Code': f"{symbol} {exp_label} {K} {opt_label}",
                    'Symbol': symbol, 'Spot': spot, 'Strike': K,
                    'Type': opt_type.upper(), 'Expiry': exp_label, 'DTE': dte,
                    'LTP': theo, 'Bid': bid, 'Ask': ask,
                    'Volume': 0, 'OI': 0, 'IV': sigma*100,
                    'ITM': (opt_type=='call' and spot>K) or (opt_type=='put' and spot<K),
                    'Theoretical': theo, 'Intrinsic': intrinsic,
                    'Time Value': max(theo-intrinsic, 0),
                    'Delta': greeks['delta'], 'Gamma': greeks['gamma'],
                    'Theta': greeks['theta'], 'Vega': greeks['vega']
                })
    
    return pd.DataFrame(opts)

def add_greeks(df, r=0.05):
    """Add Greeks to real options data"""
    for idx, row in df.iterrows():
        S, K = row['Spot'], row['Strike']
        T = max(row['DTE'], 0) / 365.0
        sigma = row['IV']/100 if row['IV']>0 else 0.30
        opt_type = 'call' if row['Type']=='CALL' else 'put'
        
        theo = black_scholes(S, K, T, r, sigma, opt_type)
        intrinsic = max(S-K, 0) if opt_type=='call' else max(K-S, 0)
        greeks = calculate_greeks(S, K, T, r, sigma, opt_type)
        
        df.at[idx, 'Theoretical'] = theo
        df.at[idx, 'Intrinsic'] = intrinsic
        df.at[idx, 'Time Value'] = max(theo-intrinsic, 0)
        df.at[idx, 'Delta'] = greeks['delta']
        df.at[idx, 'Gamma'] = greeks['gamma']
        df.at[idx, 'Theta'] = greeks['theta']
        df.at[idx, 'Vega'] = greeks['vega']
    
    return df

st.title("📊 Options Pricing & Analysis")

st.info("💡 **How it works:** Tries to fetch real market data. If APIs are blocked, generates theoretical options with Black-Scholes pricing and full Greeks.")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    symbol = st.text_input("Stock Symbol", "AAPL", help="Enter any stock symbol").upper()

with col2:
    st.write("")
    st.write("")
    use_synthetic = st.checkbox("Force Theoretical Mode", value=False)

with col3:
    st.write("")
    st.write("")
    fetch_btn = st.button("🚀 Get Options", type="primary", use_container_width=True)

if fetch_btn:
    with st.spinner(f"Processing {symbol}..."):
        
        if not use_synthetic:
            # Try real data sources
            st.info("Attempting to fetch real market data...")
            
            df, spot, err1 = fetch_options_yahoo_v1(symbol)
            
            if df is None:
                st.warning(f"Method 1 failed: {err1}")
                df, spot, err2 = fetch_options_yahoo_v2(symbol)
                
                if df is None:
                    st.warning(f"Method 2 failed: {err2}")
                    
                    # Try to at least get the spot price
                    spot, price_err = fetch_stock_price_multiple_sources(symbol)
                    
                    if spot:
                        st.info(f"Got spot price: ${spot:.2f}. Generating theoretical options...")
                        df = generate_synthetic_options(symbol, spot)
                        st.session_state['mode'] = 'theoretical'
                    else:
                        st.error("Could not fetch any data. Using estimated price.")
                        df = generate_synthetic_options(symbol, 100)
                        spot = 100
                        st.session_state['mode'] = 'estimated'
                else:
                    st.success(f"✅ Got {len(df)} real options from Method 2!")
                    df = add_greeks(df)
                    st.session_state['mode'] = 'real'
            else:
                st.success(f"✅ Got {len(df)} real options from Method 1!")
                df = add_greeks(df)
                st.session_state['mode'] = 'real'
        else:
            # Synthetic mode
            spot, _ = fetch_stock_price_multiple_sources(symbol)
            if not spot:
                spot = 100
            df = generate_synthetic_options(symbol, spot)
            st.session_state['mode'] = 'theoretical'
            st.success(f"✅ Generated {len(df)} theoretical options!")
        
        st.session_state['df'] = df
        st.session_state['spot'] = spot
        st.session_state['symbol'] = symbol

if 'df' in st.session_state:
    st.markdown("---")
    
    mode = st.session_state.get('mode', 'unknown')
    if mode == 'real':
        st.success("📊 **Real Market Data** - Actual traded options with live prices")
    elif mode == 'theoretical':
        st.info("🧮 **Theoretical Mode** - Black-Scholes pricing with full Greeks")
    elif mode == 'estimated':
        st.warning("⚠️ **Estimated Mode** - Using estimated price and theoretical options")
    
    df = st.session_state['df']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search = st.text_input("🔍 Search", placeholder="Strike, expiry...")
    with col2:
        type_f = st.selectbox("Type", ["All", "CALL", "PUT"])
    with col3:
        itm_f = st.selectbox("Money", ["All", "ITM", "OTM"])
    with col4:
        exp_f = st.selectbox("Expiry", ["All"] + sorted(df['Expiry'].unique().tolist()))
    
    filtered = df.copy()
    
    if search:
        mask = filtered['Code'].str.contains(search, case=False, na=False) | filtered['Strike'].astype(str).str.contains(search, na=False)
        filtered = filtered[mask]
    
    if type_f != "All":
        filtered = filtered[filtered['Type']==type_f]
    
    if itm_f == "ITM":
        filtered = filtered[filtered['ITM']]
    elif itm_f == "OTM":
        filtered = filtered[~filtered['ITM']]
    
    if exp_f != "All":
        filtered = filtered[filtered['Expiry']==exp_f]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total", f"{len(filtered):,}")
    with col2:
        st.metric("Calls", f"{len(filtered[filtered['Type']=='CALL']):,}")
    with col3:
        st.metric("Puts", f"{len(filtered[filtered['Type']=='PUT']):,}")
    with col4:
        st.metric("ITM", f"{len(filtered[filtered['ITM']]):,}")
    with col5:
        st.metric("Spot", f"${st.session_state['spot']:.2f}")
    
    st.dataframe(filtered, use_container_width=True, height=600)
    
    csv = filtered.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, f"options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.markdown("""
**This app:**
1. Tries multiple APIs for real data
2. Falls back to theoretical pricing
3. Always shows full Greeks
4. Works even when APIs fail

**Theoretical mode includes:**
- Black-Scholes pricing
- Delta, Gamma, Theta, Vega
- Intrinsic & Time Value
- Realistic bid/ask spreads
""")

if st.sidebar.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.success("Cleared!")
