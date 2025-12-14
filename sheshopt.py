import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import requests
import json
import time

st.set_page_config(page_title="Real Options Data", page_icon="📊", layout="wide")

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
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    delta = norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    theta_c = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d1-sigma*np.sqrt(T)))/365
    theta_p = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-(d1-sigma*np.sqrt(T))))/365
    theta = theta_c if opt_type == 'call' else theta_p
    vega = S*norm.pdf(d1)*np.sqrt(T)/100
    rho = K*T*np.exp(-r*T)*(norm.cdf(d1-sigma*np.sqrt(T)) if opt_type=='call' else -norm.cdf(-(d1-sigma*np.sqrt(T))))/100
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

@st.cache_data(ttl=60)
def fetch_nse_options(symbol):
    """Fetch REAL NSE options data"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/'
        }
        
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)
        
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'records' in data and 'data' in data['records']:
                options_data = []
                spot_price = data['records'].get('underlyingValue', 0)
                
                for item in data['records']['data']:
                    expiry = item.get('expiryDate', '')
                    strike = item.get('strikePrice', 0)
                    
                    exp_date = datetime.strptime(expiry, '%d-%b-%Y') if expiry else datetime.now()
                    days_to_expiry = (exp_date - datetime.now()).days
                    
                    if 'CE' in item:
                        ce = item['CE']
                        options_data.append({
                            'Option Code': f"{symbol} {expiry} {strike} CE",
                            'Symbol': symbol,
                            'Spot Price': spot_price,
                            'Strike': strike,
                            'Type': 'CALL',
                            'Expiry': expiry,
                            'Days to Expiry': days_to_expiry,
                            'LTP': ce.get('lastPrice', 0),
                            'Bid': ce.get('bidprice', 0),
                            'Ask': ce.get('askPrice', 0),
                            'Volume': ce.get('totalTradedVolume', 0),
                            'OI': ce.get('openInterest', 0),
                            'Change in OI': ce.get('changeinOpenInterest', 0),
                            'IV': ce.get('impliedVolatility', 0),
                            'ITM': spot_price > strike
                        })
                    
                    if 'PE' in item:
                        pe = item['PE']
                        options_data.append({
                            'Option Code': f"{symbol} {expiry} {strike} PE",
                            'Symbol': symbol,
                            'Spot Price': spot_price,
                            'Strike': strike,
                            'Type': 'PUT',
                            'Expiry': expiry,
                            'Days to Expiry': days_to_expiry,
                            'LTP': pe.get('lastPrice', 0),
                            'Bid': pe.get('bidprice', 0),
                            'Ask': pe.get('askPrice', 0),
                            'Volume': pe.get('totalTradedVolume', 0),
                            'OI': pe.get('openInterest', 0),
                            'Change in OI': pe.get('changeinOpenInterest', 0),
                            'IV': pe.get('impliedVolatility', 0),
                            'ITM': spot_price < strike
                        })
                
                return pd.DataFrame(options_data), spot_price
    except Exception as e:
        st.error(f"NSE Error: {str(e)}")
    
    return None, None

@st.cache_data(ttl=60)
def fetch_nse_index_options(symbol):
    """Fetch REAL NSE Index options (NIFTY, BANKNIFTY, etc.)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/'
        }
        
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(1)
        
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'records' in data and 'data' in data['records']:
                options_data = []
                spot_price = data['records'].get('underlyingValue', 0)
                
                for item in data['records']['data']:
                    expiry = item.get('expiryDate', '')
                    strike = item.get('strikePrice', 0)
                    
                    exp_date = datetime.strptime(expiry, '%d-%b-%Y') if expiry else datetime.now()
                    days_to_expiry = (exp_date - datetime.now()).days
                    
                    if 'CE' in item:
                        ce = item['CE']
                        options_data.append({
                            'Option Code': f"{symbol} {expiry} {strike} CE",
                            'Symbol': symbol,
                            'Spot Price': spot_price,
                            'Strike': strike,
                            'Type': 'CALL',
                            'Expiry': expiry,
                            'Days to Expiry': days_to_expiry,
                            'LTP': ce.get('lastPrice', 0),
                            'Bid': ce.get('bidprice', 0),
                            'Ask': ce.get('askPrice', 0),
                            'Volume': ce.get('totalTradedVolume', 0),
                            'OI': ce.get('openInterest', 0),
                            'Change in OI': ce.get('changeinOpenInterest', 0),
                            'IV': ce.get('impliedVolatility', 0),
                            'ITM': spot_price > strike
                        })
                    
                    if 'PE' in item:
                        pe = item['PE']
                        options_data.append({
                            'Option Code': f"{symbol} {expiry} {strike} PE",
                            'Symbol': symbol,
                            'Spot Price': spot_price,
                            'Strike': strike,
                            'Type': 'PUT',
                            'Expiry': expiry,
                            'Days to Expiry': days_to_expiry,
                            'LTP': pe.get('lastPrice', 0),
                            'Bid': pe.get('bidprice', 0),
                            'Ask': pe.get('askPrice', 0),
                            'Volume': pe.get('totalTradedVolume', 0),
                            'OI': pe.get('openInterest', 0),
                            'Change in OI': pe.get('changeinOpenInterest', 0),
                            'IV': pe.get('impliedVolatility', 0),
                            'ITM': spot_price < strike
                        })
                
                return pd.DataFrame(options_data), spot_price
    except Exception as e:
        st.error(f"NSE Index Error: {str(e)}")
    
    return None, None

@st.cache_data(ttl=60)
def fetch_yahoo_options(symbol):
    """Fetch REAL options from Yahoo Finance (US & Global)"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'optionChain' in data and 'result' in data['optionChain'] and data['optionChain']['result']:
                result = data['optionChain']['result'][0]
                
                if 'options' not in result or not result['options']:
                    return None, None
                
                quote = result.get('quote', {})
                spot_price = quote.get('regularMarketPrice', 0)
                
                options_data = []
                
                for option in result['options']:
                    exp_timestamp = option.get('expirationDate', 0)
                    exp_date = datetime.fromtimestamp(exp_timestamp)
                    exp_label = exp_date.strftime("%d-%b-%Y")
                    days_to_expiry = (exp_date - datetime.now()).days
                    
                    for call in option.get('calls', []):
                        options_data.append({
                            'Option Code': f"{symbol} {exp_label} {call.get('strike', 0)} CE",
                            'Symbol': symbol,
                            'Spot Price': spot_price,
                            'Strike': call.get('strike', 0),
                            'Type': 'CALL',
                            'Expiry': exp_label,
                            'Days to Expiry': days_to_expiry,
                            'LTP': call.get('lastPrice', 0),
                            'Bid': call.get('bid', 0),
                            'Ask': call.get('ask', 0),
                            'Volume': call.get('volume', 0),
                            'OI': call.get('openInterest', 0),
                            'Change in OI': 0,
                            'IV': call.get('impliedVolatility', 0) * 100,
                            'ITM': call.get('inTheMoney', False)
                        })
                    
                    for put in option.get('puts', []):
                        options_data.append({
                            'Option Code': f"{symbol} {exp_label} {put.get('strike', 0)} PE",
                            'Symbol': symbol,
                            'Spot Price': spot_price,
                            'Strike': put.get('strike', 0),
                            'Type': 'PUT',
                            'Expiry': exp_label,
                            'Days to Expiry': days_to_expiry,
                            'LTP': put.get('lastPrice', 0),
                            'Bid': put.get('bid', 0),
                            'Ask': put.get('ask', 0),
                            'Volume': put.get('volume', 0),
                            'OI': put.get('openInterest', 0),
                            'Change in OI': 0,
                            'IV': put.get('impliedVolatility', 0) * 100,
                            'ITM': put.get('inTheMoney', False)
                        })
                
                return pd.DataFrame(options_data), spot_price
    except Exception as e:
        st.error(f"Yahoo Error: {str(e)}")
    
    return None, None

def add_theoretical_greeks(df, r=0.065):
    """Add theoretical prices and Greeks to real options data"""
    for idx, row in df.iterrows():
        S = row['Spot Price']
        K = row['Strike']
        T = row['Days to Expiry'] / 365.0
        sigma = row['IV'] / 100 if row['IV'] > 0 else 0.30
        opt_type = 'call' if row['Type'] == 'CALL' else 'put'
        
        theo = black_scholes(S, K, T, r, sigma, opt_type)
        intrinsic = max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
        greeks = calculate_greeks(S, K, T, r, sigma, opt_type)
        
        df.at[idx, 'Theoretical'] = theo
        df.at[idx, 'Intrinsic'] = intrinsic
        df.at[idx, 'Time Value'] = theo - intrinsic
        df.at[idx, 'Delta'] = greeks['delta']
        df.at[idx, 'Gamma'] = greeks['gamma']
        df.at[idx, 'Theta'] = greeks['theta']
        df.at[idx, 'Vega'] = greeks['vega']
    
    return df

st.title("📊 REAL Options Data - Live from Exchanges")
st.markdown("**Fetch actual traded options from NSE, BSE, and global exchanges**")

st.warning("⚠️ **IMPORTANT:** This fetches REAL options contracts that are actually traded on exchanges. No fake/generated data!")

tab1, tab2, tab3 = st.tabs(["🇮🇳 NSE Stocks", "📈 NSE Indices (NIFTY/BANKNIFTY)", "🌍 US & Global (Yahoo)"])

with tab1:
    st.subheader("NSE Stock Options")
    st.info("Examples: RELIANCE, TCS, INFY, SBIN, HDFCBANK, ICICIBANK, etc.")
    
    nse_symbol = st.text_input("Enter NSE Stock Symbol", "RELIANCE", key="nse_stock").upper()
    
    if st.button("Fetch NSE Stock Options", type="primary", key="btn_nse"):
        with st.spinner(f"Fetching REAL options for {nse_symbol} from NSE..."):
            df, spot = fetch_nse_options(nse_symbol)
            
            if df is not None and not df.empty:
                st.success(f"✅ Fetched {len(df)} REAL traded options! Spot: ₹{spot:.2f}")
                df = add_theoretical_greeks(df)
                st.session_state['options_df'] = df
                st.session_state['spot'] = spot
            else:
                st.error("❌ No options data found. Check symbol or try again.")

with tab2:
    st.subheader("NSE Index Options")
    st.info("Examples: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY")
    
    index_symbol = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"], key="nse_index")
    
    if st.button("Fetch Index Options", type="primary", key="btn_index"):
        with st.spinner(f"Fetching REAL options for {index_symbol} from NSE..."):
            df, spot = fetch_nse_index_options(index_symbol)
            
            if df is not None and not df.empty:
                st.success(f"✅ Fetched {len(df)} REAL traded options! Spot: ₹{spot:.2f}")
                df = add_theoretical_greeks(df)
                st.session_state['options_df'] = df
                st.session_state['spot'] = spot
            else:
                st.error("❌ No options data found. Try again.")

with tab3:
    st.subheader("US & Global Options (via Yahoo Finance)")
    st.info("Examples: AAPL, TSLA, MSFT, GOOGL, SPY, QQQ, NVDA, etc.")
    
    yahoo_symbol = st.text_input("Enter Symbol", "AAPL", key="yahoo").upper()
    
    if st.button("Fetch Yahoo Options", type="primary", key="btn_yahoo"):
        with st.spinner(f"Fetching REAL options for {yahoo_symbol}..."):
            df, spot = fetch_yahoo_options(yahoo_symbol)
            
            if df is not None and not df.empty:
                st.success(f"✅ Fetched {len(df)} REAL traded options! Spot: ${spot:.2f}")
                df = add_theoretical_greeks(df)
                st.session_state['options_df'] = df
                st.session_state['spot'] = spot
            else:
                st.error("❌ No options data. Symbol may not have listed options.")

if 'options_df' in st.session_state and st.session_state['options_df'] is not None:
    st.markdown("---")
    st.subheader("📈 Live Options Chain")
    
    df = st.session_state['options_df']
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search = st.text_input("🔍 Search", placeholder="Strike, expiry, code...")
    with col2:
        filter_type = st.selectbox("Type", ["All", "CALL", "PUT"])
    with col3:
        filter_itm = st.selectbox("Money", ["All", "ITM", "OTM"])
    
    filtered = df.copy()
    
    if search:
        mask = (
            filtered['Option Code'].str.contains(search, case=False, na=False) |
            filtered['Strike'].astype(str).str.contains(search, na=False) |
            filtered['Expiry'].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    
    if filter_type != "All":
        filtered = filtered[filtered['Type'] == filter_type]
    
    if filter_itm == "ITM":
        filtered = filtered[filtered['ITM'] == True]
    elif filter_itm == "OTM":
        filtered = filtered[filtered['ITM'] == False]
    
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
        st.metric("Spot", f"₹{st.session_state.get('spot', 0):.2f}")
    
    st.dataframe(
        filtered.style.format({
            'Spot Price': '₹{:.2f}',
            'Strike': '₹{:.0f}',
            'LTP': '₹{:.2f}',
            'Bid': '₹{:.2f}',
            'Ask': '₹{:.2f}',
            'Theoretical': '₹{:.2f}',
            'Intrinsic': '₹{:.2f}',
            'Time Value': '₹{:.2f}',
            'IV': '{:.2f}%',
            'Delta': '{:.4f}',
            'Gamma': '{:.5f}',
            'Theta': '{:.4f}',
            'Vega': '{:.4f}',
            'Volume': '{:,.0f}',
            'OI': '{:,.0f}',
            'Change in OI': '{:,.0f}'
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

st.sidebar.markdown("---")
st.sidebar.subheader("✅ Real Data Sources")
st.sidebar.markdown("""
**This app fetches ACTUAL options:**

1. **NSE India** - Direct from NSE API
   - Stock options
   - Index options (NIFTY, BANKNIFTY)
   
2. **Yahoo Finance** - Global coverage
   - US stocks (AAPL, TSLA, etc.)
   - ETFs (SPY, QQQ)
   - All markets

**All data is REAL** - no generated/fake options!
""")

st.sidebar.markdown("---")
st.sidebar.success("✅ 100% Real Market Data!")
