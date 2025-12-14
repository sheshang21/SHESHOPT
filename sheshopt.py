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

def get_nse_session():
    """Create NSE session with proper cookies"""
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com/',
        'Connection': 'keep-alive'
    }
    session.headers.update(headers)
    try:
        session.get("https://www.nseindia.com/option-chain", timeout=10)
        time.sleep(2)
        return session
    except:
        return None

@st.cache_data(ttl=30)
def fetch_nse_options(symbol, is_index=False):
    """Fetch NSE options with improved error handling"""
    try:
        session = get_nse_session()
        if not session:
            return None, None, "Failed to establish NSE session"
        
        if is_index:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        
        response = session.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'records' not in data or 'data' not in data['records']:
                return None, None, "No option chain data available"
            
            options_data = []
            spot_price = data['records'].get('underlyingValue', 0)
            
            if spot_price == 0:
                return None, None, "Could not fetch underlying price"
            
            for item in data['records']['data']:
                expiry = item.get('expiryDate', '')
                strike = item.get('strikePrice', 0)
                
                if not expiry or strike == 0:
                    continue
                
                try:
                    exp_date = datetime.strptime(expiry, '%d-%b-%Y')
                except:
                    continue
                
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
                        'Change': ce.get('change', 0),
                        'Pct Change': ce.get('pchangeinOpenInterest', 0),
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
                        'Change': pe.get('change', 0),
                        'Pct Change': pe.get('pchangeinOpenInterest', 0),
                        'Bid': pe.get('bidprice', 0),
                        'Ask': pe.get('askPrice', 0),
                        'Volume': pe.get('totalTradedVolume', 0),
                        'OI': pe.get('openInterest', 0),
                        'Change in OI': pe.get('changeinOpenInterest', 0),
                        'IV': pe.get('impliedVolatility', 0),
                        'ITM': spot_price < strike
                    })
            
            if not options_data:
                return None, None, "No options contracts found in data"
            
            return pd.DataFrame(options_data), spot_price, None
        
        elif response.status_code == 401:
            return None, None, "NSE API: Unauthorized - Try again in a few seconds"
        elif response.status_code == 403:
            return None, None, "NSE API: Access forbidden - Rate limited"
        else:
            return None, None, f"NSE API returned status code: {response.status_code}"
    
    except requests.exceptions.Timeout:
        return None, None, "Request timeout - NSE server slow"
    except requests.exceptions.ConnectionError:
        return None, None, "Connection error - Check internet"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

@st.cache_data(ttl=30)
def fetch_yahoo_options(symbol):
    """Fetch Yahoo Finance options"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'optionChain' not in data or 'result' not in data['optionChain']:
                return None, None, "No option chain data"
            
            result = data['optionChain']['result']
            if not result or 'options' not in result[0]:
                return None, None, f"No options available for {symbol}"
            
            result = result[0]
            quote = result.get('quote', {})
            spot_price = quote.get('regularMarketPrice', 0)
            
            if spot_price == 0:
                return None, None, "Could not fetch stock price"
            
            options_data = []
            
            for option in result['options']:
                exp_timestamp = option.get('expirationDate', 0)
                exp_date = datetime.fromtimestamp(exp_timestamp)
                exp_label = exp_date.strftime("%d-%b-%Y")
                days_to_expiry = (exp_date - datetime.now()).days
                
                for call in option.get('calls', []):
                    options_data.append({
                        'Option Code': f"{symbol} {exp_label} {call.get('strike', 0):.2f} CE",
                        'Symbol': symbol,
                        'Spot Price': spot_price,
                        'Strike': call.get('strike', 0),
                        'Type': 'CALL',
                        'Expiry': exp_label,
                        'Days to Expiry': days_to_expiry,
                        'LTP': call.get('lastPrice', 0),
                        'Change': call.get('change', 0),
                        'Pct Change': call.get('percentChange', 0),
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
                        'Option Code': f"{symbol} {exp_label} {put.get('strike', 0):.2f} PE",
                        'Symbol': symbol,
                        'Spot Price': spot_price,
                        'Strike': put.get('strike', 0),
                        'Type': 'PUT',
                        'Expiry': exp_label,
                        'Days to Expiry': days_to_expiry,
                        'LTP': put.get('lastPrice', 0),
                        'Change': put.get('change', 0),
                        'Pct Change': put.get('percentChange', 0),
                        'Bid': put.get('bid', 0),
                        'Ask': put.get('ask', 0),
                        'Volume': put.get('volume', 0),
                        'OI': put.get('openInterest', 0),
                        'Change in OI': 0,
                        'IV': put.get('impliedVolatility', 0) * 100,
                        'ITM': put.get('inTheMoney', False)
                    })
            
            if not options_data:
                return None, None, "No option contracts found"
            
            return pd.DataFrame(options_data), spot_price, None
        else:
            return None, None, f"Yahoo returned status: {response.status_code}"
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def add_theoretical_greeks(df, r=0.065):
    """Add Greeks to dataframe"""
    for idx, row in df.iterrows():
        S = row['Spot Price']
        K = row['Strike']
        T = max(row['Days to Expiry'], 0) / 365.0
        sigma = row['IV'] / 100 if row['IV'] > 0 else 0.30
        opt_type = 'call' if row['Type'] == 'CALL' else 'put'
        
        theo = black_scholes(S, K, T, r, sigma, opt_type)
        intrinsic = max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
        greeks = calculate_greeks(S, K, T, r, sigma, opt_type)
        
        df.at[idx, 'Theoretical'] = theo
        df.at[idx, 'Intrinsic'] = intrinsic
        df.at[idx, 'Time Value'] = max(theo - intrinsic, 0)
        df.at[idx, 'Delta'] = greeks['delta']
        df.at[idx, 'Gamma'] = greeks['gamma']
        df.at[idx, 'Theta'] = greeks['theta']
        df.at[idx, 'Vega'] = greeks['vega']
    
    return df

st.title("📊 Real Options Data from Exchanges")

st.info("🔥 **Working Data Sources:** Yahoo Finance (US/Global stocks) works reliably. NSE may have intermittent access issues due to their API restrictions.")

tab1, tab2, tab3 = st.tabs(["🇮🇳 NSE Stocks", "📈 NSE Indices", "🌍 US & Global"])

with tab1:
    st.subheader("NSE Stock Options")
    col1, col2 = st.columns([3, 1])
    with col1:
        nse_stock = st.text_input("NSE Stock Symbol", "RELIANCE", help="Examples: RELIANCE, TCS, INFY, SBIN, HDFCBANK").upper()
    with col2:
        st.write("")
        st.write("")
        nse_stock_btn = st.button("Fetch NSE Stock", type="primary", key="btn1")
    
    if nse_stock_btn:
        with st.spinner(f"Fetching {nse_stock} options from NSE..."):
            df, spot, error = fetch_nse_options(nse_stock, is_index=False)
            
            if df is not None:
                st.success(f"✅ Fetched {len(df)} options! Spot: ₹{spot:.2f}")
                df = add_theoretical_greeks(df)
                st.session_state['df'] = df
                st.session_state['spot'] = spot
            else:
                st.error(f"❌ {error}")
                st.info("💡 NSE API has strict access controls. If this fails, try: 1) Wait 30 seconds and retry, 2) Use US stocks tab instead, 3) Clear cache and retry")

with tab2:
    st.subheader("NSE Index Options")
    col1, col2 = st.columns([3, 1])
    with col1:
        nse_index = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
    with col2:
        st.write("")
        st.write("")
        nse_index_btn = st.button("Fetch Index", type="primary", key="btn2")
    
    if nse_index_btn:
        with st.spinner(f"Fetching {nse_index} options..."):
            df, spot, error = fetch_nse_options(nse_index, is_index=True)
            
            if df is not None:
                st.success(f"✅ Fetched {len(df)} options! Index: ₹{spot:.2f}")
                df = add_theoretical_greeks(df)
                st.session_state['df'] = df
                st.session_state['spot'] = spot
            else:
                st.error(f"❌ {error}")

with tab3:
    st.subheader("US & Global Options (Yahoo Finance)")
    st.success("✅ This source works reliably!")
    col1, col2 = st.columns([3, 1])
    with col1:
        yahoo_sym = st.text_input("Symbol", "AAPL", help="Examples: AAPL, TSLA, MSFT, GOOGL, SPY, QQQ, NVDA").upper()
    with col2:
        st.write("")
        st.write("")
        yahoo_btn = st.button("Fetch Yahoo", type="primary", key="btn3")
    
    if yahoo_btn:
        with st.spinner(f"Fetching {yahoo_sym} options..."):
            df, spot, error = fetch_yahoo_options(yahoo_sym)
            
            if df is not None:
                st.success(f"✅ Fetched {len(df)} options! Spot: ${spot:.2f}")
                df = add_theoretical_greeks(df)
                st.session_state['df'] = df
                st.session_state['spot'] = spot
            else:
                st.error(f"❌ {error}")
                st.info("💡 Make sure the symbol has listed options. Try popular stocks like AAPL, TSLA, SPY, QQQ")

if 'df' in st.session_state and st.session_state['df'] is not None:
    st.markdown("---")
    st.subheader("📈 Options Chain")
    
    df = st.session_state['df']
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        search = st.text_input("🔍 Search", placeholder="Strike, expiry, code...")
    with col2:
        filt_type = st.selectbox("Type", ["All", "CALL", "PUT"])
    with col3:
        filt_itm = st.selectbox("Money", ["All", "ITM", "OTM"])
    with col4:
        filt_exp = st.selectbox("Expiry", ["All"] + sorted(df['Expiry'].unique().tolist()))
    
    filtered = df.copy()
    
    if search:
        mask = (
            filtered['Option Code'].str.contains(search, case=False, na=False) |
            filtered['Strike'].astype(str).str.contains(search, na=False)
        )
        filtered = filtered[mask]
    
    if filt_type != "All":
        filtered = filtered[filtered['Type'] == filt_type]
    
    if filt_itm == "ITM":
        filtered = filtered[filtered['ITM']]
    elif filt_itm == "OTM":
        filtered = filtered[~filtered['ITM']]
    
    if filt_exp != "All":
        filtered = filtered[filtered['Expiry'] == filt_exp]
    
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
        st.metric("Spot", f"₹{st.session_state.get('spot', 0):.2f}")
    with col6:
        total_vol = filtered['Volume'].sum()
        st.metric("Total Vol", f"{total_vol:,.0f}")
    
    st.dataframe(
        filtered,
        use_container_width=True,
        height=600,
        column_config={
            "Spot Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Strike": st.column_config.NumberColumn(format="₹%.0f"),
            "LTP": st.column_config.NumberColumn(format="₹%.2f"),
            "Change": st.column_config.NumberColumn(format="₹%.2f"),
            "Bid": st.column_config.NumberColumn(format="₹%.2f"),
            "Ask": st.column_config.NumberColumn(format="₹%.2f"),
            "Theoretical": st.column_config.NumberColumn(format="₹%.2f"),
            "Intrinsic": st.column_config.NumberColumn(format="₹%.2f"),
            "Time Value": st.column_config.NumberColumn(format="₹%.2f"),
            "IV": st.column_config.NumberColumn(format="%.2f%%"),
            "Delta": st.column_config.NumberColumn(format="%.4f"),
            "Gamma": st.column_config.NumberColumn(format="%.5f"),
            "Theta": st.column_config.NumberColumn(format="%.4f"),
            "Vega": st.column_config.NumberColumn(format="%.4f"),
            "Volume": st.column_config.NumberColumn(format="%d"),
            "OI": st.column_config.NumberColumn(format="%d"),
        }
    )
    
    csv = filtered.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, f"options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Sources")
st.sidebar.markdown("""
**Working:**
- ✅ Yahoo Finance (US/Global)

**Intermittent:**
- ⚠️ NSE API (strict access)

**Tip:** Use Yahoo Finance tab for reliable access to US stocks with options.
""")

st.sidebar.markdown("---")
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")
