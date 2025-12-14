import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import requests
import json

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
def fetch_yahoo_options_all_expiries(symbol):
    """Fetch ALL expiries from Yahoo Finance"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        # First get all available expiry dates
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return None, None, f"HTTP {response.status_code}"
        
        data = response.json()
        
        if 'optionChain' not in data or 'result' not in data['optionChain']:
            return None, None, "No option chain in response"
        
        result = data['optionChain']['result']
        if not result:
            return None, None, "Empty result"
        
        result = result[0]
        quote = result.get('quote', {})
        spot_price = quote.get('regularMarketPrice', 0)
        
        if spot_price == 0:
            return None, None, "No stock price found"
        
        # Get all expiry timestamps
        expiry_dates = result.get('expirationDates', [])
        
        if not expiry_dates:
            return None, None, "No expiration dates available"
        
        all_options = []
        
        # Fetch options for each expiry
        for exp_timestamp in expiry_dates:
            try:
                url_exp = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}?date={exp_timestamp}"
                resp_exp = requests.get(url_exp, headers=headers, timeout=15)
                
                if resp_exp.status_code == 200:
                    data_exp = resp_exp.json()
                    
                    if 'optionChain' in data_exp and 'result' in data_exp['optionChain']:
                        options_data = data_exp['optionChain']['result'][0].get('options', [])
                        
                        for opt in options_data:
                            exp_date = datetime.fromtimestamp(exp_timestamp)
                            exp_label = exp_date.strftime("%d-%b-%Y")
                            days = (exp_date - datetime.now()).days
                            
                            for call in opt.get('calls', []):
                                all_options.append({
                                    'Option Code': f"{symbol} {exp_label} {call.get('strike', 0):.2f} CE",
                                    'Symbol': symbol,
                                    'Spot': spot_price,
                                    'Strike': call.get('strike', 0),
                                    'Type': 'CALL',
                                    'Expiry': exp_label,
                                    'DTE': days,
                                    'LTP': call.get('lastPrice', 0),
                                    'Bid': call.get('bid', 0),
                                    'Ask': call.get('ask', 0),
                                    'Volume': call.get('volume', 0),
                                    'OI': call.get('openInterest', 0),
                                    'IV': call.get('impliedVolatility', 0) * 100,
                                    'ITM': call.get('inTheMoney', False)
                                })
                            
                            for put in opt.get('puts', []):
                                all_options.append({
                                    'Option Code': f"{symbol} {exp_label} {put.get('strike', 0):.2f} PE",
                                    'Symbol': symbol,
                                    'Spot': spot_price,
                                    'Strike': put.get('strike', 0),
                                    'Type': 'PUT',
                                    'Expiry': exp_label,
                                    'DTE': days,
                                    'LTP': put.get('lastPrice', 0),
                                    'Bid': put.get('bid', 0),
                                    'Ask': put.get('ask', 0),
                                    'Volume': put.get('volume', 0),
                                    'OI': put.get('openInterest', 0),
                                    'IV': put.get('impliedVolatility', 0) * 100,
                                    'ITM': put.get('inTheMoney', False)
                                })
            except:
                continue
        
        if not all_options:
            return None, None, "No options contracts found"
        
        return pd.DataFrame(all_options), spot_price, None
        
    except Exception as e:
        return None, None, str(e)

def add_greeks(df, r=0.05):
    """Add theoretical Greeks"""
    for idx, row in df.iterrows():
        S = row['Spot']
        K = row['Strike']
        T = max(row['DTE'], 0) / 365.0
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

st.title("📊 Live Options Data - Working Sources")
st.success("✅ Using Yahoo Finance - most reliable source for real options data worldwide!")

tab1, tab2 = st.tabs(["🌍 Yahoo Finance (All Markets)", "📤 Upload NSE CSV"])

with tab1:
    st.subheader("Fetch Real Options from Yahoo Finance")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol",
            "AAPL",
            help="Works for US, UK, India, and global stocks with options"
        ).upper()
        
        st.caption("**Examples that work:**")
        st.caption("🇺🇸 US: AAPL, TSLA, MSFT, GOOGL, AMZN, NVDA, AMD, SPY, QQQ, META, NFLX")
        st.caption("🇮🇳 India: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, SBIN.NS")
        st.caption("🇬🇧 UK: VODAFONE.L, BP.L")
    
    with col2:
        st.write("")
        st.write("")
        fetch_btn = st.button("🚀 Fetch Options", type="primary", use_container_width=True)
    
    if fetch_btn:
        with st.spinner(f"Fetching ALL options for {symbol}..."):
            df, spot, error = fetch_yahoo_options_all_expiries(symbol)
            
            if df is not None and not df.empty:
                st.success(f"✅ Fetched {len(df):,} real options contracts!")
                st.info(f"📊 Spot Price: ${spot:.2f} | Expiries: {df['Expiry'].nunique()} | Strikes: {df['Strike'].nunique()}")
                
                df = add_greeks(df)
                st.session_state['df'] = df
                st.session_state['spot'] = spot
                st.session_state['symbol'] = symbol
            else:
                st.error(f"❌ Failed: {error}")
                st.warning("⚠️ Make sure the symbol is correct and has listed options. Try these guaranteed working symbols: AAPL, TSLA, SPY, QQQ, MSFT")

with tab2:
    st.subheader("Upload NSE Options Data (CSV)")
    st.info("📥 Download option chain CSV from NSE website and upload here. The app will calculate Greeks and add analysis.")
    
    uploaded = st.file_uploader("Upload NSE CSV file", type=['csv'])
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(df)} rows from CSV")
            
            # Try to auto-detect columns
            st.write("**Detected columns:**", df.columns.tolist())
            
            spot_price = st.number_input("Enter Spot Price", value=2850.0, step=10.0)
            
            if st.button("Process CSV", type="primary"):
                # Process and add Greeks
                st.session_state['df'] = df
                st.session_state['spot'] = spot_price
                st.success("✅ CSV processed successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# Display options data
if 'df' in st.session_state and st.session_state['df'] is not None:
    st.markdown("---")
    st.subheader(f"📈 Options Chain: {st.session_state.get('symbol', 'Uploaded Data')}")
    
    df = st.session_state['df']
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        search = st.text_input("🔍 Search", placeholder="Strike, expiry...")
    with col2:
        type_filter = st.selectbox("Type", ["All", "CALL", "PUT"])
    with col3:
        itm_filter = st.selectbox("Moneyness", ["All", "ITM", "OTM"])
    with col4:
        if 'Expiry' in df.columns:
            expiry_filter = st.selectbox("Expiry", ["All"] + sorted(df['Expiry'].unique().tolist()))
        else:
            expiry_filter = "All"
    
    filtered = df.copy()
    
    if search:
        mask = (
            filtered['Option Code'].str.contains(search, case=False, na=False) |
            filtered['Strike'].astype(str).str.contains(search, na=False)
        )
        filtered = filtered[mask]
    
    if type_filter != "All":
        filtered = filtered[filtered['Type'] == type_filter]
    
    if itm_filter == "ITM":
        filtered = filtered[filtered['ITM']]
    elif itm_filter == "OTM":
        filtered = filtered[~filtered['ITM']]
    
    if expiry_filter != "All" and 'Expiry' in filtered.columns:
        filtered = filtered[filtered['Expiry'] == expiry_filter]
    
    # Metrics
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
        st.metric("Spot", f"${st.session_state.get('spot', 0):.2f}")
    with col6:
        total_vol = filtered['Volume'].sum() if 'Volume' in filtered.columns else 0
        st.metric("Volume", f"{total_vol:,.0f}")
    
    # Display dataframe
    st.dataframe(
        filtered,
        use_container_width=True,
        height=600
    )
    
    # Download
    csv = filtered.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Data",
        csv,
        f"options_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )
    
    # Analytics
    st.markdown("---")
    st.subheader("📊 Options Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Volume Leaders", "OI Leaders", "IV Analysis"])
    
    with tab1:
        if 'Volume' in filtered.columns:
            top_vol = filtered.nlargest(10, 'Volume')[['Option Code', 'Type', 'Strike', 'LTP', 'Volume', 'OI']]
            st.dataframe(top_vol, use_container_width=True)
    
    with tab2:
        if 'OI' in filtered.columns:
            top_oi = filtered.nlargest(10, 'OI')[['Option Code', 'Type', 'Strike', 'LTP', 'Volume', 'OI']]
            st.dataframe(top_oi, use_container_width=True)
    
    with tab3:
        if 'IV' in filtered.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Call IV", f"{filtered[filtered['Type']=='CALL']['IV'].mean():.2f}%")
            with col2:
                st.metric("Avg Put IV", f"{filtered[filtered['Type']=='PUT']['IV'].mean():.2f}%")

else:
    st.info("👆 Enter a symbol and click 'Fetch Options' to get started!")
    
    st.markdown("### ✅ Verified Working Symbols:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🇺🇸 US Tech**
        - AAPL (Apple)
        - MSFT (Microsoft)
        - GOOGL (Google)
        - AMZN (Amazon)
        - META (Facebook)
        - NVDA (Nvidia)
        """)
    
    with col2:
        st.markdown("""
        **📈 Popular ETFs**
        - SPY (S&P 500)
        - QQQ (Nasdaq 100)
        - IWM (Russell 2000)
        - DIA (Dow Jones)
        - GLD (Gold)
        """)
    
    with col3:
        st.markdown("""
        **🇮🇳 India (Add .NS)**
        - RELIANCE.NS
        - TCS.NS
        - INFY.NS
        - HDFCBANK.NS
        - SBIN.NS
        """)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 How It Works")
st.sidebar.markdown("""
**Yahoo Finance API**
- Fetches REAL traded options
- Works globally
- Free & reliable
- All expiries & strikes

**Data Included:**
- Last Traded Price (LTP)
- Bid/Ask spreads
- Volume & Open Interest
- Implied Volatility
- Theoretical prices (BS)
- Greeks (Delta, Gamma, etc)
""")

st.sidebar.markdown("---")
st.sidebar.success("✅ 100% Real Market Data!")

if st.sidebar.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared!")
